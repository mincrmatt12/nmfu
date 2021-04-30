#!/usr/bin/env python
"""
NMFU - the "no memory for you" "parser" generator.

designed to create what a compsci major would yell at me for calling a dfa to parse files/protocols character by character while using as little
RAM as possible.

Copyright (C) 2020 Matthew Mirvish

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

__version__ = "0.3.0a1"

import abc
import enum
import string
import itertools
import queue
import io
import textwrap
import os
import sys
try:
    import lark
except ImportError:
    print("... failed to import lark; must be in setup.py ...", file=sys.stderr)
    # Mock it
    class lark:
        Token = None
        Tree = None
        
        @staticmethod
        def Lark(*args, **kwargs):
            return None
        
        LarkError = RuntimeError
from collections import defaultdict
from typing import List, Optional, Iterable, Dict, Union, Set
try:
    import graphviz
    debug_enabled = True
    import lark.tree
except ImportError:
    debug_enabled = False

grammar = r"""
start: top_decl* parser_decl top_decl*

?top_decl: out_decl
         | macro_decl
         | hook_decl

out_decl: "out" out_type IDENTIFIER ";"
        | "out" out_type IDENTIFIER "=" atom ";"

out_type: "bool" -> bool_type
        | "int" -> int_type
        | "enum" "{" IDENTIFIER ("," IDENTIFIER)+ "}" -> enum_type
        | "str" "[" NUMBER "]" -> str_type

hook_decl: "hook" IDENTIFIER ";"

macro_decl: "macro" IDENTIFIER "{" statement* "}"

parser_decl: "parser" "{" statement+ "}"

?statement: block_stmt
          | simple_stmt ";"

simple_stmt: expr -> match_stmt
           | IDENTIFIER "=" expr -> assign_stmt
           | IDENTIFIER "+=" expr -> append_stmt
           | IDENTIFIER "(" (expr ("," expr)*)? ")" -> call_stmt
           | "break" IDENTIFIER? -> break_stmt
           | "finish" -> finish_stmt
           | "wait" expr -> wait_stmt

block_stmt: "loop" IDENTIFIER? "{" statement+ "}" -> loop_stmt
          | "case" "{" case_clause+ "}" -> case_stmt
          | "optional" "{" statement+ "}" -> optional_stmt
          | "try" "{" statement+ "}" catch_block -> try_stmt

catch_block: "catch" catch_options? "{" statement* "}"

case_clause: case_predicate ("," case_predicate)* "->" "{" statement* "}"
case_predicate: "else" -> else_predicate
              | expr -> expr_predicate

catch_options: "(" CATCH_OPTION ("," CATCH_OPTION)* ")" 

// EXPRESSIONS

?expr: atom // either literal or string depending on context
     | regex // not an atom to simplify things
     | "end" -> end_expr
     | "(" expr+ ")" -> concat_expr
     | "[" sum_expr "]"

atom: BOOL_CONST -> bool_const
    | NUMBER -> number_const
    | STRING "i" -> string_case_const
    | STRING -> string_const
    | IDENTIFIER -> identifier_const

?sum_expr: mul_expr (SUM_OP mul_expr)*
?mul_expr: math_atom (MUL_OP math_atom)*
?math_atom: NUMBER -> math_num
          | IDENTIFIER -> math_var
          | "$" IDENTIFIER -> builtin_math_var
          | "(" sum_expr ")"

// REGEX

regex: "/" regex_component* "/"

regex_char_class: "\\" REGEX_CHARCLASS

?regex_component: REGEX_UNIMPORTANT -> regex_raw_match // creates multiple char classes
                | regex_char_class // creates a char class
                | regex_component REGEX_OP -> regex_operation
                | "(" regex_component+ ")" -> regex_group
                | regex_component "{" NUMBER "}" -> regex_exact_repeat
                | regex_component "{" NUMBER "," NUMBER "}" -> regex_range_repeat
                | regex_component "{" NUMBER "," "}" -> regex_at_least_repeat
                | "[^" regex_set_element+ "]" -> regex_inverted_set // creates an inverted char class
                | "[" regex_set_element+ "]" -> regex_set // creates a char class
                | "." -> regex_any // creates an inverted char class
                | regex_component "|" regex_component ("|" regex_component)* -> regex_alternation

?regex_set_element: REGEX_CHARGROUP_ELEMENT_RAW
                  | REGEX_CHARGROUP_ELEMENT_RAW "-" REGEX_CHARGROUP_ELEMENT_RAW -> regex_set_range
                  | regex_char_class

// BINARY REGEX

// TERMINALS
BOOL_CONST: /true|false/

// catch options
CATCH_OPTION: /nomatch|outofspace/

%import common.CNAME -> IDENTIFIER
%import common.SIGNED_INT -> NUMBER

STRING: /"(?:[^"\\]|\\.)*"/

// regex internals
REGEX_UNIMPORTANT: /[^.?*()\[\]\\+{}|\/]|\\\.|\\\*|\\\(|\\\)|\\\[|\\\]|\\\+|\\\\|\\\{|\\\}|\\\||\\\//
REGEX_OP: /[+*?]/
REGEX_CHARGROUP_ELEMENT_RAW: /[^\-\]\\\/]|\\-|\\\]|\\\\|\\\//
REGEX_CHARCLASS: /[wWdDsSntr ]/

// math
SUM_OP: /[+-]/
MUL_OP: /[*\/%]/

// comments

%import common.WS
COMMENT: /\s*\/\/[^\n]*/

%ignore WS
%ignore COMMENT
"""

parser = lark.Lark(grammar, propagate_positions=True, lexer="dynamic_complete")

"""
NMFU operates in a few 'stages':

- 1. conversion to actual AST (removes variable assignments and turns them into actions)
- 2a. conversion to state machine recursively with abstract actions
- 2b. (optional) state machine optimizing
- 3. codegen

The overall architecture is similar to MLang, with various context classes which each steal the resources from a
predecessor ctx.
"""

# ===========
# STATE TYPES
# ===========

class DFTransition:
    """
    A transition

    on_values is a set of characters (or integers)
    """

    # Special tokens
    Else = type('_DFElse', (), {'__repr__': lambda x: "Else"})()
    End = type('_DFEnd', (), {'__repr__': lambda x: "End"})()

    def __init__(self, on_values=None, conditions=None, fallthrough=False):
        if conditions is None:
            self.conditions = []
        else:
            self.conditions = conditions

        if type(on_values) is not list:
            if type(on_values) is str or on_values in [DFTransition.Else, DFTransition.End]:
                on_values = [on_values]
            elif on_values is None:
                on_values = []
            else:
                on_values = list(on_values)

        self.on_values = on_values
        self.target = None
        self.is_fallthrough = fallthrough
        self.actions = []

    def copy(self):
        my_copy = DFTransition()
        my_copy.on_values = self.on_values.copy()
        my_copy.conditions = self.conditions.copy()
        my_copy.actions = self.actions.copy()
        my_copy.target = self.target
        my_copy.is_fallthrough = self.is_fallthrough
        return my_copy

    def __repr__(self):
        if self.actions:
            return f"<DFTransition on={self.on_values} to={self.target} actions={self.actions} fallthough={self.is_fallthrough}>"
        else:
            return f"<DFTransition on={self.on_values} to={self.target} fallthough={self.is_fallthrough}>"

    def restrict(self, *conditions):
        self.conditions.extend(conditions)
        return self

    def attach(self, *actions, prepend=False):
        if prepend:
            self.actions = list(actions) + self.actions
        else:
            self.actions.extend(actions)
        return self
    
    def to(self, target):
        if isinstance(target, int):
            self.target = DFState.all_states[target]
        else:
            self.target = target
        return self

    def fallthrough(self, fall=True):
        self.is_fallthrough = fall
        return self

    @classmethod
    def from_key(cls, on_values, conditions, inherited):
        return cls(on_values, conditions).to(inherited.target).attach(*inherited.actions).fallthrough(inherited.is_fallthrough)

class DFState:
    all_states = {}

    def __init__(self):
        self.transitions = []
        DFState.all_states[id(self)] = self

    def transition(self, transition, allow_replace=False, allow_replace_if=None):
        # Add parent relationship if this is a new transition
        if ProgramData.lookup(transition, DebugTag.PARENT) is None:
            ProgramData.imbue(transition, DebugTag.PARENT, self)
        if allow_replace_if is None:
            allow_replace = lambda x: allow_replace
        else:
            allow_replace = allow_replace_if

        contained = None
        for i in self.all_transitions():
            if i.conditions == transition.conditions and (set(i.on_values) & set(transition.on_values)):
                contained = i
                break
        if contained is not None and contained.target != transition.target:
            if allow_replace(contained):
                for x in transition.on_values:
                    try:
                        contained.on_values.remove(x)
                    except ValueError:
                        continue
                if not contained.on_values:
                    self.transitions.remove(contained)
            else:
                raise IllegalDFAStateError("Duplicate transition", self)
        elif contained is not None:
            return # don't add duplicate entries in the table

        # Check if we can "inline" this transition
        for transition_in in self.transitions:
            if transition_in.conditions == transition.conditions and transition_in.actions == transition.actions and transition_in.target == transition.target and transition_in.is_fallthrough == transition.is_fallthrough:
                if DFTransition.Else in transition_in.on_values or DFTransition.Else in transition.on_values:
                    transition_in.on_values = [DFTransition.Else]
                else:
                    transition_in.on_values = list(set(transition_in.on_values) | set(transition.on_values))
                return self
        self.transitions.append(transition)
        return self

    def all_transitions(self) -> Iterable[DFTransition]:
        yield from self.transitions

    def all_transitions_for(self, on_values, conditions=None):
        if conditions is None:
            conditions = []

        for i in self.transitions:
            if len(set(i.on_values) & set(on_values)) and i.conditions == conditions:
                yield i

    def __setitem__(self, key, data):
        if type(key) in (tuple, list):
            on_values, conditions = key
        else:
            on_values, conditions = key, []

        if type(data) is DFTransition:
            self.transition(DFTransition.from_key(on_values, conditions, data), True)
        else:
            self.transition(DFTransition(on_values, conditions).to(data), True)

    def __delitem__(self, key):
        if type(key) in (tuple, list):
            on_values, conditions = key
        else:
            on_values, conditions = key, []
        contained = None
        for i in self.all_transitions():
            if i.conditions == conditions and (set(i.on_values) & set(on_values)):
                contained = i
                break
        if contained is not None:
            if type(on_values) in (set, frozenset):
                for on_value in on_values:
                    contained.on_values.remove(on_value)
            else:
                contained.on_values.remove(on_values)
            if not contained.on_values:
                self.transitions.remove(contained)

    def __getitem__(self, data):
        """
        Get the transition that would be followed for data, including Else if applicable
        """
        if type(data) in (tuple, list):
            for i in self.all_transitions():
                if not i.on_values:
                    continue
                if (data[0] in i.on_values) if type(data[0]) is not set else data[0] <= set(i.on_values) and data[1] == i.conditions:
                    return i
            return self[DFTransition.Else]
        elif type(data) in (set, frozenset):
            for i in self.all_transitions():
                if not i.on_values:
                    continue
                if data <= set(i.on_values):
                    return i
                elif data > set(i.on_values):
                    return None
            return self[DFTransition.Else]
        else:
            for i in self.all_transitions():
                if data in i.on_values:
                    return i
            if DFTransition.Else != data:
                return self[DFTransition.Else]

class DFA:
    all_state_machines = {}

    def __init__(self):
        self.accepting_states = []
        self.starting_state = None
        self.states: List[DFState] = []

    def add(self, state):
        if ProgramData.lookup(state, DebugTag.PARENT) is None:
            ProgramData.imbue(state, DebugTag.PARENT, self)
        if self.starting_state == None:
            self.starting_state = state
        self.states.append(state)

    def mark_accepting(self, state):
        if isinstance(state, int):
            self.accepting_states.append(DFState.all_states[state])
        else:
            self.accepting_states.append(state)

    def simulate(self, actions, condition_states=None):
        """
        Attempt to simulate what would happen given the input states.
        Conditions are treated as always false, although a dictionary of {condition: value} can be provided
        """

        if condition_states is None:
            condition_states = [defaultdict(lambda: False) for x in actions]

        position = self.starting_state
        for action, _ in zip(actions, condition_states):
            # TODO: conditions
            try:
                while True:
                    if position is None:
                        return None
                    transition = position[action]
                    position = transition.target
                    if not transition.is_fallthrough:
                        break
            except AttributeError:
                return None
            else:
                if not position:
                    return position

        return position

    def dfs(self):
        """
        Construct a dfs-order traversal of the DFA
        """

        visited = set()

        def aux(state):
            if not state:
                return
            if state in visited:
                return
            visited.add(state)

            for t in state.all_transitions():
                use_real = True
                for action in t.actions:
                    if action.get_target_override_mode() == ActionOverrideMode.ALWAYS_GOTO_OTHER:
                        use_real = False
                        aux(action.get_target_override_target())
                    elif action.get_target_override_mode() == ActionOverrideMode.ALWAYS_GOTO_UNDEFINED:
                        use_real = False
                    elif action.get_target_override_mode() == ActionOverrideMode.MAY_GOTO_TARGET:
                        aux(action.get_target_override_target())
                    else:
                        continue
                    break
                if use_real:
                    aux(t.target)

        aux(self.starting_state)
        return visited


    def transitions_pointing_to(self, target_state: DFState, include_states=False):
        """
        Return a set of all transitions that point to the target state that are reachable
        from the start state
        """

        result = set()

        for state in self.dfs():
            for t in state.all_transitions():
                if t.target == target_state:
                    if include_states:
                        result.add((state, t))
                    else:
                        result.add(t)

        return result

    def transitions_that_do(self, action: "Action"):
        """
        Return a set of all transitions that contain the action and that are reachable
        from the start state
        """

        result = set()

        for state in self.dfs():
            for t in state.all_transitions():
                if action in t.actions:
                    result.add(t)

        return result

    def is_valid(self):
        """
        Can we still reach at least one accept state?
        """

        for state in self.dfs():
            if state in self.accepting_states:
                return True

        return False

    def append_after(self, chained_dfa: "DFA", treat_as_else: Union[DFState, List[DFState]], sub_states=None, mark_accept=True, chain_actions=None):
        """
        Add the chained_dfa in such a way that its start state "becomes" all of the `sub_states` (or if unspecified, the accept states of _this_ dfa)
        """

        if sub_states is None:
            sub_states = self.accepting_states

        if type(treat_as_else) is DFState:
            treat_as_else = (treat_as_else,)

        # First, add transitions between each of the sub_states corresponding to the transitions of the original starting node
        for sub_state in sub_states:
            for transition in chained_dfa.starting_state.transitions: # specifically exclude the else transition
                try:
                    if not chain_actions:
                        sub_state.transition(transition.copy(), allow_replace_if=lambda x: x.target in treat_as_else)
                    else:
                        sub_state.transition(transition.copy().attach(*chain_actions, prepend=True), allow_replace_if=lambda x: x.target in treat_as_else)
                except IllegalDFAStateError as e:
                    # Check if this is a transition to an else case
                    if transition.target in treat_as_else:
                        # If the sub_state already has something pointing there, it probably deals with properly
                        if any(x.target in treat_as_else for x in sub_state.transitions):
                            # ignore the error
                            continue
                    # Rewrite the error as something more useful
                    raise IllegalDFAStateConflictsError("Ambigious transitions detected while joining matches", sub_state, chained_dfa.starting_state) from e

        # Adopt all the states
        while chained_dfa.states:
            self.add(chained_dfa.states.pop())

        # Finally, mark all the sub_states as no longer accept states, and mark the new accept states thus
        if mark_accept:
            sub_states = sub_states.copy()
            for sub_state in sub_states:
                if sub_state in self.accepting_states:
                    self.accepting_states.remove(sub_state)
            # Additionally, if the starting state was an accept state, mark all the sub states as accept states too
            if chained_dfa.starting_state in chained_dfa.accepting_states:
                for sub_state in sub_states:
                    self.mark_accepting(sub_state)

            for state in chained_dfa.accepting_states:
                self.mark_accepting(state)


# =============
# DEBUG STORAGE
# =============

class DebugTag(enum.Enum):
    NAME = 0
    SOURCE_LINE = 1
    SOURCE_COLUMN = 2
    PARENT = 3

class ProgramFlag(int, enum.Enum):
    def __new__(cls, value, helpstr="", default=False, implies=(), exclusive_with=()):
        obj = int.__new__(cls, value)
        obj.default = default
        obj.helpstr = helpstr
        obj._value_ = value
        obj.implies = frozenset(implies)
        obj.exclusive_with = frozenset(exclusive_with)
        return obj

    # Verbosity options (enabled by various levels of -v or, ofc, -f)
    VERBOSE_REGEX_CCLASS = 0
    VERBOSE_OPTIMIZE_RESULTS = 1
    VERBOSE_SIMPLIFY_TM = 2

    # Optimization flags, set in ProgramData
    # DFA optimization options (enabled by various levels of -O)
    SIMPLIFY_ELSE_CONDITIONS = 3
    REMOVE_INACCESIBLE_STATES = 4

    # Codegen optimization options ('')
    COLLAPSE_TRANSITION_RANGES = 9

    # Codegen options 
    # |
    # - Structure options
    DYNAMIC_MEMORY = 7  # Allow use of dynamic memory
    INCLUDE_USER_PTR = (11, "Add a void* to the state structure, useful with hooks")
    STRICT_DONE_TOKEN_GENERATION = (15, "Only return DONE from _feed at accept states, not from transitions to accept states")
    EOF_SUPPORT = (16, "Generate a end function and support detecting eofs")
    # |
    # - String storage options
    ALLOCATE_STR_SPACE_IN_STRUCT = (5, "Allocate string space in struct", True, (), (6,))  # allocate the string space in the struct as an array, default
    ALLOCATE_STR_SPACE_DYNAMIC   = (6, "Allocate string space dynamically", False, (7,), (5,))  # implies DYNAMIC
    ALLOCATE_STR_SPACE_DYNAMIC_ON_DEMAND = (8, "Allocate string space on demand", False, (6,))  # implies DYNAMIC
    # |
    # - Hook options
    HOOK_GLOBAL = (12, "Place hooks as global functions", True)
    HOOK_PER_STATE = (13, "Place hooks as function pointers in the state structure", False, (), (12,))
    # |
    # - Template options
    USE_PRAGMA_ONCE = (10, "Use #pragma once instead of an #ifndef guard in the header")
    USE_CPLUSPLUS_GUARD = (14, "Include a __cplusplus extern C guard", True)

class ProgramOption(enum.Enum):
    def __init__(self, default, helpstr):
        self.default = default
        self.helpstr = helpstr

    # DFA optimization options

    # Codegen options
    COLLAPSED_RANGE_LENGTH = (4, "Minimum length of range to collapse into range comparison")

class HasDefaultDebugInfo:
    def debug_lookup(self, tag: DebugTag):
        return None

class DebugDumpable(enum.Enum):
    AST = "ast"
    DFA = "dfa"
    TRACEBACK = "traceback"

class ProgramData:
    _collection = defaultdict(dict)
    _children = defaultdict(list)
    _current_source = []
    _flags = {
            x: x.default for x in ProgramFlag
    }

    _options = {
            x: x.default for x in ProgramOption
    }
    _dump = []

    dump_prefix = None
    dry_run = False

    _OPTIMIZE_LEVELS = {
        0: (),     # -O0 (nothing)
        1: (ProgramFlag.SIMPLIFY_ELSE_CONDITIONS, ProgramFlag.REMOVE_INACCESIBLE_STATES),       # -O1 (the default)
        2: (ProgramFlag.COLLAPSE_TRANSITION_RANGES,),     #- O2 (adds to 1, 2)
        3: (),
    }

    @classmethod
    def load_source(cls, src: str):
        """
        Load the currently processing source code
        """

        cls._current_source = src.splitlines(keepends=False)

    @classmethod
    def imbue(cls, obj: object, tag: DebugTag, value: object):
        """
        Imbue this object with this debug information
        """

        if tag == DebugTag.PARENT:
            ProgramData._children[id(value.__repr__.__self__)].append(obj)
        ProgramData._collection[id(obj)][tag] = value
        return obj

    @classmethod
    def lookup(cls, obj: object, tag: DebugTag, recurse_upwards=True):
        """
        Find the imbued object's data, or -- if none exists -- find it's parents
        """

        if type(obj) in (lark.Token, lark.Tree):
            if tag == DebugTag.SOURCE_LINE:
                return obj.line
            elif tag == DebugTag.SOURCE_COLUMN:
                return obj.column
            else:
                return None

        id_obj = id(obj) if type(obj) is not int else obj

        if tag not in ProgramData._collection[id_obj]:
            if DebugTag.PARENT in ProgramData._collection[id_obj] and recurse_upwards:
                val = cls.lookup(ProgramData._collection[id_obj][DebugTag.PARENT], tag)
                if val is not None:
                    return val
            if isinstance(obj, HasDefaultDebugInfo):
                val = obj.debug_lookup(tag)
                if val is not None:
                    return val
            if tag in (DebugTag.SOURCE_COLUMN, DebugTag.SOURCE_LINE):
                for i in cls._children[id_obj]:
                    val = cls.lookup(i, tag, recurse_upwards=False)
                    if val is not None:
                        return val
            return None
        return ProgramData._collection[id_obj][tag]

    @classmethod
    def get_source_line(cls, line: int):
        line -= 1
        if line >= len(cls._current_source):
            return None
        else:
            return cls._current_source[line]
    
    @classmethod
    def _is_optimization_flag(cls, flag):
        for level in range(4):
            if flag in cls._OPTIMIZE_LEVELS[level]:
                return level
        return -1

    @classmethod
    def _print_version(cls):
        print("nmfu", __version__)
        print("Copyright (C) 2020 Matthew Mirvish")
        print("This is free software; see the source for copying conditions.  There is NO")
        print("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.")

    @classmethod
    def _print_help(cls):
        print("Usage: nmfu [options] input")
        print("")
        print("Global Options:")
        print("  -o<arg>, --output <arg>                    Output name without extension")
        print("  -O<level>                                  Optimization level (default: 1)")
        print("  -f<flag>, -fno-<flag>, --flag <flag>=<arg> Enable or disable a flag")
        print("  -d<arg>,<arg>, --dump <arg>,<arg>          Dump <args> to pdfs or stdout. Possible values are: " + ", ".join(x.value for x in DebugDumpable))
        print("  --dump-prefix <arg>                        Write dumped pdfs to files starting with <arg> (default is program name)")
        print("  -t, --dry-run                              Only convert the input to a DFA (and possibly dump), don't generate code")
        print("  -h, --help                                 Show this help screen")
        print("  --version                                  Show the version of nmfu")
        print("")
        print("Generation Options:")
        pad_length = 4 + len(max(ProgramOption, key=lambda x: len(x.name)).name) + 6
        for option in ProgramOption:
            flag_name = option.name.replace("_", "-").lower()
            opt_str = f"  --{flag_name} <arg>"
            print(f"{opt_str: <{pad_length}} {option.helpstr} (default: {option.default})")
        print("")
        print("Flags:")
        pad_length = 2 + len(max((y for y in ProgramFlag if not y.name.startswith("VERBOSE") and cls._is_optimization_flag(y) == -1), key=lambda x: len(x.name)).name)
        for flag in ProgramFlag:
            if flag.name.startswith("VERBOSE"):
                continue
            if cls._is_optimization_flag(flag) >= 0:
                continue
            flag_name = flag.name.replace("_", "-").lower()
            opt_str = f"  {flag_name}"
            if flag.helpstr:
                print(f"{opt_str: <{pad_length}} {flag.helpstr} (default: {flag.default})")
            else:
                print(opt_str)
        print("")
        print("Optimization Flags:")
        pad_length = 2 + max(len(y.name) for y in ProgramFlag if cls._is_optimization_flag(y) >= 0)
        for flag in ProgramFlag:
            if cls._is_optimization_flag(flag) == -1:
                continue
            flag_name = flag.name.replace("_", "-").lower()
            opt_str = f"  {flag_name}"
            if flag.helpstr:
                print(f"{opt_str: <{pad_length}} {flag.helpstr} (enabled at level {cls._is_optimization_flag(flag)})")
            else:
                print(f"{opt_str: <{pad_length}} enabled at level {cls._is_optimization_flag(flag)}")


    @classmethod
    def load_commandline_flags(cls, all_cmd_options: List[str]):
        """
        Load the command line flags passed in. Returns a tuple of (input_filename, program_output_name)
        """

        input_filename = None
        program_output_name = None

        optimize_level = 1
        flag_overrides = {}

        all_cmd_options_iter = iter(all_cmd_options)
        cls.dump_prefix = None

        for option in all_cmd_options_iter:
            if not option:
                continue
            try:
                if option[0] != "-":
                    if input_filename is not None:
                        raise RuntimeError("Program filename specified multiple times")
                    input_filename = option
                    program_output_name = os.path.splitext(os.path.basename(input_filename))[0]
                    continue
                elif option[1] == "-":
                    option_name = option[2:]
                    if option_name not in ["help", "dry-run", "version"]:
                        option_value = next(all_cmd_options_iter)
                else:
                    option_name = option[1]
                    option_value = option[2:]
            except IndexError:
                raise RuntimeError("Invalid argument " + option)
            except StopIteration:
                raise RuntimeError("Missing value for argument " + option)

            if option_name in ["o", "output"]:
                if "." in option_value:
                    raise RuntimeError("Program output should not contain an extension")
                program_output_name = option_value
            elif option_name == "O":
                optimize_level = int(option_value)
            elif option_name in ["f", "flag"]:
                if option_name == "f":
                    set_to = True
                    if option_value.startswith("no-"):
                        set_to = False
                        option_value = option_value[3:]
                    flag_name = option_value.upper().replace("-", "_")
                else:
                    if "=" not in option_value:
                        set_to = True
                        flag_name = option_value
                    else:
                        flag_name, set_to = option_value.split("=")
                        set_to = set_to in ["yes", "on"]
                    option_value = flag_name
                    flag_name = flag_name.upper().replace("-", "_")
                if flag_name not in ProgramFlag.__members__:
                    raise RuntimeError("Unknown flag " + option_value)
                flag_overrides[ProgramFlag[flag_name]] = set_to
            elif option_name in ["h", "help"]:
                cls._print_help()
                exit(0)
            elif option_name == "version":
                cls._print_version()
                exit(0)
            elif option_name in ["d", "dump"]:
                for i in option_value.split(","):
                    cls._dump.append(DebugDumpable(i))
            elif option_name == "dump-prefix":
                cls.dump_prefix = option_value
            elif option_name in ["t", "dry-run"]:
                cls.dry_run = True
            else:
                p_option_name = option_name.upper().replace("-", "_")
                if p_option_name not in ProgramOption.__members__:
                    raise RuntimeError("Unknown option " + option_name)
                try:
                    cls._options[ProgramOption[p_option_name]] = type(ProgramOption[p_option_name].default)(option_value)
                except ValueError as e:
                    raise RuntimeError("Invalid value for option " + option_name) from e

        if input_filename is None:
            raise RuntimeError("No input file provided!")

        for j in range(optimize_level + 1):
            for i in cls._OPTIMIZE_LEVELS[j]:
                cls._flags[i] = True

        for k, v in flag_overrides.items():
            cls._flags[k] = v

        # Set implies
        while True:
            did_something = False
            for k, v in cls._flags.items():
                if v:
                    for x in k.implies:
                        if not cls._flags[x]:
                            did_something = True
                        cls._flags[x] = True
            if not did_something:
                break

        # Fix exclusives for only the user specified values
        def aux(flag):
            for conflict in flag.exclusive_with:
                conflict = ProgramFlag(conflict)
                if conflict in flag_overrides and flag_overrides[conflict]:
                    raise RuntimeError("Conflict between " + conflict.name + " and " + flag.name)
                elif cls._flags[conflict]:
                    # ensure it's set to false, and go through _its_ conflicts
                    cls._flags[conflict] = False
            # also investigate the things it implies as if they were specified
            for implies in flag.implies:
                implies = ProgramFlag(implies)
                aux(implies)

        for flag in flag_overrides.keys():
            if cls._flags[flag]:
                aux(flag)

        if cls.dump_prefix is None:
            cls.dump_prefix = program_output_name
        
        return (input_filename, program_output_name)

    @classmethod
    def do(self, flag):
        return self._flags[flag]

    @classmethod
    def option(self, opt):
        return self._options[opt]

    @classmethod
    def dump(self, dumpable):
        return dumpable in self._dump

class IndexableInstance(type):
    def __init__(self, name, bases, dct):
        self._ii_cache = {}

    def __getitem__(cls, obj):
        if obj in cls._ii_cache:
            return cls._ii_cache[obj]
        else:
            cls._ii_cache[obj] = cls(obj) # pylint: disable=no-value-for-parameter,no-value-for-parameter
            return cls._ii_cache[obj]

class dprint(metaclass=IndexableInstance):
    def __init__(self, condition):
        self.condition = condition

    def __call__(self, *args, **kwargs):
        if ProgramData.do(self.condition):
            print(*args, **kwargs)


# ===========
# ERROR TYPES
# ===========

class NMFUError(Exception):
    def __init__(self, reasons):
        self.reasons = reasons

    def _get_message(self, show_potential_reasons=True, reasons_header="Potential reasons include:"):
        info_strs = []
        for reason in self.reasons:
            name, line, column = (ProgramData.lookup(reason, tag) for tag in (DebugTag.NAME, DebugTag.SOURCE_LINE, DebugTag.SOURCE_COLUMN))
            info_str = ""
            if name:
                info_str += f"- {name}:"
                if line is not None:
                    info_str += f"\n  at line {line}:\n{ProgramData.get_source_line(line)}"
                    if column is not None:
                        info_str += "\n" + " " * (column - 1) + "^"
                else:
                    info_str = info_str[-1]
            else:
                if line is not None:
                    info_str += f"- line {line}:\n{ProgramData.get_source_line(line)}"
                    if column is not None:
                        info_str += "\n" + " " * (column - 1) + "^"
                else:
                    continue
            info_strs.append(info_str)

        if info_strs:
            return (f"{reasons_header}\n" if show_potential_reasons else "") + "\n".join(info_strs)
        else:
            return ""

    def __str__(self):
        return self._get_message()

class IllegalASTStateError(NMFUError):
    def __init__(self, msg, source):
        super().__init__([source])
        self.source = source
        self.msg = msg

    def __str__(self):
        return self.msg + "\n" + self._get_message(reasons_header="Due to:")

class IllegalDFAStateError(IllegalASTStateError):
    pass

class IllegalParseTree(IllegalASTStateError):
    pass

class IllegalIntExpr(IllegalASTStateError):
    pass

class IllegalDFAStateConflictsError(NMFUError):
    """
    Specifically for conflicts as opposed to an invalid state detected on a single thing
    """

    def __init__(self, msg, *source):
        super().__init__(source)
        self.source = source
        self.msg = msg

    def __str__(self):
        return self.msg + "\n" + self._get_message(reasons_header="Due to:")

class UndefinedReferenceError(NMFUError):
    def __init__(self, objtype, source: lark.Token):
        super().__init__([source])
        self.objtype = objtype
        self.source = source

    def __str__(self):
        return f"Undefined reference to {self.objtype} {self.source.value}:\n" + self._get_message(show_potential_reasons=False)

class DuplicateDefinitionError(NMFUError):
    def __init__(self, objtype, source, name):
        super().__init__([source])
        self.objtype = objtype
        self.source = source
        self.name = name

    def __str__(self):
        return f"Duplicate definition of {self.objtype} {self.name}:\n" + self._get_message(show_potential_reasons=False)


# =========
# AST TYPES
# =========

class ActionMode(enum.Enum):
    EACH_CHARACTER = 0 # Run this action for each character read as part of this match
    AT_FINISH = 1      # Run this action at the end of the current match (if possible, as the last transition into the accept state, otherwise at transition to next instruction)
    AT_START = 2       # Run this action at the start of the current match, regardless of whether or not it actually passes.

class ActionOverrideMode(enum.Enum):
    NONE = 0 # action does not override next state
    MAY_GOTO_TARGET = 1 # action may goto a different state
    ALWAYS_GOTO_OTHER = 2 # action will always go to a defined state
    ALWAYS_GOTO_UNDEFINED = 3 # action will always go to an undefined state (usually something like the global "PREMATURE_EXIT" state or whatever)

class ErrorReasons(enum.Enum):
    NO_MATCH = "nomatch"
    OUT_OF_SPACE = "outofspace"

# =======
# ACTIONS
# =======

class Action:
    """
    Represents a high-level action to take either upon:
        - matching a character
        - matching an entire AST node

    This same class is used both for transition actions
    and for match actions. The timing/mode information is used to determine where to apply this
    action.
    """

    def get_mode(self) -> ActionMode:
        return ActionMode.EACH_CHARACTER

    def is_timing_strict(self):
        """
        Should we error out with an "ambiguous timing of finish action" if the action is unable to be scheduled with a valid $last
        """
        return False

    def get_target_override_mode(self) -> ActionOverrideMode:
        return ActionOverrideMode.NONE

    def get_target_override_target(self):
        return None

class FinishAction(Action, HasDefaultDebugInfo):
    def get_mode(self):
        return ActionMode.AT_FINISH

    def is_timing_strict(self):
        return True

    def get_target_override_mode(self):
        return ActionOverrideMode.ALWAYS_GOTO_UNDEFINED

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return "exit action"

class CallHook(Action, HasDefaultDebugInfo):
    def __init__(self, name):
        self.name = name

    def get_mode(self):
        return ActionMode.AT_FINISH

    def is_timing_strict(self):
        return True

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return f"call to hook {self.name}"

class BreakAction(Action, HasDefaultDebugInfo):
    def __init__(self, refers_to):
        self.refers_to = refers_to

    def get_mode(self):
        return ActionMode.AT_FINISH

    def is_timing_strict(self):
        return True

    def get_target_override_mode(self):
        return ActionOverrideMode.ALWAYS_GOTO_UNDEFINED

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return "break action for {}".format(ProgramData.lookup(self.refers_to, DebugTag.NAME))

class AppendTo(Action, HasDefaultDebugInfo):
    def __init__(self, end_target, into_storage: "OutputStorage"):
        self.end_target = end_target
        self.into_storage = into_storage

    def get_target_override_mode(self):
        return ActionOverrideMode.MAY_GOTO_TARGET

    def get_target_override_target(self):
        return self.end_target

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return "append action ({})".format(ProgramData.lookup(self.into_storage, DebugTag.NAME))

class AppendCharTo(Action, HasDefaultDebugInfo):
    def __init__(self, end_target, append_value: "IntegerExpr", into_storage: "OutputStorage"):
        self.end_target = end_target
        self.append_value = append_value
        self.into_storage = into_storage

    def get_mode(self):
        return ActionMode.AT_FINISH

    def get_target_override_mode(self):
        return ActionOverrideMode.MAY_GOTO_TARGET

    def get_target_override_target(self):
        return self.end_target

    def is_timing_strict(self):
        return True

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return "append character action ({})".format(ProgramData.lookup(self.into_storage, DebugTag.NAME))

class SetTo(Action, HasDefaultDebugInfo):
    def __init__(self, value_expr: "IntegerExpr", into_storage: "OutputStorage"):
        self.into_storage = into_storage
        self.value_expr = value_expr
        ProgramData.imbue(value_expr, DebugTag.PARENT, self)

    def get_mode(self):
        return ActionMode.AT_FINISH

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            if self.value_expr.is_literal():
                return "set into {} {}".format(ProgramData.lookup(self.into_storage, DebugTag.NAME), self.value_expr.get_literal_result())
            else:
                return "set into {}".format(ProgramData.lookup(self.into_storage, DebugTag.NAME))

class SetToStr(Action, HasDefaultDebugInfo):
    def __init__(self, value_expr: str, into_storage: "OutputStorage"):
        self.into_storage = into_storage
        if into_storage.type != OutputStorageType.STR:
            raise IllegalASTStateError("SetToStr used without a string", self)
        self.value_expr = value_expr

    def get_mode(self):
        return ActionMode.AT_FINISH

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return "set into {} {!r}".format(ProgramData.lookup(self.into_storage, DebugTag.NAME), self.value_expr)


class Node(abc.ABC):
    """
    Represents an arbitrary AST node:
    - contains a pointer to the next node
    - processed bottom up
    - convertable to a full DFA
    """

    @abc.abstractmethod
    def set_next(self, next_node: "Node"):
        """
        Set the next node that follows this node.
        _MUST_ be called before convert _UNLESS_ there is no next node
        """

        pass

    @abc.abstractmethod
    def get_next(self) -> "Node":
        """
        Get the next node
        """
        pass

    @abc.abstractmethod
    def convert(self, current_error_handlers: dict) -> DFA:
        return None

class ActionNode(Node):
    """
    An ephemeral node which represents a single node. There is guaranteed to only be one of these at any time and
    only as the topmost node in a stack
    """

    def __init__(self, *actions):
        self.actions = list(actions)
        self.next = None

    def set_next(self, next_node):
        if isinstance(next_node, ActionNode):
            # adopt this node
            self.actions.extend(next_node.actions)
            self.next = next_node.get_next()
        else:
            # just set next
            self.next = next_node

    def get_next(self):
        return self.next

    def convert(self, *args):
        raise IllegalASTStateError("Action not inherited by a match left in AST", self)

class ActionSinkNode(Node):
    """
    Represents a node that can accept/adopt ActionNodes into itself, distributing them to appropriate
    Match objects (or deferring that process to sometime before conversion and hiding the ActionNodes from the sequence
    """

    @abc.abstractmethod
    def _set_next(self, actual_next_node: Node):
        """
        Called by our set_next when the next node is not an action type
        """
        pass

    @abc.abstractmethod
    def _adopt_actions(self, actions: List[Action]):
        pass

    def set_next(self, next_node):
        if isinstance(next_node, ActionNode):
            self._adopt_actions(next_node.actions)
            self._set_next(next_node.get_next())
        else:
            self._set_next(next_node)

class Match(abc.ABC):
    """
    Represents an arbitrary AST node which matches part(s) of the input.
    Directly corresponds to string literal and regex matches.

    - convertable to a normal, accept/nonaccept DFA with else clauses pointing to the appropriate states.
    """

    def __init__(self):
        self.start_actions = []
        self.finish_actions = []
        self.char_actions = []

    @abc.abstractmethod
    def convert(self, current_error_handlers: dict) -> DFA:
        return None

    def attach(self, action: Action):
        ProgramData.imbue(action, DebugTag.PARENT, self)
        if action.get_mode() == ActionMode.AT_FINISH:
            self.finish_actions.append(action)
        elif action.get_mode() == ActionMode.EACH_CHARACTER:
            self.char_actions.append(action)
        else:
            self.start_actions.append(action)

class DirectMatch(Match, HasDefaultDebugInfo):
    """A direct string match"""
    def __init__(self, match_contents):
        super().__init__()
        self.match_contents = match_contents

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return f"direct match {self.match_contents!r}"

    def convert(self, current_error_handlers: dict):
        sm = DFA() # Create a new SM
        ProgramData.imbue(sm, DebugTag.PARENT, self) # Mark us as the parent of this SM
        state = DFState()
        sm.add(state)
        """
        Create a DFA which looks like

          i    i      i
           1    2      n
        1 -> 2 -> ... -> n

        where i is the input string
        """
        for j, character in enumerate(self.match_contents):
            next_state = DFState()
            start_action_holder = (self.start_actions if j == 0 else [])
            t = DFTransition([character]).attach(*start_action_holder, *self.char_actions).to(next_state)
            state[DFTransition.Else] = DFTransition().to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*start_action_holder).fallthrough()
            if j == len(self.match_contents) - 1:
                t.attach(*self.finish_actions)
                sm.mark_accepting(next_state)
            state.transition(t)
            sm.add(next_state)
            state = next_state
        return sm

class CaseDirectMatch(Match, HasDefaultDebugInfo):
    """A direct string match w/ case insensitivity"""
    def __init__(self, match_contents):
        super().__init__()
        self.match_contents = match_contents

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return f"casei match {self.match_contents!r}"

    def _create_casei_from(self, character: str):
        # TODO: UNICODE handling
        if character in string.ascii_letters:
            return [character, string.ascii_letters[(string.ascii_letters.index(character) + 26) % len(string.ascii_letters)]]
        else:
            return [character]

    def convert(self, current_error_handlers: dict):
        sm = DFA() # Create a new SM
        ProgramData.imbue(sm, DebugTag.PARENT, self) # Mark us as the parent of this SM
        state = DFState()
        sm.add(state)
        """
        Create a DFA which looks like

          i    i      i
           1    2      n
        1 -> 2 -> ... -> n

        where i is the input string
        """
        for j, character in enumerate(self.match_contents):
            next_state = DFState()
            start_action_holder = (self.start_actions if j == 0 else [])
            t = DFTransition(self._create_casei_from(character)).attach(*start_action_holder, *self.char_actions).to(next_state)
            state[DFTransition.Else] = DFTransition().to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*start_action_holder).fallthrough()
            if j == len(self.match_contents) - 1:
                t.attach(*self.finish_actions)
                sm.mark_accepting(next_state)
            state.transition(t)
            sm.add(next_state)
            state = next_state
        return sm

class OutputStorageType(enum.Enum):
    BOOL = 0
    INT = 1
    ENUM = 2
    STR = 3

class OutputStorage(HasDefaultDebugInfo):
    def __init__(self, typ: OutputStorageType, name, default_value=None, enum_values: List[str] = [], str_size=0):
        self.type = typ
        self.name = name
        self.default_value = default_value
        self.enum_values = enum_values
        self.str_size = str_size

    def holds_a(self, typ):
        return self.type == typ

    def debug_lookup(self, tag):
        if tag == DebugTag.NAME:
            return f"output '{self.name}'"

    def __repr__(self):
        return f"<OutputStorage name={self.name} type={self.type}>"

# ==================
# INTEGER EXPR TYPES
# ==================

class IntegerExprUseContext(enum.Enum):
    ASSIGN_ON_MATCH = 0
    ASSIGN_INITIAL = 1

class IntegerExpr(abc.ABC):
    @abc.abstractmethod
    def result_type(self) -> OutputStorageType:
        """
        Get the result type of this expression (integer expression technically corresponds to 
        bool/int/enum
        """
        pass

    def is_literal(self):
        """
        Is this expr evaluatable right now?
        """
        return False

    def get_literal_result(self):
        """
        Evaluate to a native python representation
        """
        return None

    def get_invalid_contexts(self):
        """
        Return an iterable of all the contexts in which this expression is invalid
        """
        return []

class LiteralIntegerExpr(IntegerExpr):
    def __init__(self, value, typ: OutputStorageType = OutputStorageType.INT):
        self.value = value
        self.typ = typ

    def result_type(self):
        return self.typ

    def is_literal(self):
        return True

    def get_literal_result(self):
        return self.value

class OutIntegerExpr(IntegerExpr):
    def __init__(self, ref: "OutputStorage"):
        self.ref = ref

    def result_type(self):
        return self.ref.type

class LastCharIntegerExpr(IntegerExpr):
    def result_type(self):
        return OutputStorageType.INT

    def get_invalid_contexts(self):
        return [IntegerExprUseContext.ASSIGN_INITIAL]

class MathIntegerExpr(IntegerExpr):
    def __init__(self, children: List[IntegerExpr]):
        self.children = children
        
        for child in self.children:
            ProgramData.imbue(child, DebugTag.PARENT, self)

        if not self.children:
            raise IllegalParseTree("Empty math expression", self)
    
        if not all(x.result_type() == OutputStorageType.INT for x in self.children):
            raise IllegalParseTree("Invalid arithmetic type", self)

    def result_type(self):
        return self.children[0].result_type()

    def is_literal(self):
        return all(x.is_literal() for x in self.children)

    def get_invalid_contexts(self):
        returned = set()

        for ctx in itertools.chain(*(x.get_invalid_contexts() for x in self.children)):
            if ctx in returned:
                continue
            returned.add(ctx)
            yield ctx

class SumIntegerExpr(MathIntegerExpr):
    def __init__(self, children: List[IntegerExpr], negate: List[bool]):
        super().__init__(children)
        self.negate = negate

    def get_literal_result(self):
        total = self.children[0].get_literal_result()
        for operand, operator in itertools.islice(zip(self.children, self.negate), 1, None):
            if operator:
                total -= operand
            else:
                total += operand

class MulIntegerExprOp(enum.Enum):
    MUL = '*'
    DIV = '/'
    MOD = '%'

class MulIntegerExpr(MathIntegerExpr):
    def __init__(self, children: List[IntegerExpr], divide: List[MulIntegerExprOp]):
        super().__init__(children)
        self.divide = divide

    def get_literal_result(self):
        total = self.children[0].get_literal_result()
        for operand, operator in itertools.islice(zip(self.children, self.divide), 1, None):
            if operator == MulIntegerExprOp.DIV:
                total //= operand
            elif operator == MulIntegerExprOp.MOD:
                total %= operand
            else:
                total *= operand


# ==========
# GENERAL RE 
# ==========

class RegexCharClass:
    def __init__(self, chars):
        self.chars = frozenset(chars)
        
    def isdisjoint(self, other):
        """
        Are these two character classes disjoint?
        """
        if isinstance(other, InvertedRegexCharClass):
            return self.chars <= other.chars
        elif isinstance(other, RegexCharClass):
            return other.chars.isdisjoint(self.chars)

    def split(self, other):
        """
        Split these two character clases into a tuple:

        (overlap, this_without_overlap, other_without_overlap)
        """
        if isinstance(other, InvertedRegexCharClass):
            # Any character which _we_ match, but which isn't in the Inverted chars set (i.e. which _it_ matches) is the overlap.
            # or-ing characters is the inverse of subtracting them
            overlap = self.chars - other.chars
            return (RegexCharClass(overlap), RegexCharClass(self.chars - overlap), InvertedRegexCharClass(other.chars | overlap))
        elif isinstance(other, RegexCharClass):
            # The overlap is just the intersection, and the non-overlapping parts make sense
            overlap = other.chars & self.chars
            return (RegexCharClass(overlap), RegexCharClass(self.chars - overlap), RegexCharClass(other.chars - overlap)) 

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.chars == self.chars

    def __ne__(self, other):
        return not self.__eq__(other)

    def empty(self):
        return not self.chars

    def union(self, other):
        if isinstance(other, InvertedRegexCharClass):
            return other.union(self)
        else:
            return RegexCharClass(self.chars | other.chars)

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.chars)

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self.chars!r}>"

    def invert(self):
        return InvertedRegexCharClass(self.chars)

class InvertedRegexCharClass(RegexCharClass):
    def isdisjoint(self, other):
        if isinstance(other, InvertedRegexCharClass):
            return len(self.chars | other.chars) >= 256 # TODO: add unicode/specific number of symbols support; for all real uses inverted sets are _never_ disjoint
        elif isinstance(other, RegexCharClass):
            return other.isdisjoint(self)

    def split(self, other):
        if isinstance(other, InvertedRegexCharClass):
            # ^S1 & ^S2 = ^(S1 | S2)
            # ^S1 - ^S2 = S2 - S1
            # ^() complements are represented as InvertedSets so we don't actually need to know what the full set is
            overlap = self.chars | other.chars
            return (InvertedRegexCharClass(overlap), RegexCharClass(other.chars - self.chars), RegexCharClass(self.chars - other.chars))
        elif isinstance(other, RegexCharClass):
            overlap, other, self = other.split(self)
            return (overlap, self, other)

    def empty(self):
        return len(self.chars) >= 256

    def union(self, other):
        if isinstance(other, InvertedRegexCharClass):
            return InvertedRegexCharClass(self.chars & other.chars)
        else:
            return InvertedRegexCharClass(self.chars - other.chars)

    def invert(self):
        return RegexCharClass(self.chars)


class RegexKleene:
    def __init__(self, sub_match):
        ProgramData.imbue(sub_match, DebugTag.PARENT, self)
        self.sub_match = sub_match

class RegexOptional:
    def __init__(self, sub_match):
        ProgramData.imbue(sub_match, DebugTag.PARENT, self)
        self.sub_match = sub_match

class RegexAlternation:
    def __init__(self, sub_matches):
        self.sub_matches = set(sub_matches)
        for sub_match in self.sub_matches:
            ProgramData.imbue(sub_match, DebugTag.PARENT, self)

class RegexSequence:
    def __init__(self, sub_matches):
        self.sub_matches = list(sub_matches)
        for sub_match in self.sub_matches:
            ProgramData.imbue(sub_match, DebugTag.PARENT, self)

class RegexNFState:
    Epsilon = object()

    def __init__(self):
        self.transitions = {}
        self.epsilon_moves = set() 

    def transition(self, symbol, target):
        if symbol == RegexNFState.Epsilon:
            self.epsilon_moves.add(target)
        else:
            self.transitions[symbol] = target
        return self

    def epsilon_closure(self, visited=None):
        if visited is None:
            visited = set()
        total_moves = set((self,))
        for move in self.epsilon_moves:
            total_moves.add(move)
            if move in visited:
                continue
            visited.add(move)
            total_moves |= move.epsilon_closure(visited)
        return total_moves

class RegexNFA:
    def __init__(self):
        self.states = []
        self.start_state = None
        self.finishing_states = []

    def add(self, *states):
        if self.start_state is None:
            self.start_state = states[0]
        self.states.extend(states)

    def mark_finishing(self, state):
        self.finishing_states.append(state)

    def convert_to_dfa(self, alphabet, target_dfa):
        """
        Convert the nfa to a dfa (but keep it in the same container type for simplicity's sake)
        """

        visited_states = {}  # frozenset of index to state
        index_cache = {}
        to_process = queue.Queue()

        def get_index(state):
            if state in index_cache:
                return index_cache[state]
            index_cache[state] = self.states.index(state)
            return index_cache[state]

        def moves(states, on):
            results = set() 
            for i in states:
                if on in self.states[i].transitions:
                    results.add(get_index(self.states[i].transitions[on]))
            return results

        def epsilon_closure(states):
            total = set()
            for i in states:
                total |= set(get_index(x) for x in self.states[i].epsilon_closure())
            return frozenset(total)


        start_dfa_state = frozenset(get_index(x) for x in self.start_state.epsilon_closure())
        finishing_idx = get_index(self.finishing_states[0])

        visited_states[start_dfa_state] = RegexNFState()
        if finishing_idx in start_dfa_state:
            target_dfa.mark_finishing(visited_states[start_dfa_state])
        to_process.put(start_dfa_state)
        ProgramData.imbue(visited_states[start_dfa_state], DebugTag.PARENT, self.states[next(iter(start_dfa_state))])

        while not to_process.empty():
            processing = to_process.get()
            # find all potential edges
            for potential_move in alphabet:
                # is there anything here
                move_result = moves(processing, potential_move)
                if move_result:
                    # the new state is the e-closure
                    new_state = epsilon_closure(move_result)
                    if new_state not in visited_states:
                        # create the new state
                        visited_states[new_state] = RegexNFState()
                        ProgramData.imbue(visited_states[new_state], DebugTag.PARENT, self.states[next(iter(new_state))])
                        # should it be a finishing state?
                        if finishing_idx in new_state:
                            target_dfa.mark_finishing(visited_states[new_state])
                        to_process.put(new_state)
                    visited_states[processing].transition(potential_move, visited_states[new_state])

        # add all the states
        for i in visited_states.values():
            target_dfa.add(i)

        return target_dfa

    def minimize_dfa(self, alphabet, new_dfa):
        """
        Use hopcroft's algorithm to minize the dfa_1

        Effectively, we continually separate the states into partitions based on an equality relationship and then
        use those new sets as the states
        """

        def initial_partition():
            T = {False: set(), True: set()}
            for state in self.states:
                if state in self.finishing_states:
                    T[True].add(state)
                else:
                    T[False].add(state)
            return set(frozenset(x) for x in T.values())

        P = set()
        T = initial_partition()

        def partition_containing(state):
            try:
                return next(p for p in P if state in p)
            except StopIteration:
                return None

        def split(S: Iterable[RegexNFState]):
            def splits(c):
                for state in S:
                    s1 = set()
                    s2 = set()

                    expected = state.transitions.get(c, None)
                    expected = partition_containing(expected)
                    for other in S:
                        actual = other.transitions.get(c, None)
                        actual = partition_containing(actual)
                        if actual == expected:
                            s1.add(other)
                        else:
                            s2.add(other)

                    if s1 and s2:
                        return {frozenset(s1), frozenset(s2)}

            for char in alphabet:
                split = splits(char)
                if split:
                    return split
            return {S}

        while P != T:
            P = T
            T = set()
            for p in P:
                T |= split(p)
        new_states = {}
        
        # Reconstruct the new dfa from the set equivalences
        
        def add_back(subset):
            state = next(iter(subset))
            new_state = RegexNFState()
            new_states[subset] = new_state
            new_dfa.add(new_state)
            ProgramData.imbue(new_state, DebugTag.PARENT, state)

            if state in self.finishing_states:
                new_dfa.mark_finishing(new_state)

            for (character, target) in state.transitions.items():
                target_subset = partition_containing(target)

                if target_subset not in new_states:
                    add_back(target_subset)
                new_state.transition(character, new_states[target_subset])

            return new_state

        new_dfa.start_state = add_back(partition_containing(self.start_state))
        return new_dfa


class RegexMatch(Match):
    def __init__(self, regex_parse_tree):
        super().__init__()
        # First, get all the character classes we will encounter and create a mapping for them
        all_char_classes = self._visit_all_char_classes(regex_parse_tree)
        self.character_class_mappings = self._make_disjoint_groupings(all_char_classes)
        self.alphabet = set()
        for x in self.character_class_mappings.values():
            self.alphabet |= set(x)
        # Create the simplified representation
        self.regex_tree = self._interpret_parse_tree(regex_parse_tree)
        self.regex_tree = self._simplify_regex_tree(self.regex_tree)
        ProgramData.imbue(self.regex_tree, DebugTag.PARENT, self)

        # Variables used during construction (similarly to how mlang works, to reduce arguments in recursive methods)
        self.nfa: Optional[RegexNFA] = None
        self.dfa_1: Optional[RegexNFA] = None
        self.dfa_2: Optional[RegexNFA] = None
        self.out_dfa_cache = {}

    def _convert_to_nfa(self, r, start_state: RegexNFState):
        """
        Convert the regex tree object into the NFA using Thompson construction. Return the finish state
        """
        ProgramData.imbue(r, DebugTag.PARENT, start_state)

        if isinstance(r, RegexCharClass):
            # Simply convert to a boring form
            end_state = RegexNFState()
            ProgramData.imbue(end_state, DebugTag.PARENT, start_state)
            self.nfa.add(end_state)
            start_state.transition(r, end_state)
            return end_state
        elif isinstance(r, RegexAlternation):
            r"""
            Use the union form:
             /e-(s    f)-e\
            q              f 
             \e-(s2  f2)-e/
            """
            # Create a set of start states
            sub_starts = [RegexNFState() for x in r.sub_matches]
            end_state = RegexNFState()
            self.nfa.add(end_state, *sub_starts)
            ProgramData.imbue(end_state, DebugTag.PARENT, start_state)
            # Link everything up
            for i, j in zip(sub_starts, r.sub_matches):
                # Link e from start to sub start
                start_state.transition(RegexNFState.Epsilon, i)
                # Create sub expr & link to end
                self._convert_to_nfa(j, i).transition(RegexNFState.Epsilon, end_state)
            return end_state
        elif isinstance(r, RegexSequence):
            # Chain them all together
            for i in r.sub_matches:
                start_state = self._convert_to_nfa(i, start_state)
            return start_state
        elif isinstance(r, RegexOptional):
            # Create a kleene star without the repetition bit
            end_state = RegexNFState()
            sub_start = RegexNFState()
            self.nfa.add(end_state, sub_start)
            ProgramData.imbue(end_state, DebugTag.PARENT, start_state)
            start_state.transition(RegexNFState.Epsilon, end_state).transition(RegexNFState.Epsilon, sub_start)
            self._convert_to_nfa(r.sub_match, sub_start).transition(RegexNFState.Epsilon, end_state)
            return end_state
        elif isinstance(r, RegexKleene):
            r"""
            Use the kleene form:
                     <-e-
                    /    \
            q -e-> (s    f) -e-> f
             \                  /
              --- ----e--> -----
            """
            end_state = RegexNFState()
            sub_start = RegexNFState()
            self.nfa.add(end_state, sub_start)
            ProgramData.imbue(end_state, DebugTag.PARENT, start_state)
            start_state.transition(RegexNFState.Epsilon, end_state).transition(RegexNFState.Epsilon, sub_start)
            self._convert_to_nfa(r.sub_match, sub_start).transition(RegexNFState.Epsilon, end_state).transition(RegexNFState.Epsilon, sub_start)
            return end_state
        else:
            raise NotImplementedError("unknown")

    def _simplify_regex_tree(self, r):
        """
        Simplify the regex tree recursively
        """
        if isinstance(r, RegexAlternation) or isinstance(r, RegexSequence):
            r.sub_matches = [self._simplify_regex_tree(x) for x in r.sub_matches]
            if len(r.sub_matches) == 1:
                return r.sub_matches[0]
            return r
        if isinstance(r, RegexOptional) or isinstance(r, RegexKleene):
            r.sub_match = self._simplify_regex_tree(r.sub_match)
            return r
        return r
    
    def _interpret_parse_tree(self, regex_tree: lark.Tree):
        if regex_tree.data in ("regex", "regex_group"):
            return RegexSequence(self._interpret_parse_tree(x) for x in regex_tree.children)
        elif regex_tree.data in ("regex_any", "regex_char_class", "regex_set", "regex_inverted_set"):
            return RegexAlternation(self.character_class_mappings[list(self._visit_all_char_classes(regex_tree))[0]])
        elif regex_tree.data == "regex_alternation":
            return RegexAlternation(self._interpret_parse_tree(x) for x in regex_tree.children)
        elif regex_tree.data == "regex_raw_match":
            return RegexAlternation(self.character_class_mappings[self._convert_raw_regex_unimportant(regex_tree.children[0])])
        elif regex_tree.data == "regex_operation":
            sub_match = self._interpret_parse_tree(regex_tree.children[0])
            if regex_tree.children[1].value == "+":
                val = RegexSequence((sub_match, RegexKleene(sub_match)))
            else:
                val = {"*": RegexKleene, "?": RegexOptional}[regex_tree.children[1].value](sub_match)
            ProgramData.imbue(val, DebugTag.SOURCE_LINE, regex_tree.children[1].line)
            ProgramData.imbue(val, DebugTag.SOURCE_COLUMN, regex_tree.children[1].column)
            return val
        else:
            raise NotImplementedError("don't handle {} yet".format(regex_tree.data))

    def _convert_raw_regex_unimportant(self, regex_tree: lark.Token):
        if regex_tree.value[0] == '\\':
            v = RegexCharClass((regex_tree.value[1],))
        else:
            v = RegexCharClass((regex_tree.value[0],))
        ProgramData.imbue(v, DebugTag.SOURCE_LINE, regex_tree.line)
        ProgramData.imbue(v, DebugTag.SOURCE_COLUMN, regex_tree.column)
        return v

    def _convert_raw_regex_char_class(self, regex_char_class: lark.Tree):
        val = {
            "n": RegexCharClass("\n"),
            "t": RegexCharClass("\t"),
            "r": RegexCharClass("\r"),
            "w": RegexCharClass(string.ascii_letters + string.digits + "_"),
            "W": InvertedRegexCharClass(string.ascii_letters + string.digits + "_"),
            "d": RegexCharClass(string.digits),
            "D": InvertedRegexCharClass(string.digits),
            "s": RegexCharClass(string.whitespace),
            "S": InvertedRegexCharClass(string.whitespace),
            " ": RegexCharClass(" ")
        }[regex_char_class.children[0].value[0]]
        ProgramData.imbue(val, DebugTag.SOURCE_LINE, regex_char_class.children[0].line)
        ProgramData.imbue(val, DebugTag.SOURCE_COLUMN, regex_char_class.children[0].column)
        return val

    def _visit_all_char_classes(self, regex_tree: lark.Tree):
        """
        Recursively find all character classes in the tree and return them as a set.
        """

        if regex_tree.data in ("regex", "regex_group", "regex_alternation"):
            result = set()
            for child in regex_tree.children:
                result |= self._visit_all_char_classes(child)
            return result
        if regex_tree.data == "regex_raw_match":
            return set(self._convert_raw_regex_unimportant(x) for x in regex_tree.children)
        if regex_tree.data == "regex_any":
            return set((InvertedRegexCharClass(()),))
        if regex_tree.data == "regex_char_class":
            return set((self._convert_raw_regex_char_class(regex_tree),))
        if regex_tree.data in ("regex_operation", "regex_exact_repeat", "regex_range_repeat", "regex_at_least_repeat"):
            return self._visit_all_char_classes(regex_tree.children[0])
        if regex_tree.data in ("regex_set", "regex_inverted_set"):
            inverted = regex_tree.data != "regex_set"
            incoming_set = RegexCharClass(())
            for child in regex_tree.children:
                if isinstance(child, lark.Token):
                    new_set = self._convert_raw_regex_unimportant(child)
                elif child.data == "regex_set_range":
                    start = list(self._convert_raw_regex_unimportant(child.children[0]).chars)[0]
                    end = list(self._convert_raw_regex_unimportant(child.children[1]).chars)[0]
                    new_set = RegexCharClass(chr(x) for x in range(ord(start), ord(end)+1))
                else:
                    new_set = self._convert_raw_regex_char_class(child)
                incoming_set = incoming_set.union(new_set)
            if not inverted:
                return set((incoming_set,))
            else:
                return set((incoming_set.invert(),))


    def _make_disjoint_groupings(self, original_char_classes: List[RegexCharClass]):
        """
        Take the potentially disjoint character classes and convert them to a dictionary of those original classes to a list
        of guaranteed disjoint classes.

        The union of all the values of the dictionary is the new temporary input alphabet for the regex-NFA

        Every time a given class occurs, it can be replaced with an alternation of these classes.
        """
        total_char_classes = list(original_char_classes)
        new_character_classes = {
            original: [i] for i, original in enumerate(total_char_classes)
        }

        """
        In effect, keep finding non-disjoint character classes and splitting them up.
        The large amount of logic is primarily to deal with iteration problems
        """
        keepgoing = True
        while keepgoing:
            keepgoing = False
            for ia, ib in itertools.combinations(range(len(total_char_classes)), 2):
                a = total_char_classes[ia]
                b = total_char_classes[ib]
                if not a.isdisjoint(b):
                    dprint[ProgramFlag.VERBOSE_REGEX_CCLASS]("splitting", a, "and", b)
                    keepgoing = True
                    # Split
                    overlap, newa, newb = a.split(b)
                    dprint[ProgramFlag.VERBOSE_REGEX_CCLASS]("into", overlap, newa, newb)
                    io = len(total_char_classes)
                    # Add overlap
                    total_char_classes.append(overlap)
                    # Search
                    for k in new_character_classes:
                        if ia in new_character_classes[k]:
                            dprint[ProgramFlag.VERBOSE_REGEX_CCLASS]("adding overlap for a at", k)
                            new_character_classes[k].append(io)
                        if ib in new_character_classes[k]:
                            dprint[ProgramFlag.VERBOSE_REGEX_CCLASS]("adding overlap for b at", k)
                            new_character_classes[k].append(io)
                    # Overwrite
                    total_char_classes[ia] = newa
                    total_char_classes[ib] = newb
                    # Do again
                    break

        return {k: [*(total_char_classes[i] for i in v if not total_char_classes[i].empty())] for k, v in new_character_classes.items()}

    def _create_dfa_state(self, nfdfa_state: RegexNFState, into: DFA, is_start: bool, else_path):
        """
        Recursively create the new states
        """

        if id(nfdfa_state) in self.out_dfa_cache:
            return self.out_dfa_cache[id(nfdfa_state)]

        transitions = nfdfa_state.transitions
        new_transitions = {frozenset((DFTransition.Else,)): (else_path, False)}

        new_state = DFState()
        ProgramData.imbue(new_state, DebugTag.PARENT, nfdfa_state)
        self.out_dfa_cache[id(nfdfa_state)] = new_state
        into.add(new_state)

        if nfdfa_state in self.dfa_2.finishing_states:
            into.mark_accepting(new_state)

        for source, target in transitions.items():
            if isinstance(source, InvertedRegexCharClass):
                # TODO: handle multiple of these
                # Convert to a normal set
                new_transitions[source.chars | frozenset((DFTransition.End,))] = (else_path, False)
                new_transitions[frozenset((DFTransition.Else,))] = (self._create_dfa_state(target, into, False, else_path), target in self.dfa_2.finishing_states)
            else:
                new_transitions[source.chars] = (self._create_dfa_state(target, into, False, else_path), target in self.dfa_2.finishing_states)

        # Simplify
        new_transitions_inverse = defaultdict(list) 
        for source, target in new_transitions.items():
            new_transitions_inverse[target].append(source)

        new_transitions = {}
        for target, sources in new_transitions_inverse.items():
            total = set()
            for x in sources:
                total |= x
            new_transitions[frozenset(total)] = target

        for source, (target, use_finish) in new_transitions.items():
            source = set(source)
            use_each = target != else_path

            actions = []
            if is_start:
                actions.extend(self.start_actions)
            if use_each:
                actions.extend(self.char_actions)
            if use_finish:
                actions.extend(self.finish_actions)
            
            # resolve conflicts
            for conflict in new_state.all_transitions_for(source):
                # if we are else, remove from us
                if target == else_path:
                    source -= set(conflict.on_values)
                # otherwise, throw error
                elif conflict.target != else_path:
                    raise IllegalDFAStateConflictsError("Duplicate transition", target, new_state)
                else:
                    # They are else, remove from them
                    for x in source:
                        if x in conflict.on_values:
                            conflict.on_values.remove(x)

            new_state.transition(DFTransition(list(source)).to(target).attach(*actions).fallthrough(target == else_path))

        return new_state


    def convert(self, current_error_handlers: dict):
        # First convert to an NFA
        self.nfa = RegexNFA()
        start_state = RegexNFState()
        self.nfa.add(start_state)
        self.nfa.mark_finishing(self._convert_to_nfa(self.regex_tree, start_state))
        # Then convert to a DFA

        new_dfa = RegexNFA()
        ProgramData.imbue(new_dfa, DebugTag.PARENT, self)
        self.dfa_1 = self.nfa.convert_to_dfa(self.alphabet, new_dfa)
        # Minimize the DFA

        new_dfa = RegexNFA()
        ProgramData.imbue(new_dfa, DebugTag.PARENT, self)
        self.dfa_2 = self.dfa_1.minimize_dfa(self.alphabet, new_dfa)
        
        # Check if it's possible to schedule the finish actions. TODO: instead of throwing an error, attempt to move them to the next thing's start
        if any(x.is_timing_strict() for x in self.finish_actions) and any(x.transitions for x in self.dfa_2.finishing_states):
            # Complain early
            raise IllegalDFAStateConflictsError("Unable to schedule finish actions only once", *(x for x in self.dfa_2.finishing_states if x.transitions), 
                    *(x for x in self.finish_actions if x.is_timing_strict()))

        # Create a normal SM
        out_dfa = DFA()
        self._create_dfa_state(self.dfa_2.start_state, out_dfa, True, current_error_handlers[ErrorReasons.NO_MATCH])
        return ProgramData.imbue(out_dfa, DebugTag.PARENT, self)

class WaitMatch(Match):
    def __init__(self, sub_match: Match):
        super().__init__()
        ProgramData.imbue(sub_match, DebugTag.PARENT, self)
        self.match_contents = sub_match

    def attach(self, action: Action):
        self.match_contents.attach(action)
        super().attach(action)

    def convert(self, current_error_handlers: dict):
        sm = self.match_contents.convert(current_error_handlers)
        for state, trans in sm.transitions_pointing_to(current_error_handlers[ErrorReasons.NO_MATCH], True):
            trans.to(sm.starting_state)
            if state == sm.starting_state:
                trans.fallthrough(False)
        return sm

class EndMatch(Match):
    def convert(self, current_error_handlers: dict):
        """
            END
        s   -->  s
         0        1
        """
        sm = DFA()
        ProgramData.imbue(sm, DebugTag.PARENT, self)
        start_state = DFState()
        sm.add(start_state)
        ok_state = DFState()
        sm.add(ok_state)
        sm.mark_accepting(ok_state)
        start_state.transition(DFTransition([DFTransition.End]).attach(*self.start_actions, *self.char_actions, *self.finish_actions).to(ok_state))
        start_state.transition(DFTransition([DFTransition.Else]).to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*self.start_actions)).fallthrough()
        return sm

class ConcatMatch(Match):
    def __init__(self, sub_matches: List[Match]):
        super().__init__()
        self.sub_matches = sub_matches
        for i in sub_matches:
            ProgramData.imbue(i, DebugTag.PARENT, self)

    def convert(self, current_error_handlers: dict):
        # distribute actions
        self.sub_matches[0].start_actions.extend(self.start_actions)
        self.sub_matches[-1].finish_actions.extend(self.finish_actions)
        for i in self.sub_matches:
            i.char_actions.extend(self.char_actions.copy())
        # convert in order
        sm = self.sub_matches[0].convert(current_error_handlers)
        for i in self.sub_matches[1:]:
            sm.append_after(i.convert(current_error_handlers), current_error_handlers[ErrorReasons.NO_MATCH])
        return sm

class MatchNode(ActionSinkNode):
    """
    Node which executes a match.
    """

    def __init__(self, match: Match):
        ProgramData.imbue(match, DebugTag.PARENT, self)
        self.match = match
        self.next = None

    def _adopt_actions(self, actions: List[Action]):
        for action in actions:
            self.match.attach(action)

    def _set_next(self, next_node):
        self.next = next_node

    def get_next(self):
        return self.next

    def convert(self, current_error_handlers: dict):
        base_dfa = self.match.convert(current_error_handlers)
        if self.next is not None:
            base_dfa.append_after(self.next.convert(current_error_handlers), current_error_handlers[ErrorReasons.NO_MATCH])
        return base_dfa

class CaseNode(Node):
    """
    Handles cases
    """

    def __init__(self, sub_matches: Dict[Set[Optional[Match]], Node]):
        super().__init__()
        self.sub_matches = {k: v for k, v in sub_matches.items() if v is not None}
        self.empty_matches = [k for k, v in sub_matches.items() if v is None]
        self.case_match_actions = defaultdict(list)
        self.next = None

        self._find_case_actions()

    def _find_case_actions(self):
        """
        Find all the case actions and delete ActionNodes
        """

        to_replace = []

        for sub_matches, target in self.sub_matches.items():
            if not isinstance(target, ActionNode):
                continue

            # adopt into our internal list
            self.case_match_actions.update({sub_matches: target.actions})
            if target.get_next() is not None:
                self.sub_matches[sub_matches] = target.get_next()
            else:
                to_replace.append(sub_matches)

        for i in to_replace:
            del self.sub_matches[i]
            self.empty_matches.append(i)

    def _merge(self, ds: Iterable[DFA], treat_as_else: DFState):
        r"""
        Merge the DFAs in the list ds, ensuring that all finishing states are kept as-is.

        This uses largely the same algorithm as the NFA-to-DFA conversion, treating the input as an NFA of the form
             s
            /| \ 
          e  e   e
         /   |    \ 
        D1  D2 .. Dn

        As such, we can generally use a similar, if simplified algorithm.

        The technique is similar, keeping track of new states as sets of sub-states. Instead of directly using indices to name states, we use tuples of DFA-index and index in that DFA.

        In order to deal with the complexities of the DFA representation in NMFU, we do a simplification pass on the input alphabet. We consider every matched symbol (no conditions allowed) individually to form
        an alphabet. We then add an additional entry for "true else". An Else transition is therefore equal to "true else" + every symbol not included in the transitions
        from the given state.

        We also ignore all transitions that go to the entry marked as treat_as_else, as we re-create all the else transitions later.

        In order to avoid having to run the DFA minimizer, we utilize a similar approach to the regex class to turn sets of alphabet entries into non-overlapping forms. These are then used as
        localized alphabet symbols.

        We start in state (D1,D1_start),(D2,D2_start),...,(Dn,Dn_start)
        """

        new_dfa = DFA()
        alphabet = set()

        for dfa in ds:
            for state in dfa.states:
                for trans in state.transitions:
                    if trans.conditions:
                        raise IllegalDFAStateError("Conditions are not allowed in case matches", state)
                    for v in trans.on_values:
                        alphabet.add(v)

        alphabet.add(DFTransition.Else) 

        converted_states = {}
        corresponding_finish_states = {dfa: [] for dfa in ds}
        to_process = queue.Queue()

        def create_real_state_of(state):
            new_state = DFState()
            ProgramData.imbue(new_state, DebugTag.PARENT, next(iter(state))[1])

            corresponds_to_finishes_in = set() 
            is_part_of = set()
            for sub_dfa, sub_state in state:
                if sub_state in sub_dfa.accepting_states:
                    corresponds_to_finishes_in.add(sub_dfa)
                is_part_of.add(sub_dfa)

            if len(corresponds_to_finishes_in) > 1:
                raise IllegalDFAStateConflictsError("Ambigious case label: multiple possible finishes", *corresponds_to_finishes_in)
            elif len(corresponds_to_finishes_in) == 1:
                if len(is_part_of) != 1:
                    raise IllegalDFAStateConflictsError("Ambigious case label: should finish or check next", *is_part_of)
                corresponding_finish_states[next(iter(corresponds_to_finishes_in))].append(new_state)
                new_dfa.mark_accepting(new_state)

            # Add the state
            converted_states[state] = new_state
            new_dfa.add(new_state)
            return new_state

        start_state = frozenset((dfa, dfa.starting_state) for dfa in ds)
        create_real_state_of(start_state)
        to_process.put(start_state)

        while not to_process.empty():
            processing = to_process.get()

            local_alphabet = set()

            # grab the local alphabet as a set of sets (on_values) IGNORING things that go to else_path
            for _, sub_state in processing:
                for trans in sub_state.transitions:
                    if trans.target == treat_as_else:
                        continue
                    local_alphabet.add(frozenset(trans.on_values))

            # simplify such that each element in local_alphabet is both disjoint and are all subsets of
            # at least one entry in the original local_alphabet (i.e. such that any original element can
            # be created by combining new ones)

            while True:
                try:
                    a, b = next((x, y) for x, y in itertools.combinations(local_alphabet, 2) if (x & y))
                    overlap = a & b
                    local_alphabet.remove(a)
                    local_alphabet.remove(b)
                    a = a - overlap
                    b = b - overlap
                    if a: local_alphabet.add(a)
                    if b: local_alphabet.add(b)
                    local_alphabet.add(overlap)
                except StopIteration:
                    break

            # process all moves with those sets as alphabets

            for symbol in local_alphabet:
                # construct the set of next states
                next_state = set()
                for (sub_dfa, sub_state) in processing:
                    potential_transition = sub_state[symbol]
                    if potential_transition is None:
                        continue
                    if potential_transition.target == treat_as_else:
                        continue
                    else:
                        next_state.add((sub_dfa, potential_transition.target))

                if not next_state:
                    continue

                next_state = frozenset(next_state)

                if next_state not in converted_states:
                    to_process.put(next_state)
                    next_state = create_real_state_of(next_state)
                else:
                    next_state = converted_states[next_state]

                converted_states[processing][symbol] = next_state

            # check if we need else (is this a finishing state)
            if converted_states[processing] in new_dfa.accepting_states:
                continue


            # construct the actual else transition based on the inverse of alphabet
            flat_local_alphabet = set(itertools.chain(*local_alphabet))

            actual_else = alphabet - flat_local_alphabet
            if DFTransition.Else in actual_else:
                # just use it
                actual_else = set((DFTransition.Else,))
            elif DFTransition.Else in flat_local_alphabet:
                continue # Ignore it because else will
            
            if actual_else: # sometimes you actually don't need one
                converted_states[processing].transition(DFTransition(list(actual_else)).to(treat_as_else).fallthrough(), allow_replace=True)

        return new_dfa, corresponding_finish_states

    def get_next(self):
        return self.next

    def set_next(self, next_node):
        if isinstance(next_node, ActionNode):
            our_next_node = next_node.get_next()
            next_node.set_next(None)
            for sub_ast in self.sub_matches.values():
                # Go to end of sub_ast
                while sub_ast.get_next() is not None:
                    sub_ast = sub_ast.get_next()

                # Adopt it
                sub_ast.set_next(next_node)
            for empty_match in self.empty_matches:
                self.case_match_actions[empty_match].extend(next_node.actions)
            self.next = our_next_node
        else:
            self.next = next_node

    def convert(self, current_error_handlers):
        has_else = any(None in x for x in itertools.chain(self.sub_matches.keys(), self.empty_matches))

        # First, render out all of the sub_dfas
        sub_dfas = {x: y.convert(current_error_handlers) for x, y in self.sub_matches.items()}

        # Map of (DFA) -> (set of Matches)
        original_backreference = {}
        # Map of (DFA) -> (set of Empty Matches) aka matches that have nothing but actions
        empty_backreference = {}
        # All dfas that need to be merged
        mergeable_ds = set() 
        # Flatten the sub_matches
        for sub_matches in self.sub_matches:
            for sub_match in sub_matches:
                if sub_match is not None:
                    converted = sub_match.convert(current_error_handlers)
                    original_backreference[converted] = sub_matches
                    mergeable_ds.add(converted)
                else:
                    original_backreference[None] = sub_matches
        for empty_matches in self.empty_matches:
            for empty_match in empty_matches:
                if empty_match is not None:
                    converted = empty_match.convert(current_error_handlers)
                    original_backreference[converted] = None
                    empty_backreference[converted] = empty_matches
                    mergeable_ds.add(converted)
                else:
                    original_backreference[None] = None
                    empty_backreference[None] = None

        # Create the merged acceptor
        decider_dfa, corresponding_finish_states = self._merge(mergeable_ds, current_error_handlers[ErrorReasons.NO_MATCH])

        # Check if we need to handle else
        if has_else:
            try:
                else_actions = next(v for k, v in self.case_match_actions.items() if None in k)
            except StopIteration:
                else_actions = []

            # Is there a sub-DFA (actual clause to execute) for the else transition?
            if original_backreference[None] is not None:
                # Find all transitions that would be pointing to a nomatch...
                for trans in decider_dfa.transitions_pointing_to(current_error_handlers[ErrorReasons.NO_MATCH]):
                    # ... and reattach them to the new else state machine
                    trans.to(sub_dfas[original_backreference[None]].starting_state).attach(*else_actions)
                    ProgramData.imbue(trans, DebugTag.NAME, "else action on case node")
                # We have to add that DFA's state to the overall dfa's states array, since it won't get picked up by mergable_ds _UNLESS_ original_backreference contains more than frozenset({None})
                if len(original_backreference[None]) == 1:
                    for state in sub_dfas[original_backreference[None]].states:
                        if state in sub_dfas[original_backreference[None]].accepting_states:
                            decider_dfa.mark_accepting(state)
                        decider_dfa.add(state)
            else:
                # Otherwise, make up our own very stupid one.
                new_state = DFState()
                decider_dfa.mark_accepting(new_state)
                decider_dfa.add(new_state)
                for trans in decider_dfa.transitions_pointing_to(current_error_handlers[ErrorReasons.NO_MATCH]):
                    trans.to(new_state).attach(*else_actions, prepend=True).fallthrough()  # this is an error handler, make sure it's a fallthrough
                    # give the transition better debug info
                    ProgramData.imbue(trans, DebugTag.NAME, "else action on case node")

        # Go through and link up all the states
        for i in mergeable_ds:
            # If there was no state machine associated with the DFA
            if original_backreference[i] is None:
                # Find the actual backref
                true_backref = empty_backreference[i]
                # If this was _not_ the else
                if true_backref is not None:
                    # Handle empty matches
                    all_transitions_empty = set().union(*(decider_dfa.transitions_pointing_to(x) for x in corresponding_finish_states[i]))
                    if len(all_transitions_empty) != 1 and any(x.is_timing_strict() for x in self.case_match_actions[true_backref]):
                        raise IllegalDFAStateError("Unable to schedule strict finish action for case", i)
                    # Add actions
                    for j in all_transitions_empty:
                        j.attach(*self.case_match_actions[true_backref], prepend=True)
            else:
                refers_to = sub_dfas[original_backreference[i]]
                decider_dfa.append_after(refers_to, current_error_handlers[ErrorReasons.NO_MATCH], sub_states=corresponding_finish_states[i], chain_actions=self.case_match_actions[original_backreference[i]])

        ProgramData.imbue(decider_dfa, DebugTag.PARENT, self)

        # If we need to, add a boring after thing
        if self.next is not None:
            decider_dfa.append_after(self.next.convert(current_error_handlers), current_error_handlers[ErrorReasons.NO_MATCH])
        
        return decider_dfa

class OptionalNode(ActionSinkNode):
    def __init__(self, sub_contents: Node):
        self.sub_contents = sub_contents
        self.start_actions = []
        self.finish_actions = []
        self.next = None

    def _set_next(self, next_node):
        self.next = next_node

    def get_next(self):
        return self.next

    def _adopt_actions(self, actions):
        if any(action.get_mode() == ActionMode.EACH_CHARACTER for action in actions):
            raise IllegalASTStateError("Each character action in optional makes no sense", self)

        for action in actions:
            if action.get_mode() == ActionMode.AT_START:
                self.start_actions.append(action)
            else:
                self.finish_actions.append(action)

    def convert(self, current_error_handlers):
        sub_dfa = self.sub_contents.convert(current_error_handlers)
        if sub_dfa.starting_state in sub_dfa.accepting_states:
            raise IllegalDFAStateError("Ambigious path in optional: should use optional or go to next", sub_dfa)

        sub_dfa.mark_accepting(sub_dfa.starting_state)

        # Add starting actions
        for trans in sub_dfa.starting_state.transitions:
            trans.attach(*self.start_actions)

        # If we need to, add a boring after thing
        if self.next is not None:
            sub_dfa.append_after(self.next.convert(current_error_handlers), current_error_handlers[ErrorReasons.NO_MATCH], chain_actions=self.finish_actions)

        return sub_dfa

class LoopNode(ActionSinkNode, HasDefaultDebugInfo):
    def __init__(self, name):
        self.name = name
        self.next = None
        self.after_break_actions = []
        self.loop_start_actions = []
        self.child_node = None

        self.break_action = BreakAction(id(self))
    
    def debug_lookup(self, tag):
        if tag == DebugTag.NAME:
            return "loop node {}".format(self.name)

    def _set_next(self, next_node):
        self.next = next_node

    def get_next(self):
        return self.next

    def _adopt_actions(self, actions):
        if any(x.get_mode() != ActionMode.AT_FINISH for x in actions):
            raise IllegalASTStateError("Invalid action type for loop", self)

        self.after_break_actions.extend(actions)

    def get_break_handler(self):
        return self.break_action

    def set_child(self, child: Node):
        if isinstance(child, ActionNode):
            self.after_break_actions = child.actions
            child = child.get_next()
        self.child_node = child

    def convert(self, current_error_handlers):
        if self.child_node is None:
            raise IllegalDFAStateError("Empty loop body", self)
        # First, create the sub_dfa 
        sub_dfa = self.child_node.convert(current_error_handlers)

        # Attempt to add the loop start actions to the start state
        for transition in sub_dfa.starting_state.all_transitions():
            transition.attach(*self.loop_start_actions)

        # Create the finish state
        end_state = DFState()
        ProgramData.imbue(end_state, DebugTag.PARENT, self)
        sub_dfa.add(end_state)
        # don't mark it accepting yet

        should_try_to_append = False

        # Reroute all transitions with a BreakAction in them that corresponds to our break action to go to us
        for transition in sub_dfa.transitions_that_do(self.break_action):
            transition.to(end_state)
            transition.actions.remove(self.break_action)
            transition.actions.extend(self.after_break_actions)
            should_try_to_append = True

        # Verify that the accepting states are all distinct
        for accept_state in sub_dfa.accepting_states:
            for transition in accept_state.all_transitions():
                if transition.target in sub_dfa.accepting_states:
                    raise IllegalDFAStateConflictsError("Ambigious loop: should loop or continue matching", accept_state, transition.target)

        # Replace all transitions that go to the finishing states with transitions that go to the sub_dfa starting state
        for transition in itertools.chain(*(sub_dfa.transitions_pointing_to(x) for x in sub_dfa.accepting_states)):
            transition.to(sub_dfa.starting_state)

        sub_dfa.accepting_states = [end_state]
        
        if self.next and not should_try_to_append:
            raise IllegalASTStateError("Unreachable states after loop", self.next)

        elif should_try_to_append and self.next:
            sub_dfa.append_after(self.next.convert(current_error_handlers), current_error_handlers[ErrorReasons.NO_MATCH])

        return sub_dfa

class TryExceptNode(ActionSinkNode):
    def __init__(self, handles):
        self.handler_node = DFState()
        self.handles = handles
        self.after_actions = []
        self.incoming_handler_actions = []
        self.incoming_body_actions = []

        self.body = None
        self.handler = None

        self.next = None

    def get_handler(self):
        return self.handler_node

    def set_body(self, body):
        if isinstance(body, ActionNode):
            self.incoming_body_actions = body.actions
            body = body.get_next()
        self.body = body

    def set_handler(self, handler):
        if isinstance(handler, ActionNode):
            self.incoming_handler_actions = handler.actions
            handler = handler.get_next()
        self.handler = handler

    def _adopt_actions(self, actions):
        if any(x.get_mode() != ActionMode.AT_FINISH for x in actions):
            raise IllegalASTStateError("Invalid action type following try-except", self)
            
        self.after_actions.extend(actions)

    def _set_next(self, next_node):
        self.next = next_node

    def get_next(self):
        return self.next

    def convert(self, current_error_handlers):
        if self.body is None:
            raise IllegalASTStateError("Empty try-except body", self)
        # Convert the main DFA to form the "sub-dfa"

        body_error_handlers = current_error_handlers.copy()
        body_error_handlers.update({x: self.handler_node for x in self.handles})

        sub_dfa: DFA = self.body.convert(body_error_handlers)

        # Add the incoming_body_actions array to all incoming actions
        for trans in sub_dfa.starting_state.transitions:
            trans.attach(*self.incoming_body_actions)

        def add_finish_to(dfa):
            if any(any(x.target in dfa.accepting_states for x in y.all_transitions()) for y in dfa.accepting_states) and any(x.is_timing_strict() for x in self.after_actions):
                raise IllegalASTStateError("Unable to schedule finish action for try-except at strict time", dfa)

            for trans in itertools.chain(*(dfa.transitions_pointing_to(x) for x in dfa.accepting_states)):
                trans.attach(*self.after_actions)

        add_finish_to(sub_dfa)

        # This _should_ have transitions going to the handler node. Add it to the tree now
        sub_dfa.add(self.handler_node)

        # If there _is_ a handler, add it after
        if self.handler is not None:
            handler_dfa = self.handler.convert(current_error_handlers)
            add_finish_to(handler_dfa)
            sub_dfa.append_after(handler_dfa, current_error_handlers[ErrorReasons.NO_MATCH], sub_states=[self.handler_node], chain_actions=self.incoming_handler_actions)
        else:
            # Otherwise, just mark the handler as a finish state
            sub_dfa.mark_accepting(self.handler_node)
            # And add the finish actions to everything that pointed at it
            for trans in sub_dfa.transitions_pointing_to(self.handler_node):
                trans.attach(*self.incoming_handler_actions).attach(*self.after_actions)

        # If there is a next node, append it
        if self.next is not None:
            sub_dfa.append_after(self.next.convert(current_error_handlers), [current_error_handlers[ErrorReasons.NO_MATCH], self.handler_node])

        return sub_dfa

class Macro:
    def __init__(self, name_token: lark.Token, parse_tree: lark.Tree):
        self.name = name_token.value
        self.parse_tree = parse_tree
        ProgramData.imbue(self, DebugTag.SOURCE_LINE, name_token.line)
        ProgramData.imbue(self, DebugTag.NAME, "macro " + self.name)

# =========
# PARSE CTX
# =========

class ParseCtx:
    def __init__(self, parse_tree: lark.Tree):
        self._parse_tree = parse_tree
        self.macros = {} # all macros, name --> AST
        self.state_object_spec = {}
        self.hooks = []
        self.ast = None
        self.start_actions = []
        self.generic_fail_state = DFState()

        self.exception_handlers = defaultdict(lambda: self.generic_fail_state)  # normal ErrorReason -> State
        self.break_handlers = {}      # "string name" -> Action
        self.innermost_break_handler = None  # just an Action
    
    def parse(self):
        # Parse state_object_spec
        for out in self._parse_tree.find_data("out_decl"):
            out_obj = self._parse_out_decl(out)
            if out_obj.name in self.state_object_spec:
                raise DuplicateDefinitionError("output variable", out_obj, out_obj.name)
            self.state_object_spec[out_obj.name] = out_obj
        # Parse macros
        for macro in self._parse_tree.find_data("macro_decl"):
            macro_obj = Macro(macro.children[0], macro.children[1:])
            if macro_obj.name in self.macros:
                raise DuplicateDefinitionError("macro", macro_obj, macro_obj.name)
            self.macros[macro_obj.name] = macro_obj

        for hook in self._parse_tree.find_data("hook_decl"):
            if hook.children[0].value in self.hooks:
                raise DuplicateDefinitionError("hook", hook.children[0], hook.children[0].value)
            self.hooks.append(hook.children[0].value)
        # Parse main
        parser_decl = next(self._parse_tree.find_data("parser_decl"))
        self.ast = self._parse_stmt_seq(parser_decl.children)

        if isinstance(self.ast, ActionNode):
            self.start_actions = self.ast.actions
            self.ast = self.ast.get_next()

    def _convert_string(self, escaped_string: str):
        """
        Parse the string `escaped_string`, which is the direct token from lark (still with quotes and escapes)
        """

        contents = escaped_string[1:-1]
        result = ""
        i = 0
        while i < len(contents):
            if contents[i] != '\\':
                result += contents[i]
                i += 1
            else:
                i += 1
                if contents[i] == "x" or contents[i] == "u":
                    raise NotImplementedError("don't support uescapes yet")
                else:
                    result += {
                        'n': '\n',
                        'r': '\r',
                        't': '\t',
                        'b': '\b',
                        '0': '\x00',
                        '"': '"'
                    }[contents[i]]
                    i += 1
        return result

    def _parse_out_decl(self, decl: lark.Tree) -> OutputStorage:
        """
        Parse an output declaration
        """

        type_obj = decl.children[0]
        name = decl.children[1].value
        if len(decl.children) == 3:
            if type_obj.data != "str_type":
                default_value = self._parse_integer_expr(decl.children[2])
                if not default_value.is_literal():
                    raise IllegalParseTree("Default value for out-decl must be constant", decl.children[2])
            else:
                if decl.children[2].data != "string_const":
                    raise IllegalParseTree("Default value for string must be a string constant", decl.children[2])
                default_value = self._convert_string(decl.children[2].children[0].value)
        else:
            default_value = None

        if type_obj.data == "bool_type":
            return OutputStorage(OutputStorageType.BOOL, name, default_value=default_value)
        elif type_obj.data == "int_type":
            return OutputStorage(OutputStorageType.INT, name, default_value=default_value)
        elif type_obj.data == "enum_type":
            return OutputStorage(OutputStorageType.ENUM, name, default_value=default_value, enum_values=list(x.value for
                x in type_obj.children))
        elif type_obj.data == "str_type":
            return OutputStorage(OutputStorageType.STR, name, default_value=default_value, str_size=int(type_obj.children[0].value))

    def _parse_math_expr(self, expr: lark.Tree):
        """
        Parse a math expr [(something)]
        """

        if expr.data == "math_num":
            return ProgramData.imbue(ProgramData.imbue(LiteralIntegerExpr(int(expr.children[0].value)), DebugTag.SOURCE_LINE, expr.line), DebugTag.SOURCE_COLUMN, expr.column)
        elif expr.data == "math_var":
            try:
                return ProgramData.imbue(ProgramData.imbue(OutIntegerExpr(self.state_object_spec[expr.children[0].value]), DebugTag.SOURCE_LINE, expr.line), DebugTag.SOURCE_COLUMN, expr.column)
            except KeyError:
                raise UndefinedReferenceError("output", expr.children[0])
        elif expr.data == "builtin_math_var":
            ref = expr.children[0]
            if ref.value == "last":
                return ProgramData.imbue(ProgramData.imbue(LastCharIntegerExpr(), DebugTag.SOURCE_LINE, expr.line), DebugTag.SOURCE_COLUMN, expr.column)
            else:
                raise UndefinedReferenceError("builtin math variable", ref)
        elif expr.data == "sum_expr":
            return SumIntegerExpr([self._parse_integer_expr(x) for x in expr.children[::2]], [False, *(x.value == "-" for x in expr.children[1::2])])
        elif expr.data == "mul_expr":
            return MulIntegerExpr([self._parse_integer_expr(x) for x in expr.children[::2]], [MulIntegerExprOp.MUL, *(MulIntegerExprOp(x.value) for x in expr.children[1::2])])
        else:
            raise IllegalParseTree("Invalid math expression", expr)

    def _parse_integer_expr(self, expr: lark.Tree, into_storage: OutputStorage=None) -> IntegerExpr:
        """
        Parse an integer type expr (also has bool/etc.)
        """

        BANNED_TYPES = ["end_expr", "concat_expr", "regex", "string_const", "string_case_const"]
        if expr.data in BANNED_TYPES:
            raise IllegalParseTree("String-typed value encountered for integer-typed expression", expr)

        if expr.data == "number_const":
            val = LiteralIntegerExpr(int(expr.children[0].value))
            ProgramData.imbue(val, DebugTag.SOURCE_LINE, expr.children[0].line)
            ProgramData.imbue(val, DebugTag.SOURCE_COLUMN, expr.children[0].column)
            return val
        elif expr.data == "identifier_const":
            if into_storage is None:
                raise IllegalParseTree("Undefined enumeration value, no into_storage", expr)

            if into_storage.type != OutputStorageType.ENUM:
                raise IllegalParseTree("Use of enumeration constant for non-enumeration type output (perhaps you meant to use []?)", expr)

            if expr.children[0].value not in into_storage.enum_values:
                raise UndefinedReferenceError("enumeration constant", expr)
            
            val = LiteralIntegerExpr(expr.children[0].value, OutputStorageType.ENUM)
            ProgramData.imbue(val, DebugTag.SOURCE_LINE, expr.children[0].line)
            ProgramData.imbue(val, DebugTag.SOURCE_COLUMN, expr.children[0].column)
            return val
        elif expr.data == "bool_const":
            if into_storage is None:
                result_type = OutputStorageType.BOOL
            else:
                if into_storage.type not in [OutputStorageType.BOOL, OutputStorageType.INT]:
                    raise IllegalParseTree("Use of boolean expression in non integral-type", expr)
                result_type = into_storage.type

            try:
                return ProgramData.imbue(ProgramData.imbue(LiteralIntegerExpr({"true": 1, "false": 0}[expr.children[0].value], result_type), DebugTag.SOURCE_LINE, expr.line), DebugTag.SOURCE_COLUMN, expr.column)
            except KeyError as e:
                raise UndefinedReferenceError("boolean constant", expr.children[0]) from e
        elif expr.data in ["math_num", "math_var", "builtin_math_var", "sum_expr", "mul_expr"]:
            return self._parse_math_expr(expr)
        else:
            raise IllegalParseTree("Invalid expression in integer expr", expr)


    def _parse_match_expr(self, expr: lark.Tree) -> Match:
        """
        Parse a match expression into a match object
        """
        if expr.data == "string_const":
            actual_content = expr.children[0]
            match = DirectMatch(self._convert_string(actual_content.value))
            ProgramData.imbue(match, DebugTag.SOURCE_LINE, actual_content.line)
            ProgramData.imbue(match, DebugTag.SOURCE_COLUMN, actual_content.column)
            return match
        elif expr.data == "string_case_const":
            actual_content = expr.children[0]
            match = CaseDirectMatch(self._convert_string(actual_content.value))
            ProgramData.imbue(match, DebugTag.SOURCE_LINE, actual_content.line)
            ProgramData.imbue(match, DebugTag.SOURCE_COLUMN, actual_content.column)
            return match
        elif expr.data == "regex":
            return RegexMatch(expr)
        elif expr.data == "end_expr":
            if not ProgramData.do(ProgramFlag.EOF_SUPPORT):
                raise IllegalParseTree("end match but EOF support is not enabled", expr)
            match = EndMatch()
            ProgramData.imbue(match, DebugTag.SOURCE_LINE, expr.line)
            ProgramData.imbue(match, DebugTag.SOURCE_COLUMN, expr.column)
            return match
        elif expr.data == "concat_expr":
            return ConcatMatch(list(self._parse_match_expr(x) for x in expr.children))
        else:
            raise IllegalParseTree("Invalid expression in match expression context", expr)

    def _parse_assign_stmt(self, stmt: lark.Tree, is_append) -> Node:
        """
        Parse an assignment into its underlying action
        """

        # Resolve the target
        targeted = stmt.children[0].value
        if targeted not in self.state_object_spec:
            raise UndefinedReferenceError("output", stmt.children[0])
        targeted = self.state_object_spec[targeted]

        if targeted.type == OutputStorageType.STR and is_append:
            # Check if this is a math expression (only valid append type other than match)
            if stmt.children[1].data in ["math_num", "math_var", "builtin_math_var", "sum_expr", "mul_expr"]:
                # Create an AppendCharTo action
                return ActionNode(AppendCharTo(self.exception_handlers[ErrorReasons.OUT_OF_SPACE], self._parse_math_expr(stmt.children[1]), targeted))
            # Create an append expression
            match_node = MatchNode(self._parse_match_expr(stmt.children[1]))
            match_node.match.attach(AppendTo(self.exception_handlers[ErrorReasons.OUT_OF_SPACE], targeted))
            return match_node
        elif targeted.type == OutputStorageType.STR:
            if stmt.children[1].data != "string_const":
                raise IllegalParseTree("String assignment only supports string constants, did you mean +=?", stmt.children[1])
            else:
                return ActionNode(SetToStr(self._convert_string(stmt.children[1].children[0].value), targeted))
        elif not is_append:
            return ActionNode(SetTo(self._parse_integer_expr(stmt.children[1], targeted), targeted))

    def _parse_case_clause(self, clause: lark.Tree):
        result_set = set()
        target_dfa = None
        
        offset = None 

        for j, predicate in enumerate(clause.children):
            if predicate.data == "else_predicate":
                result_set.add(None)
            elif predicate.data == "expr_predicate":
                result_set.add(self._parse_match_expr(predicate.children[0]))
            else:
                offset = j
                break

        if offset is not None:
            target_dfa = self._parse_stmt_seq(clause.children[offset:])

        return frozenset(result_set), target_dfa

    def _parse_stmt(self, stmt: lark.Tree) -> Node:
        """
        Parse a statement into a node
        """
        if stmt.data == "match_stmt":
            return MatchNode(self._parse_match_expr(stmt.children[0]))
        elif stmt.data == "wait_stmt":
            return MatchNode(WaitMatch(self._parse_match_expr(stmt.children[0])))
        elif stmt.data == "call_stmt":
            name = stmt.children[0].value
            try:
                return ProgramData.imbue(self._parse_stmt_seq(self.macros[name].parse_tree), DebugTag.PARENT, self.macros[name])
            except KeyError:
                if name in self.hooks:
                    return ActionNode(ProgramData.imbue(ProgramData.imbue(CallHook(name), DebugTag.SOURCE_LINE, stmt.line), DebugTag.SOURCE_COLUMN, stmt.column))
                else:
                    raise UndefinedReferenceError("macro", stmt.children[0])
        elif stmt.data == "finish_stmt":
            act = FinishAction()
            ProgramData.imbue(act, DebugTag.SOURCE_LINE, stmt.line)
            ProgramData.imbue(act, DebugTag.SOURCE_COLUMN, stmt.column)
            return ActionNode(act)
        elif stmt.data == "break_stmt":
            if stmt.children:
                target_name = stmt.children[0].value
                if target_name not in self.break_handlers:
                    raise UndefinedReferenceError("loop", stmt.children[0])
                else:
                    act = self.break_handlers[target_name]
            else:
                act = self.innermost_break_handler
            if act is None:
                raise IllegalParseTree("Break outside of loop", stmt)
            ProgramData.imbue(act, DebugTag.SOURCE_LINE, stmt.line)
            ProgramData.imbue(act, DebugTag.SOURCE_COLUMN, stmt.column)
            return ActionNode(act)
        elif stmt.data in ("assign_stmt", "append_stmt"):
            return self._parse_assign_stmt(stmt, stmt.data == "append_stmt")
        elif stmt.data == "case_stmt":
            # Find all of the matches
            return CaseNode({k: v for k, v in (self._parse_case_clause(x) for x in stmt.children)})
        elif stmt.data == "optional_stmt":
            return OptionalNode(self._parse_stmt_seq(stmt.children))
        elif stmt.data == "loop_stmt":
            loop_name = None
            statements = stmt.children[:]
            if len(stmt.children) and isinstance(stmt.children[0], lark.Token) and stmt.children[0].type == "IDENTIFIER":
                loop_name = stmt.children[0].value
                statements = stmt.children[1:]
            loop_node = LoopNode(loop_name)
            previous_break = self.innermost_break_handler
            self.break_handlers[loop_name] = loop_node.get_break_handler()
            self.innermost_break_handler = loop_node.get_break_handler()
            child_node = self._parse_stmt_seq(statements)
            self.innermost_break_handler = previous_break
            loop_node.set_child(child_node)
            return loop_node
        elif stmt.data == "try_stmt":
            catch_block = stmt.children[-1]
            # try to see if there are options, otherwise use all
            catch_block_stmts = catch_block.children[:]
            catch_handles = set(ErrorReasons)

            if catch_block.children and catch_block.children[0].data == "catch_options":
                catch_handles = set()
                catch_block_stmts = catch_block.children[1:]

                for option in catch_block.children[0].children:
                    try:
                        catch_handles.add(ErrorReasons(option.value))
                    except ValueError as e:
                        raise UndefinedReferenceError("error type", option) from e

            body_block_stmts = stmt.children[:-1]
            try_node = TryExceptNode(catch_handles)

            prior_error_reasons = self.exception_handlers.copy()
            self.exception_handlers.update({x: try_node.get_handler() for x in catch_handles})
            
            try_node.set_body(self._parse_stmt_seq(body_block_stmts))
            try_node.set_handler(self._parse_stmt_seq(catch_block_stmts))
            
            self.exception_handlers = prior_error_reasons

            return try_node
        else:
            raise IllegalParseTree("Unknown statement", stmt)

    def _parse_stmt_seq(self, stmts: List[lark.Tree]) -> Node:
        """
        Parse a set of statements into one big node
        """

        next_node = None
        for stmt in reversed(stmts):
            node = self._parse_stmt(stmt)
            if node.get_next() is not None:
                # We need to find the actual end
                end_node = node
                while end_node.get_next() is not None:
                    end_node = end_node.get_next()
                end_node.set_next(next_node)
            else:
                node.set_next(next_node)
            next_node = node
        return next_node

class DfaCompileCtx:
    def __init__(self, parse_ctx: ParseCtx):
        self.state_object_spec = parse_ctx.state_object_spec
        self.hooks = parse_ctx.hooks
        self.ast = parse_ctx.ast
        self.start_actions = parse_ctx.start_actions
        self.generic_fail_state = parse_ctx.generic_fail_state
        self.dfa = None

    def _optimize_remove_inaccessible(self):
        if not ProgramData.do(ProgramFlag.REMOVE_INACCESIBLE_STATES):
            return 0
        accessible = self.dfa.dfs()
        mod = 0
        for i in self.dfa.states.copy():
            if i not in accessible:
                mod += 1
                self.dfa.states.remove(i)
        dprint[ProgramFlag.VERBOSE_OPTIMIZE_RESULTS]("removed {} nonaccessible".format(mod))
        return mod

    def _optimize_minimize_dfa(self):
        # TODO: make this use Hopcroft's algorithm or something similar
        # 
        # on hold because the necessary logic for dealing with actions is painful
        return 0

    def _optimize_simplify_transition_matches(self):
        """
        Simplify transitions that match overlapping symbols.
        e.g.: () ---a, else----> () ==> () ---else----> ()
        """

        mod = 0
        if not ProgramData.do(ProgramFlag.SIMPLIFY_ELSE_CONDITIONS):
            return 0

        for state in self.dfa.states:
            for transition in state.transitions:
                if len(transition.on_values) > 1 and DFTransition.Else in transition.on_values:
                    dprint[ProgramFlag.VERBOSE_SIMPLIFY_TM]("simplfying {} with Else".format(transition))
                    transition.on_values = [DFTransition.Else]
                    mod += 1

        dprint[ProgramFlag.VERBOSE_OPTIMIZE_RESULTS]("simplified {} transitions".format(mod))
        return mod

    def _verify_fallthrough_loop(self):
        """
        Check if any transitions pointing to their own state are fallthrough, which is invalid
        since it would cause an infinite loop
        """

        for state in self.dfa.states:
            for transition in state.transitions:
                if transition.target == state and transition.is_fallthrough:
                    raise IllegalDFAStateError("Inifinite loop due to self-referential fallthrough", transition)
        

    def compile(self):
        """
        Convert the AST into a (potentially optimized) DFA.
        """

        self.dfa = self.ast.convert(defaultdict(lambda: self.generic_fail_state))
        self.dfa.add(self.generic_fail_state)

        while self._optimize_remove_inaccessible() + self._optimize_simplify_transition_matches():
            pass

        # verify correctness of DFA
        self._verify_fallthrough_loop()

class Outputter:
    SHIFT_WIDTH = 4

    def __init__(self, indent=0, target=None):
        if target:
            self.result = target
        else:
            self.result = io.StringIO()
        self.indent = indent

    def __enter__(self):
        return Outputter(self.indent + Outputter.SHIFT_WIDTH, self.result)

    def __exit__(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        self.result.write(" " * self.indent)
        print(*args, **kwargs, file=self.result)

    def value(self):
        return self.result.getvalue()

    def __iadd__(self, tgt):
        self.result.write(textwrap.indent(tgt, " "*self.indent))
        return self

class CodegenCtx:
    def __init__(self, dfa_compile_ctx: DfaCompileCtx, program_name: str):
        self.start_actions = dfa_compile_ctx.start_actions
        self.hooks = dfa_compile_ctx.hooks
        self.dfa = dfa_compile_ctx.dfa
        self.state_object_spec: List[OutputStorage] = list(dfa_compile_ctx.state_object_spec.values())
        self.generic_fail_state = dfa_compile_ctx.generic_fail_state
        self.program_name = program_name

    def generate_header(self):
        result = Outputter()
        if ProgramData.do(ProgramFlag.USE_PRAGMA_ONCE):
            result.add("#pragma once")
        else:
            result.add(f"#ifndef {self.program_name.upper()}_H")
            result.add(f"#define {self.program_name.upper()}_H")
        result.add("#include <stdbool.h>")
        result.add("#include <stdint.h>")
        result.add()
        result.add(f"// ============================" + "=" * len(self.program_name))
        result.add(f"// header file for nmfu parser {self.program_name}")
        result.add(f"// ============================" + "=" * len(self.program_name))
        result.add()

        if ProgramData.do(ProgramFlag.USE_CPLUSPLUS_GUARD):
            result.add("#ifdef __cplusplus")
            result.add("extern \"C\" {")
            result.add("#endif")

        result += self._generate_state_object_decl()

        result.add()
        result.add(f"enum {self.program_name}_result {{")
        with result as enum_content:
            enum_content.add(f"{self.program_name.upper()}_OK,")
            enum_content.add(f"{self.program_name.upper()}_FAIL,")
            enum_content.add(f"{self.program_name.upper()}_DONE,")
            # todo: allow custom
        result.add("};")
        result.add(f"typedef enum {self.program_name}_result {self.program_name}_result_t;")
        result.add()

        result.add(f"{self.program_name}_result_t {self.program_name}_start({self.program_name}_state_t *state);")
        result.add(f"{self.program_name}_result_t {self.program_name}_feed(const uint8_t *start, const uint8_t *end, {self.program_name}_state_t *state);")
        if ProgramData.do(ProgramFlag.EOF_SUPPORT):
            result.add(f"{self.program_name}_result_t {self.program_name}_end({self.program_name}_state_t *state);")
        if ProgramData.do(ProgramFlag.DYNAMIC_MEMORY):
            result.add(f"void {self.program_name}_free({self.program_name}_state_t *state);")
        
        if ProgramData.do(ProgramFlag.HOOK_GLOBAL):
            for hook in self.hooks:
                result.add(f"void {self.program_name}_{hook}_hook({self.program_name}_state_t *state, uint8_t inval);")

        if ProgramData.do(ProgramFlag.USE_CPLUSPLUS_GUARD):
            result.add("#ifdef __cplusplus")
            result.add("}")
            result.add("#endif")

        if not ProgramData.do(ProgramFlag.USE_PRAGMA_ONCE):
            result.add("#endif")

        return result.value()

    def generate_source(self):
        result = Outputter()
        result.add(f"// ============================" + "=" * len(self.program_name))
        result.add(f"// source file for nmfu parser {self.program_name}")
        result.add(f"// ============================" + "=" * len(self.program_name))
        result.add(f"#include \"{self.program_name}.h\"")
        result.add("#include <string.h>")
        if ProgramData.do(ProgramFlag.DYNAMIC_MEMORY):
            result.add("#include <stdlib.h>")
        result.add()
        result += self._generate_start_implementation()
        result += self._generate_feed_implementation()
        if ProgramData.do(ProgramFlag.EOF_SUPPORT):
            result += self._generate_end_implementation()
        if ProgramData.do(ProgramFlag.DYNAMIC_MEMORY):
            result += self._generate_free_implementation()
        return result.value()

    def _integer_containing(self, maxval, signed=True):
        # TODO: customization point for non 32-bit machines
        if signed:
            if maxval is None:
                return "int32_t"
            elif maxval < 128:
                return "int8_t"
            elif maxval < 32768:
                return "int16_t"
            elif maxval < (1 << 31):
                return "int32_t"
            else:
                return "intmax_t"
        else:
            if maxval is None:
                return "uint32_t"
            elif maxval < 256:
                return "uint8_t"
            elif maxval < 65536:
                return "uint16_t"
            elif maxval < (1 << 32):
                return "uint32_t"
            else:
                return "uintmax_t"

    def _generate_state_object_decl(self):
        """
        Generate the ```
        struct program_name_state {
            
        };
        ```
        object, along with any enums.
        """

        result = Outputter()
        
        for out_decl in self.state_object_spec:
            if out_decl.type == OutputStorageType.ENUM:
                result += self._generate_out_enum(out_decl)
                result.add()

        if ProgramData.do(ProgramFlag.HOOK_PER_STATE):
            # Add a typedef
            result.add("// hook typedef")
            result.add(f"struct {self.program_name}_state;")
            result.add(f"typedef void (*{self.program_name}_hook_t)(struct {self.program_name}_state *, uint8_t);")

        result.add("// state object")
        result.add(f"struct {self.program_name}_state", "{")
        with result as contents:
            if self.state_object_spec:
                contents.add("struct {")
                with contents as out_contents:
                    for out_decl in self.state_object_spec:
                        out_contents.add(self._get_state_object_out_declaration(out_decl) + ";")
                contents.add("} c;")

            if any(x.type == OutputStorageType.STR for x in self.state_object_spec):
                for out_str in (x for x in self.state_object_spec if x.type == OutputStorageType.STR):
                    contents.add(self._integer_containing(out_str.str_size, signed=False), f"{out_str.name}_counter;")

            contents.add(self._integer_containing(len(self.dfa.states), signed=False), "state;")

            # Add a user ptr if desired.
            if ProgramData.do(ProgramFlag.INCLUDE_USER_PTR):
                contents.add("void * userptr;")

            # Add hooks if desired
            if ProgramData.do(ProgramFlag.HOOK_PER_STATE):
                for hook in self.hooks:
                    contents.add(f"{self.program_name}_hook_t {hook}_hook;")

        result.add("};")
        result.add(f"typedef struct {self.program_name}_state {self.program_name}_state_t;")
        return result.value()

    def _get_state_object_out_declaration(self, out_decl: OutputStorage):
        """
        Get the type of a state object out decl
        """
        
        if out_decl.type != OutputStorageType.STR:
            return {
                OutputStorageType.INT: self._integer_containing(None),
                OutputStorageType.ENUM: f"{self.program_name}_out_{out_decl.name}_t",
                OutputStorageType.BOOL: "bool",
            }[out_decl.type] + " " + out_decl.name
        else:
            if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_IN_STRUCT):
                return f"char {out_decl.name}[{out_decl.str_size}]"
            else:
                return f"char * {out_decl.name}"

    def _generate_out_enum(self, out_decl: OutputStorage):
        """
        Generate an output enum type
        """

        result = Outputter()
        result.add("// enum for output {}".format(out_decl.name))
        result.add(f"enum {self.program_name}_out_{out_decl.name} {{")
        
        with result as contents:
            for val in out_decl.enum_values:
                contents.add(f"{self.program_name.upper()}_{out_decl.name.upper()}_{val},")

        result.add("};")
        result.add(f"typedef enum {self.program_name}_out_{out_decl.name} {self.program_name}_out_{out_decl.name}_t;")
        return result.value()

    def _convert_literal_value(self, literal, out_expr: OutputStorage):
        if out_expr.type == OutputStorageType.ENUM:
            return f"{self.program_name.upper()}_{out_expr.name.upper()}_{literal.upper()}"
        elif out_expr.type == OutputStorageType.BOOL:
            return "true" if literal else "false"
        elif out_expr.type == OutputStorageType.INT:
            return str(literal)
        else:
            return '"{}"'.format(self._escape_string(literal))

    def _generate_code_for_int_expr(self, intexpr: IntegerExpr, ctx: IntegerExprUseContext, out_expr: OutputStorage):
        """
        Generate C expression that evaluates to the value of intexpr, used in context ctx and which will fill out_expr (or have its type)
        """

        if any(x == ctx for x in intexpr.get_invalid_contexts()):
            raise IllegalIntExpr("Illegal use of integer expression in context {}".format(ctx.name), intexpr)

        if intexpr.result_type() != out_expr.type:
            raise IllegalIntExpr("Mismatched types for target storage", intexpr)

        if isinstance(intexpr, LiteralIntegerExpr):
            return self._convert_literal_value(intexpr.get_literal_result(), out_expr)
        elif isinstance(intexpr, OutIntegerExpr):
            return f"state->c.{out_expr.name}"
        elif isinstance(intexpr, LastCharIntegerExpr):
            return f"({self._integer_containing(None)})(inval)" # name of the last character value
        elif isinstance(intexpr, SumIntegerExpr):
            result = f"({self._generate_code_for_int_expr(intexpr.children[0], ctx, out_expr)})"
            for child, operator in zip(intexpr.children[1:], intexpr.negate[1:]):
                result += " "
                result += "-" if operator else "+"
                result += " "
                result += f"({self._generate_code_for_int_expr(child, ctx, out_expr)})"
            return result
        elif isinstance(intexpr, MulIntegerExpr):
            result = f"({self._generate_code_for_int_expr(intexpr.children[0], ctx, out_expr)})"
            for child, operator in zip(intexpr.children[1:], intexpr.divide[1:]):
                result += " "
                result += operator.value
                result += " "
                result += f"({self._generate_code_for_int_expr(child, ctx, out_expr)})"
            return result
        else:
            raise NotImplementedError("unsupported intexpr type")

    def _escape_string(self, value: str):
        result = ""
        bytes_value = value.encode('utf-8')
        for i in bytes_value:
            if chr(i) in ["\\", '"']:
                result += "\\" + i
            elif not (32 <= i < 127):
                result += "\\x{:02x}".format(i)
            else:
                result += chr(i)
        return result

    def _generate_set_string(self, value: str, into: OutputStorage):
        """
        Generate code that sets value into into

        : strncpy(state->c.into, "value", into.str_size)
        """

        return f"strncpy(state->c.{into.name}, \"{self._escape_string(value)}\", {into.str_size});"

    def _generate_action_implementation(self, action: Action, is_start: bool = False):
        result = Outputter()
        if isinstance(action, FinishAction):
            result.add(f"return {self.program_name.upper()}_DONE;")
        elif isinstance(action, SetTo):
            target = action.into_storage
            value = self._generate_code_for_int_expr(action.value_expr, IntegerExprUseContext.ASSIGN_INITIAL if is_start else IntegerExprUseContext.ASSIGN_ON_MATCH, target)
            result.add(f"state->c.{target.name} = {value};")
        elif isinstance(action, SetToStr):
            # Check if we need to allocate
            if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC_ON_DEMAND) and action.into_storage.default_value is None:  # if it wasn't None it'd be allocated in the start()
                if is_start:
                    # if we're at the start, and there's no default value, and on demand is in effect, there's no possible way for state->c to have any value other than NULL
                    result.add(f"state->c.{action.into_storage.name} = malloc({action.into_storage.str_size});")
                else:
                    result.add(f"if (!state->c.{action.into_storage.name}) state->c.{action.into_storage.name} = malloc({action.into_storage.str_size});")
            result.add(self._generate_set_string(action.value_expr, action.into_storage))
            result.add(f"state->{action.into_storage.name}_counter = {len(action.value_expr)};")
        elif isinstance(action, (AppendTo, AppendCharTo)):
            # Check if we need to allocate
            if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC_ON_DEMAND) and action.into_storage.default_value is None:  # if it wasn't None it'd be allocated in the start()
                result.add(f"if (!state->c.{action.into_storage.name}) state->c.{action.into_storage.name} = malloc({action.into_storage.str_size});")
            # We treat the size given in by the user as including a terminating null
            # Admittedly, this is a little hit or miss, but if they give too long a string it's really their fault so /shrug
            # TODO: customization point to disable this and not put terminating nulls on strings
            result.add(f"if (state->{action.into_storage.name}_counter == {action.into_storage.str_size-1}) state->state = {self.dfa.states.index(action.end_target)};")
            result.add("else {")
            with result as body:
                target_expression = "inval" if isinstance(action, AppendTo) else self._generate_code_for_int_expr(
                    action.append_value, IntegerExprUseContext.ASSIGN_INITIAL if is_start else IntegerExprUseContext.ASSIGN_ON_MATCH, OutputStorage(OutputStorageType.INT, "$appendctx")
                )
                body.add(f"state->c.{action.into_storage.name}[state->{action.into_storage.name}_counter++] = (char)({target_expression});")
                body.add(f"state->c.{action.into_storage.name}[state->{action.into_storage.name}_counter] = 0;")
            result.add("}")
        elif isinstance(action, CallHook):
            parm = "0" if is_start else "inval"
            if ProgramData.do(ProgramFlag.HOOK_GLOBAL):
                result.add(f"{self.program_name}_{action.name}_hook(state, {parm});")
            elif ProgramData.do(ProgramFlag.HOOK_PER_STATE):
                result.add(f"(state->{action.name}_hook)(state, {parm});")
            else:
                raise IllegalDFAStateError("Attempt to use hook while no hook method is enabled", action)
        return result.value()

    def _generate_start_implementation(self):
        result = Outputter()

        result.add(f"{self.program_name}_result_t {self.program_name}_start({self.program_name}_state_t *state)", "{")

        with result as contents:
            # Initialize all state variables
            # First, any (if specified) default values.
            for out_expr in self.state_object_spec:
                # If a string, first init the counter value
                if out_expr.type == OutputStorageType.STR:
                    counter_val = 0
                    if out_expr.default_value is not None:
                        counter_val = len(out_expr.default_value)
                    contents.add("// initialize append counter for", out_expr.name)
                    contents.add(f"state->{out_expr.name}_counter = {counter_val};")
                if out_expr.default_value is not None:
                    contents.add("// initialize default for", out_expr.name)
                    if out_expr.type == OutputStorageType.STR:
                        # Also allocate the data if not included
                        if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC):
                            contents.add(f"state->c.{out_expr.name} = malloc({out_expr.str_size});")
                        contents.add(self._generate_set_string(out_expr.default_value, out_expr))
                    else:
                        contents.add(f"state->c.{out_expr.name} = {self._generate_code_for_int_expr(out_expr.default_value, IntegerExprUseContext.ASSIGN_INITIAL, out_expr)};")

            # Set starting state
            contents.add("// set starting state")
            contents.add(f"state->state = {self.dfa.states.index(self.dfa.starting_state)};")

            if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC):
                for out_expr in self.state_object_spec:
                    if out_expr.type != OutputStorageType.STR or out_expr.default_value is not None:
                        continue # these cases are handled above
                    if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC_ON_DEMAND):
                        contents.add(f"// set {out_expr.name} to null")
                        contents.add(f"state->c.{out_expr.name} = NULL;")
                    else:
                        contents.add(f"// allocate space for {out_expr.name}")
                        contents.add(f"state->c.{out_expr.name} = malloc({out_expr.str_size});")

            # Run any start actions
            if self.start_actions:
                contents.add("// run start actions")
            for action in self.start_actions:
                contents += self._generate_action_implementation(action, True)
        
            contents.add(f"return {self.program_name.upper()}_OK;")
        result.add("}")
        return result.value()

    def _generate_equal_check(self, on_value):
        return f"inval == {ord(on_value)} /* {on_value!r} */"

    def _generate_range_check(self, min_cpoint, max_cpoint):
        """
        Get a range check for min_cpoint <= x <= max_cpoint
        """

        return f"({ord(min_cpoint)} <= inval && inval <= {ord(max_cpoint)} /* {repr(min_cpoint)} - {repr(max_cpoint)} */)"

    def _generate_condition_for_transition(self, transition: DFTransition):
        """
        Generate a string which can be inserted into an if () condition that returns true
        if the transition should be taken
        """

        # The general approach here is to use the various match techniques to extract "better" conditions, until
        # being left with either a few sparse values in an on_values copy or indeed nothing. This forms the initial
        # value of the result string

        on_values_remaining = transition.on_values[:]
        checks = []

        if ProgramData.do(ProgramFlag.COLLAPSE_TRANSITION_RANGES) and len(on_values_remaining) >= ProgramData.option(ProgramOption.COLLAPSED_RANGE_LENGTH):
            # sort the on values in ascending order
            on_values_remaining.sort(key=lambda x: x if isinstance(x, str) else '') # end goes at the beginning

            used = []  # store all removed values into an array
            start_idx = 1 if DFTransition.End in on_values_remaining else 0

            range_start = start_idx
            range_end = start_idx

            for i in range(start_idx + 1, len(on_values_remaining)):
                if ord(on_values_remaining[i-1]) + 1 == ord(on_values_remaining[i]):
                    range_end = i
                else:
                    if range_end - range_start >= ProgramData.option(ProgramOption.COLLAPSED_RANGE_LENGTH):
                        # this is a valid range
                        for j in range(range_start, range_end+1):
                            used.append(on_values_remaining[j])
                        checks.append(self._generate_range_check(on_values_remaining[range_start], on_values_remaining[range_end]))
                    range_start = i
                    range_end = i

            if range_end - range_start >= ProgramData.option(ProgramOption.COLLAPSED_RANGE_LENGTH):
                # this is a valid range
                for j in range(range_start, range_end+1):
                    used.append(on_values_remaining[j])
                checks.append(self._generate_range_check(on_values_remaining[range_start], on_values_remaining[range_end]))

            for x in used:
                on_values_remaining.remove(x)

        # Match the remaining ones with boring equals
        checks.extend(self._generate_equal_check(x) for x in on_values_remaining if x != DFTransition.End)

        result = " || ".join(checks)

        condition_strs = [self._generate_condition_for_transition_condition(x) for x in transition.conditions]
        if condition_strs:
            result = f"({result}) && {' && '.join(condition_strs)}"
        return result

    def _generate_condition_for_transition_condition(self, trans_condition):
        return "false"

    def _generate_transition_body(self, transition: DFTransition, from_end=False):
        transition_body = Outputter()
        # Set the next state
        try:
            transition_body.add(f"state->state = {self.dfa.states.index(transition.target)};")
        except ValueError:
            transition_body.add("// terminating state")
        # Generate actions
        for action in transition.actions:
            transition_body.add()
            transition_body.add(f"// action {action!r} ")
            transition_body += self._generate_action_implementation(action)
        # Check if we should fallthrough and generate a goto
        if transition.is_fallthrough:
            transition_body.add(f"// fallthrough")
            transition_body.add(f"goto fall_{self.dfa.states.index(transition.target)};")
        # Otherwise, if this state is targeting an accept state, return DONE instead of OK
        elif transition.target in self.dfa.accepting_states and not ProgramData.do(ProgramFlag.STRICT_DONE_TOKEN_GENERATION):
            transition_body.add("// immediately return DONE")
            transition_body.add(f"return {self.program_name.upper()}_DONE;")
        # Normally, though, just generate a jump to the next jpto
        elif not from_end:
            if transition.target in self.dfa.states:
                transition_body.add(f"if (++start == end) return {self.program_name.upper()}_OK;");
                transition_body.add("inval = *start;")
                if all(x.get_target_override_mode() == ActionOverrideMode.NONE for x in transition.actions):
                    transition_body.add(f"goto jpto_{self.dfa.states.index(transition.target)};");
                else:
                    # use the repeatswitch case
                    transition_body.add(f"goto repeatswitch;");
            else:
                pass # terminating state
        return transition_body.value()

    def _generate_switch_body(self, state: DFState):
        result = Outputter()

        # Split transitions into else groups
        else_transitions = list(x for x in state.transitions if DFTransition.Else in x.on_values and x.conditions)
        try:
            actual_else_transition = next(state.all_transitions_for((DFTransition.Else,)))
        except StopIteration:
            actual_else_transition = None

        result.add("// transitions")
        generated_if = False
        
        # Create all transition if cases
        for j, transition in enumerate((x for x in state.transitions if x not in else_transitions and x != actual_else_transition)):
            cond_name = "else if"
            conditions = self._generate_condition_for_transition(transition)
            if not conditions:
                continue
            if not generated_if:
                generated_if = True
                cond_name = "if"
            result.add(f"{cond_name} ({conditions}) {{")
            with result as transition_body:
                transition_body += self._generate_transition_body(transition)
            result.add("}")

        for j, transition in enumerate(else_transitions):
            cond_name = "else if"
            if not generated_if:
                generated_if = True
                cond_name = "if"
            # TODO: condition else
            result.add(f"{cond_name} (/* todo conditions */ false) {{}}")

        if actual_else_transition:
            if generated_if:
                result.add("else {")
            with result as transition_body:
                transition_body += self._generate_transition_body(actual_else_transition)
            if generated_if:
                result.add("}")
        if state in self.dfa.accepting_states:
            result.add(f"return {self.program_name.upper()}_DONE;")
        else:
            result.add(f"return {self.program_name.upper()}_OK;")
        return result.value()

    def _generate_feed_implementation(self):
        result = Outputter()

        result.add(f"{self.program_name}_result_t {self.program_name}_feed(const uint8_t *start, const uint8_t *end, {self.program_name}_state_t *state) {{")
        with result as contents:
            # Generate the `inval` variable
            contents.add("uint8_t inval = *start;");
            contents.add();
            # Generate a target for states with actions that modify the state in an unpredictable way
            contents.add("repeatswitch:");
            # The body of feed is a massive switch statement that has a bunch of internal gotos
            contents.add("switch (state->state) {")
            for idx, state in enumerate(self.dfa.states):
                # Emit the case label
                contents.add(f"case {idx}:")
                # Emit goto target for fallthroughs if anything falls here (these are separate to make it slightly easier to read)
                if any(x.is_fallthrough for x in self.dfa.transitions_pointing_to(state)):
                    contents.add(f"fall_{idx}:")
                contents.add(f"jpto_{idx}:")
                with contents as state_body:
                    # Is this a normal state
                    if state is not self.generic_fail_state:
                        state_body += self._generate_switch_body(state)
                    else:
                        state_body.add(f"return {self.program_name.upper()}_FAIL;")

            contents.add(f"default: return {self.program_name.upper()}_FAIL;")
            contents.add("}")

        result.add("}")
        return result.value()

    def _generate_end_switch_body(self, state: DFState):
        result = Outputter()

        # Find all transitions that operate on End
        end_transitions = list(x for x in state.transitions if DFTransition.End in x.on_values and x.conditions)
        try:
            unconditional_end_transition = next(state.all_transitions_for((DFTransition.End,)))
        except StopIteration:
            unconditional_end_transition = None

        result.add("// possible end transitions")
        generated_if = False
        
        # Create all transitions for possible conditions
        for j, transition in enumerate(end_transitions):
            cond_name = "else if"
            if j == 0 and not generated_if:
                generated_if = True
                cond_name = "if"
            # TODO: condition else
            result.add(f"{cond_name} (/* todo conditions */ false) {{}}")

        if unconditional_end_transition:
            if generated_if:
                result.add("else {")
            with result as transition_body:
                transition_body += self._generate_transition_body(unconditional_end_transition, True)
            if generated_if:
                result.add("}")
        else:
            if generated_if:
                result.add("else")
        if state in self.dfa.accepting_states:
            result.add(f"return {self.program_name.upper()}_DONE;")
        else:
            result.add(f"return {self.program_name.upper()}_FAIL;")
        return result.value()
    
    def _generate_end_implementation(self):
        result = Outputter()

        result.add(f"{self.program_name}_result_t {self.program_name}_end({self.program_name}_state_t *state) {{")
        result.add(f"#define inval 255") # generate a define for this so that hooks still work
        with result as contents:
            # Generate a big switch statement for all states
            contents.add("switch (state->state) {")
            for idx, state in enumerate(self.dfa.states):
                # Emit the case label
                contents.add(f"case {idx}:")
                # Emit goto target for fallthroughs if anything falls here (these are separate to make it slightly easier to read)
                if any(x.is_fallthrough for x in self.dfa.transitions_pointing_to(state)):
                    contents.add(f"fall_{idx}:")
                with contents as state_body:
                    # Is this a normal state
                    if state is not self.generic_fail_state:
                        state_body += self._generate_end_switch_body(state)
                    else:
                        state_body.add(f"return {self.program_name.upper()}_FAIL;")

            contents.add(f"default: return {self.program_name.upper()}_FAIL;")
            contents.add("}")

        result.add(f"#undef inval")
        result.add("}")
        return result.value()

    def _generate_free_implementation(self):
        result = Outputter()

        result.add(f"void {self.program_name}_free({self.program_name}_state_t *state) {{")
        
        with result as contents:
            # If strings were allocated dynamically, free every string (rationale is that they will be nullptrs)
            if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC):
                for out_expr in self.state_object_spec:
                    if out_expr.type == OutputStorageType.STR:
                        contents.add(f"// free storage for {out_expr.name}")
                        contents.add(f"free(state->c.{out_expr.name});")

        result.add("}")
        return result.value()

# =============
# DEBUG DUMPERS
# =============

def debug_dump_dfa(dfa: DFA, out_name="dfa", highlight=None):
    if not debug_enabled:
        raise RuntimeError("Debugging was disabled! You probably need to install graphviz")

    g = graphviz.Digraph(name='dfa', comment=ProgramData.lookup(dfa, DebugTag.NAME))

    nodes = []
    edges = []

    for j, state in enumerate(dfa.states):
        shape = "circle"
        if state in dfa.accepting_states:
            shape = "doublecircle"
        elif state == dfa.starting_state:
            shape = "square"
        if state == highlight:
            shape = "triangle"
        g.node(str(id(state)), shape=shape, label=str(j))
        for transition in state.all_transitions():
            if not transition.target or not transition.on_values:
                continue
            label = ",".join(repr(x) for x in transition.on_values)
            label = graphviz.escape(label)
            is_real = True

            for action in transition.actions:
                acname = ProgramData.lookup(action, DebugTag.NAME, recurse_upwards=False)
                if acname:
                    label += "\n{}".format(acname)
                if action.get_target_override_mode() in [ActionOverrideMode.ALWAYS_GOTO_OTHER, ActionOverrideMode.MAY_GOTO_TARGET]:
                    g.edge(str(id(state)), str(id(action.get_target_override_target())), label=f"{acname} side effect")
                if action.get_target_override_mode() in [ActionOverrideMode.ALWAYS_GOTO_OTHER]:
                    is_real = False
            if is_real:
                g.edge(str(id(state)), str(id(transition.target)), label=label, style="dashed" if transition.is_fallthrough else "solid")

    g.render(out_name, format="pdf", cleanup=True)

def debug_dump_regexnfa(nfa: RegexNFA, out_name="nfa"):
    if not debug_enabled:
        raise RuntimeError("Debugging was disabled! You probably need to install graphviz")

    g = graphviz.Digraph(name='nfa', comment=ProgramData.lookup(nfa, DebugTag.NAME))

    nodes = []
    edges = []

    for j, state in enumerate(nfa.states):
        shape = "circle"
        if state in nfa.finishing_states:
            shape = "doublecircle"
        elif state == nfa.start_state:
            shape = "square"
        g.node(str(id(state)), shape=shape, label=str(j))
        for source, target in state.transitions.items():
            label = "^" + repr(source.chars).replace("frozenset", "") if isinstance(source, InvertedRegexCharClass) else repr(source.chars).replace("frozenset", "")
            label = graphviz.escape(label)
            g.edge(str(id(state)), str(id(target)), label=label)
        for target in state.epsilon_moves:
            label = "e"
            g.edge(str(id(state)), str(id(target)), label=label)

    g.render(out_name, format="pdf", cleanup=True)

def debug_dump_ast(ast, out_name="ast", into=None, coming_from=None, make_id=None, make_subgraph=None):
    if into is None:
        g = graphviz.Digraph(name='ast')
        g.attr(ranksep="0.01", rankdir="LR", labeljust="l")
        idx = 0
        sidx = 0

        def _make_id():
            nonlocal idx
            idx += 1
            return f"a_{idx}"

        def _make_sid():
            nonlocal sidx
            sidx += 1
            return f"cluster_{idx}"

        g.node("c_0", style="invis")
        
        debug_dump_ast(ast, into=g, coming_from=["c_0"], make_id=_make_id, make_subgraph=_make_sid)

        g.render(out_name, format="pdf", cleanup=True)
        return

    def label_of(x):
        base = type(x).__name__
        if type(x).__repr__ != object.__repr__:
            base = repr(x)

        name = ProgramData.lookup(x, DebugTag.NAME, recurse_upwards=False)
        if name:
            base += f" ({name})"

        return graphviz.escape(base)

    def tie_to(sources, cfrm=None, extra_kwargs=None):
        if cfrm is None:
            cfrm = coming_from
        if type(sources) is str:
            sources = [sources]
        for i in sources:
            for x in cfrm:
                if x:
                    if extra_kwargs:
                        into.edge(x, i, **extra_kwargs)
                    else:
                        into.edge(x, i)
    
    with into.subgraph(name=make_subgraph()) as c:
        c.attr(label=label_of(ast))

        if isinstance(ast, ActionNode):
            # Add nodes for each action
            
            for action in ast.actions:
                node = make_id()
                c.node(node, label=label_of(action))
                tie_to(node)
                coming_from = [node]

        elif isinstance(ast, MatchNode):
            # List the adopted actions
            for name, l in zip(("start", "each", "finish"), (ast.match.start_actions, ast.match.char_actions, ast.match.finish_actions)):
                if not l:
                    continue
                node = make_id()
                c.node(node, f"{name}: {','.join(label_of(x) for x in l)}", style="filled", color="white")

            # Put the match in as a node
            node = make_id()
            c.node(node, label=label_of(ast.match))
            tie_to(node)

            coming_from = [node]

        elif isinstance(ast, CaseNode):
            start_node = make_id()
            c.node(start_node, "", style="filled", shape="circle", color="grey")
            tie_to(start_node)

            next_coming_from = []
            
            for empty_matches in ast.empty_matches:
                for empty_match in empty_matches:
                    if empty_match is None:
                        ematch_node = make_id()
                        c.node(ematch_node, "else", color="orange")
                    else:
                        ematch_node = make_id()
                        c.node(ematch_node, label_of(empty_match), color="orange")

                    tie_to(ematch_node, [start_node], extra_kwargs={"label": "\n".join(label_of(x) for x in ast.case_match_actions[empty_matches])})
                    next_coming_from.append(ematch_node)

            for real_matches, sub_ast in ast.sub_matches.items():
                sub_coming_from = []
                for real_match in real_matches:
                    if real_match is None:
                        ematch_node = make_id()
                        c.node(ematch_node, "else")
                    else:
                        ematch_node = make_id()
                        c.node(ematch_node, label_of(real_match))

                    tie_to(ematch_node, [start_node], extra_kwargs={"label": "\n".join(label_of(x) for x in ast.case_match_actions[real_matches])})
                    sub_coming_from.append(ematch_node)

                next_coming_from.extend(debug_dump_ast(sub_ast, into=c, coming_from=sub_coming_from, make_id=make_id, make_subgraph=make_subgraph))

            coming_from = next_coming_from
        
        elif isinstance(ast, OptionalNode):
            # List the adopted actions
            for name, l in zip(("start", "finish"), (ast.start_actions, ast.finish_actions)):
                if not l:
                    continue
                node = make_id()
                c.node(node, f"{name}: {','.join(label_of(x) for x in l)}", style="filled", color="white")

            start_node = make_id()
            c.node(start_node, "", style="filled", shape="circle", color="grey")
            tie_to(start_node)

            coming_from = [start_node, *debug_dump_ast(ast.sub_contents, into=c, coming_from=[start_node], make_id=make_id, make_subgraph=make_subgraph)]
        
        elif isinstance(ast, LoopNode):
            start_node = make_id()
            c.node(start_node, "", style="filled", shape="circle", color="grey")
            tie_to(start_node)

            end_node = make_id()
            c.node(end_node, "end", shape="circle", color="orange")
            coming_from = [end_node]

            if ast.loop_start_actions:
                c.node(make_id(), f"start: " + ",".join(label_of(x) for x in ast.loop_start_actions), style="filled", color="white")

            into.edge(start_node, end_node, label="break\n" + "\n".join(label_of(x) for x in ast.after_break_actions), color="blue")

            tie_to(end_node, cfrm=debug_dump_ast(ast.child_node, into=c, coming_from=[start_node], make_id=make_id, make_subgraph=make_subgraph))

        elif isinstance(ast, TryExceptNode):
            next_coming_from = debug_dump_ast(ast.body, into=c, coming_from=coming_from, make_id=make_id, make_subgraph=make_subgraph)

            handler_start = make_id()
            c.node(handler_start, "catch " + ",".join(x.value for x in ast.handles), color="orange")

            if ast.incoming_body_actions:
                c.node(make_id(), f"start: " + ",".join(label_of(x) for x in ast.incoming_body_actions), style="filled", color="white")

            if ast.incoming_handler_actions:
                c.node(make_id(), f"hstart: " + ",".join(label_of(x) for x in ast.incoming_handler_actions), style="filled", color="white")

            if ast.after_actions:
                c.node(make_id(), f"finish: " + ",".join(label_of(x) for x in ast.after_actions), style="filled", color="white")

            if not ast.handles:
                next_coming_from.append(handler_start)
            elif ast.handler is not None:
                next_coming_from.extend(debug_dump_ast(ast.handler, into=c, coming_from=[handler_start], make_id=make_id, make_subgraph=make_subgraph))

            coming_from = next_coming_from

        else:
            node = make_id()
            c.node(node, "?", color="blue")
            tie_to(node)
            coming_from = [node]
            
    if ast.get_next():
        return debug_dump_ast(ast.get_next(), into=into, coming_from=coming_from, make_id=make_id, make_subgraph=make_subgraph)
    return coming_from

def debug_dump_regextree(rx, indent=0):
    def lprint(*args, **kwargs):
        print(" "*indent, end="")
        print(*args, **kwargs)
    if isinstance(rx, RegexCharClass):
        lprint(repr(rx))
    elif isinstance(rx, RegexSequence):
        lprint("seq")
        for i in rx.sub_matches:
            debug_dump_regextree(i, indent=indent+1)
    elif isinstance(rx, RegexAlternation):
        lprint("alt")
        for i in rx.sub_matches:
            debug_dump_regextree(i, indent=indent+1)
    elif isinstance(rx, RegexKleene):
        lprint("kleene")
        debug_dump_regextree(rx.sub_match, indent=indent+1)
    elif isinstance(rx, RegexOptional):
        lprint("optional")
        debug_dump_regextree(rx.sub_match, indent=indent+1)

def main():
    try:
        input_file, program_name = ProgramData.load_commandline_flags(sys.argv[1:])
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        print("Try nmfu --help for more information", file=sys.stderr)
        exit(1)

    try:
        with open(input_file) as f:
            contents = f.read()
    except IOError as e:
        print("Unable to read input file:", str(e), file=sys.stderr)
        exit(2)

    ProgramData.load_source(contents)
    try:
        parse_tree = parser.parse(contents)
    except lark.LarkError as e:
        print("Syntax error:", str(e), file=sys.stderr)
        exit(3)

    pctx = ParseCtx(parse_tree)

    try:
        pctx.parse()
    except NMFUError as e:
        if ProgramData.dump(DebugDumpable.TRACEBACK):
            raise
        else:
            print("Parse error:", str(e), file=sys.stderr)
            exit(4)

    if ProgramData.dump(DebugDumpable.AST): debug_dump_ast(pctx.ast, ProgramData.dump_prefix + ".ast")

    dctx = DfaCompileCtx(pctx)
    try:
        dctx.compile()
    except NMFUError as e:
        if ProgramData.dump(DebugDumpable.TRACEBACK):
            raise
        else:
            print("Compile error:", str(e), file=sys.stderr)
            exit(5)

    if ProgramData.dump(DebugDumpable.DFA): debug_dump_dfa(dctx.dfa, ProgramData.dump_prefix + ".dfa")
    if ProgramData.dry_run:
        print("... dry run, skipping code generation")
        exit(0)

    cctx = CodegenCtx(dctx, program_name)
    try:
        header = cctx.generate_header()
        source = cctx.generate_source()
    except NMFUError as e:
        if ProgramData.dump(DebugDumpable.TRACEBACK):
            raise
        else:
            print("Codegen error:", str(e), file=sys.stderr)
            exit(5)

    with open(program_name + ".h", "w") as f:
        f.write(header)
    with open(program_name + ".c", "w") as f:
        f.write(source)

if __name__ == "__main__":
    main()
