#!/usr/bin/env python
"""
NMFU - the "no memory for you" "parser" generator.

designed to create what a compsci major would yell at me for calling a dfa to parse files/protocols character by character while using as little
RAM as possible.

Copyright (C) 2020-2023 Matthew Mirvish

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

__version__ = "0.5.2"

import abc
import enum
import string
import itertools
import queue
import io
import textwrap
import os
import sys
import weakref
try:
    import lark
except ImportError: # pragma: no cover
    print("... failed to import lark; must be in setup.py ...", file=sys.stderr)
    # Mock it
    class lark:
        Token = None
        Tree = None
        
        @staticmethod
        def Lark(*args, **kwargs):
            return None
        
        LarkError = RuntimeError
from collections import defaultdict, Counter
from typing import List, Optional, Iterable, Dict, Union, Set, Tuple
try: # pragma: no cover
    import graphviz
    debug_enabled = True
    import lark.tree
except ImportError: # pragma: no cover
    debug_enabled = False

grammar = r"""
start: top_decl* parser_decl top_decl*

?top_decl: out_decl
         | macro_decl
         | hook_decl
         | code_decl

out_decl: "out" out_type IDENTIFIER ";"
        | "out" out_type IDENTIFIER "=" atom ";"

code_decl: RESULT_CODE IDENTIFIER ("," IDENTIFIER)* ";"

out_type: "bool" -> bool_type
        | "int" ("{" int_attr ("," int_attr)* "}")? -> int_type
        | "enum" "{" IDENTIFIER ("," IDENTIFIER)+ "}" -> enum_type
        | "str" "[" RADIX_NUMBER "]" -> str_type
        | "unterminated" "str" "[" RADIX_NUMBER "]" -> unterm_str_type
        | "raw" "{" IDENTIFIER "}" -> raw_type

int_attr: SIGNED -> signed_attr
        | "size" NUMBER -> width_attr

hook_decl: "hook" IDENTIFIER ";"

macro_decl: "macro" IDENTIFIER macro_args "{" statement* "}"

macro_args: "(" macro_arg ("," macro_arg)* ")"
          | "(" ")" -> macro_arg_empty

macro_arg: "macro" IDENTIFIER -> macro_macro_arg
         | "out"   IDENTIFIER -> macro_out_arg
         | "match" IDENTIFIER -> macro_match_expr_arg
         | "expr"  IDENTIFIER -> macro_int_expr_arg
         | "hook"  IDENTIFIER -> macro_hook_arg
         | "loop"  IDENTIFIER -> macro_breaktgt_arg
         | RESULT_CODE IDENTIFIER -> macro_rescode_arg

parser_decl: "parser" "{" statement+ "}"

?statement: block_stmt
          | simple_stmt ";"

simple_stmt: expr -> match_stmt
           | IDENTIFIER "=" expr -> assign_stmt
           | IDENTIFIER "+=" expr -> append_stmt
           | IDENTIFIER "(" (expr ("," expr)*)? ")" -> call_stmt
           | "break" IDENTIFIER? -> break_stmt
           | "delete" IDENTIFIER -> delete_stmt
           | "finish" -> finish_stmt
           | "finish" IDENTIFIER -> custom_finish_stmt
           | "yield" IDENTIFIER -> custom_yield_stmt
           | "wait" expr -> wait_stmt

block_stmt: "loop" IDENTIFIER? "{" statement+ "}" -> loop_stmt
          | "case" "{" case_clause+ "}" -> case_stmt
          | "greedy" "case" "{" greedy_prio_block+ "}" -> greedy_case_stmt
          | "optional" "{" statement+ "}" -> optional_stmt
          | "try" "{" statement+ "}" catch_block -> try_stmt
          | "foreach" "{" statement+ "}" "do" "{" foreach_actions "}" -> foreach_stmt
          | "if" if_condition ("elif" if_condition)* else_condition? -> if_stmt

if_condition: _math_expr "{" statement+ "}"
else_condition: "else" "{" statement+ "}"

foreach_actions: statement+

catch_block: "catch" catch_options? "{" statement* "}"

?greedy_prio_block: "prio" NUMBER case_clause
                  | "prio" NUMBER "{" case_clause+ "}"
                  | case_clause

case_clause: case_predicate ("," case_predicate)* "->" "{" statement* "}"
case_predicate: "else" -> else_predicate
              | expr -> expr_predicate

catch_options: "(" CATCH_OPTION ("," CATCH_OPTION)* ")" 

// EXPRESSIONS

?expr: atom // either literal or string depending on context
     | regex // not an atom to simplify things
     | binary_regex
     | "end" -> end_expr
     | "(" expr+ ")" -> concat_expr
     | "[" _math_expr "]"

atom: BOOL_CONST -> bool_const
    | RADIX_NUMBER -> number_const
    | CHAR_CONSTANT -> char_const
    | STRING "i" -> string_case_const
    | STRING "b" -> binary_string_const
    | STRING -> string_const
    | IDENTIFIER -> identifier_const

_math_expr: disjunction_expr

?disjunction_expr: conjunction_expr ("||" conjunction_expr)*
?conjunction_expr: bit_or_expr ("&&" bit_or_expr)*
?bit_or_expr: bit_xor_expr ("|" bit_xor_expr)*
?bit_xor_expr: bit_and_expr ("^" bit_and_expr)*
?bit_and_expr: comp_expr ("&" comp_expr)*
?comp_expr: shift_expr (CMP_OP shift_expr)?
// nmfu does not allow (1 << 2 << 3) because that is dumb
?shift_expr: sum_expr (SHIFT_OP sum_expr)?
?sum_expr: mul_expr (SUM_OP mul_expr)*
?mul_expr: math_unary (MUL_OP math_unary)*
?math_unary: math_atom
           | "!" math_atom -> not_expr
           | "-" math_atom -> negate_expr
?math_atom: RADIX_NUMBER -> math_num
          | IDENTIFIER -> math_var
          | IDENTIFIER ".len" -> math_str_len
          | IDENTIFIER "[" _math_expr "]" -> math_str_index
          | CHAR_CONSTANT -> math_char_const
          | BOOL_CONST -> bool_const
          | "$" IDENTIFIER -> builtin_math_var
          | "(" _math_expr ")"

// REGEX

// Uses templates to avoid duplication

regex: "/" regex_alternation "/"

regex_char_class: "\\" REGEX_CHARCLASS
?regex_alternation: regex_group ("|" regex_group)*
?regex_group: regex_alternation_element+
?regex_alternation_element: regex_literal
                          | regex_literal REGEX_OP -> regex_operation
                          | regex_literal "{" NUMBER "}" -> regex_exact_repeat
                          | regex_literal "{" NUMBER "," NUMBER "}" -> regex_range_repeat
                          | regex_literal "{" NUMBER "," "}" -> regex_at_least_repeat

?regex_literal: REGEX_UNIMPORTANT -> regex_raw_match // creates multiple char classes
              | regex_char_class // creates a char class
              | "(" regex_alternation ")" -> regex_group
              | "[^" regex_set_element+ "]" -> regex_inverted_set // creates an inverted char class
              | "[" regex_set_element+ "]" -> regex_set // creates a char class
              | "." -> regex_any // creates an inverted char class

?regex_set_element: REGEX_CHARGROUP_ELEMENT_RAW
                  | REGEX_CHARGROUP_ELEMENT_RAW "-" REGEX_CHARGROUP_ELEMENT_RAW -> regex_set_range
                  | regex_char_class

binary_regex: "b/" binary_regex_alternation "/"
?binary_regex_alternation: binary_regex_group ("|" binary_regex_group)*

?binary_regex_group: binary_regex_alternation_element+
?binary_regex_alternation_element: binary_regex_literal
                          | binary_regex_literal REGEX_OP -> binary_regex_operation
                          | binary_regex_literal "{" NUMBER "}" -> binary_regex_exact_repeat
                          | binary_regex_literal "{" NUMBER "," NUMBER "}" -> binary_regex_range_repeat
                          | binary_regex_literal "{" NUMBER "," "}" -> binary_regex_at_least_repeat

?binary_regex_literal: REGEX_BYTE -> binary_regex_raw_match // creates multiple char classes
                     | "(" binary_regex_alternation ")" -> binary_regex_group
                     | "[^" binary_regex_set_element+ "]" -> binary_regex_inverted_set // creates an inverted char class
                     | "[" binary_regex_set_element+ "]" -> binary_regex_set // creates a char class
                     | "." -> binary_regex_any // creates an inverted char class

?binary_regex_set_element: REGEX_BYTE
                         | REGEX_BYTE "-" REGEX_BYTE -> binary_regex_set_range

// BINARY REGEX

// TERMINALS
BOOL_CONST: /true|false/

// catch options
CATCH_OPTION: /nomatch|outofspace/

RESULT_CODE: /yieldcode|finishcode/

%import common.CNAME -> IDENTIFIER
%import common.SIGNED_INT -> NUMBER
%import common.HEXDIGIT

HEX_NUMBER: ["+"|"-"] "0x" HEXDIGIT+
BIN_NUMBER: "0b" ["0"|"1"]+
RADIX_NUMBER: HEX_NUMBER | BIN_NUMBER | NUMBER

STRING: /"(?:[^"\\]|\\.)*"/

// regex internals
REGEX_UNIMPORTANT: /[^.?*()\[\]\\+{}|\/]|\\\.|\\\*|\\\(|\\\)|\\\[|\\\]|\\\+|\\\\|\\\{|\\\}|\\\||\\\//
REGEX_OP: /[+*?]/
REGEX_CHARGROUP_ELEMENT_RAW: /[^\-\]\\\/]|\\-|\\\]|\\\\|\\\//
REGEX_CHARCLASS: /[wWdDsSntr ]/
REGEX_BYTE: /[0-9a-fA-F]{2}/

// math
SUM_OP: /[+-]/
MUL_OP: /[*\/%]/
CMP_OP: /[!=]=/ | /[<>]=?/
CHAR_CONSTANT: /'[^'\\]'/ | /'\\.'/
SHIFT_OP: "<<" | ">>" 

SIGNED: "signed" | "unsigned"

// comments

%import common.WS
COMMENT: /\s*\/\/[^\n]*/

%ignore WS
%ignore COMMENT
"""

all_sum_expr_nodes = [
    "disjunction_expr",
    "conjunction_expr",
    "comp_expr",
    "sum_expr",
    "mul_expr",
    "math_num",
    "negate_expr",
    "not_expr",
    "shift_expr",
    "bit_or_expr",
    "bit_xor_expr",
    "bit_and_expr",
    "math_str_len",
    "math_str_index",
    "math_var",
    "math_char_const",
    "builtin_math_var"
]

parser = lark.Lark(grammar, propagate_positions=True, lexer="dynamic_complete", start=["start", "regex"])

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

    def __init__(self, on_values=None, fallthrough=False, error_handling=False):
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
        self.error_handling = error_handling
        self.actions = []

    def copy(self):
        my_copy = DFTransition()
        my_copy.on_values = self.on_values.copy()
        my_copy.actions = self.actions.copy()
        my_copy.target = self.target
        my_copy.is_fallthrough = self.is_fallthrough
        my_copy.error_handling = self.error_handling
        return my_copy

    def __repr__(self):
        error = "error " if self.error_handling else ""
        if self.actions:
            return f"<DFTransition {error}on={self.on_values} to={self.target} actions={self.actions} fallthough={self.is_fallthrough}>"
        else:
            return f"<DFTransition {error}on={self.on_values} to={self.target} fallthough={self.is_fallthrough}>"

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

    def handles_else(self, handles=True):
        self.error_handling = handles
        return self

    @classmethod
    def from_key(cls, on_values, inherited):
        return cls(on_values).to(inherited.target).attach(*inherited.actions).fallthrough(inherited.is_fallthrough).handles_else(inherited.error_handling)

class DFConditionalTransition(DFTransition):
    def __init__(self, condition: "DFCondition"):
        super().__init__(on_values=(), fallthrough=True)
        self.condition = condition

    def constrain(self, *conditions, join_as_or=False):
        # TODO: work properly
        self.condition = conditions[0]

    def __repr__(self):
        if self.actions:
            return f"<DFConditionalTransition on={self.condition!r} to={self.target} actions={self.actions}>"
        else:
            return f"<DFConditionalTransition on={self.condition!r} to={self.target}>"

class DFState:
    all_states = {}

    def __init__(self):
        self.transitions = []
        DFState.all_states[id(self)] = self

    def transition(self, transition, allow_replace=False, allow_replace_if=None, collapse_else=True):
        if isinstance(transition, DFConditionalTransition):
            raise IllegalDFAStateError("Invalid condition transition on normal state", self)
        # Add parent relationship if this is a new transition
        if ProgramData.lookup(transition, DTAG.PARENT) is None:
            ProgramData.imbue(transition, DTAG.PARENT, self)
        if allow_replace_if is None:
            allow_replace_constant = True if allow_replace else False  # don't allow reference binding
            allow_replace = lambda x: allow_replace_constant
        else:
            allow_replace = allow_replace_if

        contained = set()
        for i in self.all_transitions():
            if set(i.on_values) & set(transition.on_values):
                contained.add(i)
                break
        if contained and any(x.target != transition.target or x.is_fallthrough != transition.is_fallthrough or x.error_handling != transition.error_handling for x in contained):
            for contain in contained:
                if allow_replace(contain):
                    for x in transition.on_values:
                        try:
                            contain.on_values.remove(x)
                        except ValueError:
                            continue
                    if not contain.on_values:
                        self.transitions.remove(contain)
                else:
                    raise IllegalDFAStateError("Duplicate transition", self)
        elif contained:
            return # don't add duplicate entries in the table

        # Check if we can "inline" this transition
        if collapse_else:
            for transition_in in self.transitions:
                if transition_in.actions == transition.actions and transition_in.target == transition.target and transition_in.is_fallthrough == transition.is_fallthrough and transition_in.error_handling == transition.error_handling:
                    if DFTransition.Else in transition_in.on_values or DFTransition.Else in transition.on_values:
                        transition_in.on_values = [DFTransition.Else]
                    else:
                        transition_in.on_values = list(set(transition_in.on_values) | set(transition.on_values))
                    return self
        self.transitions.append(transition)
        return self

    def all_transitions(self) -> Iterable[DFTransition]:
        yield from self.transitions

    def all_transitions_for(self, on_values):
        for i in self.transitions:
            if len(set(i.on_values) & set(on_values)):
                yield i

    def compute_foreign_else_definition(self, other_state: "DFState") -> Set:
        r"""
        Given two states

           (self) -(a,b,c)-> ...
             \-----(ELSE)-> ...       

        and

           (other) ---(a)-> ...
             \-----(ELSE)-> ...

        it should be clear that the specific meaning of ELSE differs on both.

        This function returns a (potentially super-)set of the symbols which ELSE
        on a _foreign_ state (like "other") will match on this state.
        """

        our_alphabet = self.local_alphabet()
        their_alphabet = other_state.local_alphabet()

        local_else_additions = our_alphabet - their_alphabet
        local_else_additions.add(DFTransition.Else)
        return local_else_additions

    def __setitem__(self, key, data):
        if type(key) in (tuple, list, set, frozenset):
            on_values = key
        else:
            on_values = [key]

        if type(data) is DFTransition:
            self.transition(DFTransition.from_key(on_values, data), True)
        else:
            self.transition(DFTransition(on_values).to(data), True)

    def __delitem__(self, key):
        if type(key) in (tuple, list, set, frozenset):
            on_values = key
        else:
            on_values = [key]
        contained = None
        for i in self.all_transitions():
            if set(i.on_values) & set(on_values):
                contained = i
                break
        if contained is not None:
            for on_value in on_values:
                contained.on_values.remove(on_value)
            if not contained.on_values:
                self.transitions.remove(contained)

    def __getitem__(self, data):
        """
        Get the transition that would be followed for data, including Else if applicable
        """
        if type(data) in (list, tuple, set, frozenset):
            data = frozenset(data)
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

    def local_alphabet(self, excluding=(DFTransition.Else,)) -> Set:
        """
        Get the 'local alphabet': all symbols that are matched by this state excluding some set.
        """

        local_alphabet = set()
        for trans in self.transitions:
            for i in trans.on_values:
                if i in excluding: continue
                local_alphabet.add(i)
        return local_alphabet

class DFProxyState(DFState):
    """
    Represents a state that only contains the equivalent of a fallthrough else
    tansition to another state; in other words a pure "proxy" to another state, potentially
    with some number of actions/conditions. 

    Note this class on its own does not _enforce_ these constraints, and will break if used in a way that
    violates them.
    """

    def can_eliminate(self):
        """
        Can this node be eliminated?
        """

        for transition in self.all_transitions():
            if transition.actions:
                return False

        return True

    def equivalent_on_values(self):
        r"""
        Compute an "equivalent" set of on_values, such that taking any one will have at least one valid path through all directly-attached condition points, in addition
        to a set of on_values which map to the treat_as_else state in _all_ outcomes.

        i.e. for  a -t-> b -<c>-> c
                   \-f-> d -<else>-> e
                          \-<f>-> ELSE

        we'd return ({c, else}, {f}), since f should go to an error state in some equivalent state E replacing a, and c/else should continue on. This lets us rewrite the dfa as

        E -{c,else}=> a -> ...
         \-f=> ELSE 

        which is a valid construction for append_after.
        """

        encountered = set()
        candidate_else = set()

        visited = set()
        sources = set()

        def aux(state):
            if state in visited:
                return
            visited.add(state)
            if isinstance(state, DFProxyState):
                for t in state.all_transitions():
                    aux(t.target)
                    # Also consider actions with overrides
                    for action in t.actions:
                        if action.get_target_override_mode() in (ActionOverrideMode.MAY_GOTO_TARGET, ActionOverrideMode.ALWAYS_GOTO_OTHER):
                            for extra in action.get_target_override_targets():
                                aux(extra)
            else:
                sources.add(state)
                for t in state.all_transitions():
                    if t.error_handling:
                        candidate_else.update(t.on_values)
                    else:
                        encountered.update(t.on_values)

        aux(self)

        filtered = set()
        for possible in candidate_else:
            for state in sources:
                if state[possible] is None:
                    continue
                if not state[possible].error_handling:
                    filtered.add(possible)

        encountered.update(filtered)
        candidate_else.difference_update(filtered)

        return encountered, candidate_else


class DFConditionPoint(DFProxyState):
    """
    Represents a single "condition point": a node which only has fallthrough conditional-transitions. A conditional-transition
    doesn't match input, rather it just selects some unambiguous path to continue on
    """

    def __init__(self):
        super().__init__()
        self.transitions = []

    def transition(self, transition):
        if not isinstance(transition, DFConditionalTransition):
            raise IllegalDFAStateError("Invalid normal transition on condition point", self)
        if transition in self.transitions:
            return
        if ProgramData.lookup(transition, DTAG.PARENT) is None: 
            ProgramData.imbue(transition, DTAG.PARENT, self)
        self.transitions.append(transition)

    def __getitem__(self, key): # pragma: no cover
        raise NotImplementedError()
    def __delitem__(self, key): # pragma: no cover
        raise NotImplementedError()
    def __setitem__(self, key): # pragma: no cover
        raise NotImplementedError()

    def can_eliminate(self):
        return False
        

class DFA:
    all_state_machines = {}

    def __init__(self):
        self.accepting_states = []
        self.starting_state = None
        self.states: List[DFState] = []

    def add(self, state):
        if ProgramData.lookup(state, DTAG.PARENT) is None:
            ProgramData.imbue(state, DTAG.PARENT, self)
        if self.starting_state == None:
            self.starting_state = state
        self.states.append(state)

    def mark_accepting(self, state):
        if isinstance(state, int):
            self.accepting_states.append(DFState.all_states[state])
        else:
            self.accepting_states.append(state)

    def simulate(self, actions):
        """
        Attempt to simulate what would happen given the input states.
        """

        return list(st for _, st, act in self.trace(actions) if act is None)[-1]

    def trace(self, actions):
        """
        Record a trace of all actions and state transitions executed given some input, in the form
        (char, state, action)
        """

        position = self.starting_state
        yield (None, position, None)

        for action_char in actions:
            try:
                while True:
                    if position is None:
                        break
                    transition = position[action_char]
                    position = transition.target
                    yield (action_char, position, None)
                    for action in transition.actions:
                        yield (action_char, position, action)
                        if action.get_target_override_mode() == ActionOverrideMode.ALWAYS_GOTO_OTHER:
                            potential = action.get_target_override_targets() # TODO: handle correctly
                            position = potential[0]
                        elif action.get_target_override_mode() == ActionOverrideMode.ALWAYS_GOTO_UNDEFINED:
                            yield (action_char, action, None)
                            return
                    if not transition.is_fallthrough:
                        break
            except AttributeError:
                yield (action_char, None, None)
                return
            else:
                if not position:
                    yield (action_char, None, None)
                    return

    def simulate_accepts(self, actions):
        """
        Determine if a given input ends with a success or fail (considers FinishAction)
        """

        result = self.simulate(actions)
        if result in self.accepting_states:
            return True
        if isinstance(result, FinishAction):
            return True
        return False

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
            yield state

            for t in state.all_transitions():
                use_real = True
                for action in t.actions:
                    if action.get_target_override_mode() == ActionOverrideMode.ALWAYS_GOTO_OTHER:
                        use_real = False
                        for tgt in action.get_target_override_targets():
                            yield from aux(tgt)
                        break
                    elif action.get_target_override_mode() == ActionOverrideMode.ALWAYS_GOTO_UNDEFINED:
                        use_real = False
                        break
                    elif action.get_target_override_mode() == ActionOverrideMode.MAY_GOTO_TARGET:
                        for tgt in action.get_target_override_targets():
                            yield from aux(tgt)
                if use_real:
                    yield from aux(t.target)

        yield from aux(self.starting_state)

    def error_handling_transitions(self, include_states=False):
        """
        Return a set of all transitions that are marked as error handling from the start state
        """

        result = set()

        for state, t in self.all_transitions(True):
            if t.error_handling:
                if include_states:
                    result.add((state, t))
                else:
                    result.add(t)

        return result


    def transitions_pointing_to(self, target_state: DFState, include_states=False):
        """
        Return a set of all transitions that point to the target state that are reachable
        from the start state
        """

        result = set()

        for state, t in self.all_transitions(True):
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

        for t in self.all_transitions():
            if action in t.actions:
                result.add(t)

        return result

    def all_transitions(self, include_states=False):
        """
        Yield all reachable transitions
        """

        for state in self.dfs():
            for action in state.all_transitions():
                if include_states:
                    yield (state, action)
                else:
                    yield action

    def is_valid(self):
        """
        Can we still reach at least one accept state?
        """

        for state in self.dfs():
            if state in self.accepting_states:
                return True

        return False

    def append_after(self, chained_dfa: "DFA", sub_states=None, mark_accept=True, chain_actions=None):
        """
        Add the chained_dfa in such a way that its start state "becomes" all of the `sub_states` (or if unspecified, the accept states of _this_ dfa)

        If mark_accept is True, we should replace the accept states that we currently have with corresponding ones based on the chained dfa.
        If chain_actions is not empty, we add those actions to all new transitions. This does _not_ create potential ambiguity, as the original start
        states of chained DFAs are kept in-tact, so loops work properly. In other wors, chain_actions will only be run once and so is a suitable mechanism
        for adding finish actions.
        """

        if sub_states is None:
            sub_states = self.accepting_states

        if chain_actions is None:
            chain_actions = []

        if isinstance(chained_dfa.starting_state, DFProxyState):
            # Add an extra state that will represent the condition point properly (see docs for equivalent_on_values)
            valid, to_else = chained_dfa.starting_state.equivalent_on_values()
            if valid and to_else:
                fake_start = DFState()
                chained_dfa.add(fake_start)

                fake_start[valid] = chained_dfa.starting_state
                fake_start[valid].fallthrough(True)
                fake_start[to_else] = chained_dfa.starting_state
                fake_start[to_else].fallthrough(True).handles_else()
                chained_dfa.starting_state = fake_start

        # Check for ambiguity: if any transitions added to a sub_state try to redirect a valid character a different valid
        # state, there is ambiguity. In other words, we only want to add transitions that would previously lead to the error state.

        chained_transitions = [x for x in chained_dfa.starting_state.transitions if DFTransition.Else in x.on_values]
        chained_transitions.extend(x for x in chained_dfa.starting_state.transitions if DFTransition.Else not in x.on_values)

        valid_replacers = set()

        for transition in chained_transitions:
            if transition.error_handling:
                continue
            valid_replacers.add(transition.target)

        for sub_state in sub_states:
            # Compute the "local else additions": what things that are matched by Else in the chained start state which Else in the sub_state does _not_ match.
            # Note that we only handle this in one direction; technically sub_state[Else] could refer to a smaller set than chained.start[Else] but that shouldn't
            # matter too much.

            sub_local_alphabet = sub_state.local_alphabet()
            local_else_meaning = sub_state.compute_foreign_else_definition(chained_dfa.starting_state)

            culled_chained_transitions = []

            for transition in chained_transitions:
                culled_transition = transition.copy()

                relevant_values = set(transition.on_values)
                if DFTransition.Else in relevant_values:
                    relevant_values.update(local_else_meaning)
                target = sub_state[relevant_values]
                if target is None:
                    targets = set()
                    target_map = {}
                    for value in relevant_values:
                        v = sub_state[value]
                        if v is None:
                            continue
                        targets.add(v)
                        if v not in target_map: target_map[v] = set()
                        target_map[v].add(value)
                else:
                    targets = {target} 
                    target_map = {target: relevant_values}

                if transition.error_handling:
                    # Cull values
                    for j in relevant_values.copy():
                        if sub_state[j] is not None and not sub_state[j].error_handling:
                            relevant_values.remove(j)
                elif not all(x.error_handling or x.target == transition.target for x in targets):
                    if ProgramData.dump(DebugDumpable.DFA) and ProgramData.do(ProgramFlag.VERBOSE_AMBIG_ERRORS): # pragma: no cover
                        debug_dump_dfa(self, "ae_dfa2", highlight=sub_state)
                        debug_dump_dfa(chained_dfa, "ae_dfa1")
                    dprint[ProgramFlag.VERBOSE_AMBIG_ERRORS]("TT", transition)
                    dprint[ProgramFlag.VERBOSE_AMBIG_ERRORS]("TA", targets)
                    dprint[ProgramFlag.VERBOSE_AMBIG_ERRORS]("RV", relevant_values)
                    raise IllegalDFAStateConflictsError("Ambiguous transitions detected while joining DFAs", sub_state, transition)

                # Create transition to add
                culled_transition.on_values = list(relevant_values)
                culled_transition.attach(*chain_actions, prepend=True)
                culled_chained_transitions.append(culled_transition)

            for new_transition in culled_chained_transitions:
                sub_state.transition(new_transition, allow_replace_if=lambda x: x.error_handling or x.target in valid_replacers)

        # Adopt all the states
        for state in chained_dfa.states:
            if state not in self.states:
                self.add(state)

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

    def chain_actions_at_end(self, actions: Iterable["Action"]):
        """
        Attempt to chain the given actions on all transitions pointing into accept states. If these actions cannot be scheduled once (if requested) an error will be thrown
        """

        for finish in self.accepting_states:
            for incoming, trans in self.transitions_pointing_to(finish, include_states=True):
                for action in actions:
                    if action.is_timing_strict() and any(not x.error_handling for x in finish.transitions):
                        # Complain early
                        raise UnableToScheduleActionError([incoming], [action])
                    else:
                        trans.attach(action)

# =============
# DEBUG STORAGE
# =============

class DTAG(enum.Enum):
    NAME = 0
    SOURCE_LINE = 1
    SOURCE_COLUMN = 2
    PARENT = 3

    # Specific to actions
    STRICT_TIMING_REASON = 20

    # Used for tracking action skips
    ACTION_MAY_SKIP = 40
    ACTION_MAY_RETURN = 41

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
    VERBOSE_REGEX_CCLASS = 200
    VERBOSE_OPTIMIZE_RESULTS = 201
    VERBOSE_SIMPLIFY_TM = 202
    VERBOSE_AMBIG_ERRORS = 203

    # Optimization flags, set in ProgramData
    # DFA optimization options (enabled by various levels of -O)
    SIMPLIFY_ELSE_CONDITIONS = 300
    REMOVE_INACCESIBLE_STATES = 301
    USE_DELETE_FOR_EMPTY_STRING = 302
    SHORTCIRCUIT_FALLTHROUGHS = 303

    # Codegen optimization options ('')
    COLLAPSE_TRANSITION_RANGES = (9, "Rewrite large transition values as codepoint range checks")

    # Codegen options 
    # |
    # - Structure options
    DYNAMIC_MEMORY = 7  # Allow use of dynamic memory
    INCLUDE_USER_PTR = (11, "Add a void* to the state structure, useful with hooks")
    STRICT_DONE_TOKEN_GENERATION = (15, "Only return DONE from _feed at accept states, not from transitions to accept states")
    EOF_SUPPORT = (16, "Generate a end function and support detecting eofs")
    INDIRECT_START_PTR = (17, "Use an indirect pointer for start to report where the input ends precisely")
    YIELD_SUPPORT = (30, "Support yield expressions; implies indirect start pointers", False, (17,))
    ZERO_LEN_INPUT_SUPPORT = (31, "Allow calling feed with start == end", False)
    # |
    # - String storage options
    ALLOCATE_STR_SPACE_IN_STRUCT = (5, "Allocate string space in struct", True, (), (6,))  # allocate the string space in the struct as an array, default
    ALLOCATE_STR_SPACE_DYNAMIC   = (6, "Allocate string space dynamically", False, (7,), (5,))  # implies DYNAMIC
    ALLOCATE_STR_SPACE_DYNAMIC_ON_DEMAND = (8, "Allocate string space on demand", False, (6,))  # implies DYNAMIC
    UNSAFE_STRING_INDEXING = (18, "Don't perform runtime bounds checks on string indexing")
    STRINGS_AS_U8 = (19, "Store strings as uint8_t[] arrays instead of char arrays")
    DELETE_STRING_FREE_MEMORY = (21, "delete-ing a string should also free() it", False, (), (5,))
    # |
    # - Hook options
    HOOK_GLOBAL = (12, "Place hooks as global functions", True)
    HOOK_PER_STATE = (13, "Place hooks as function pointers in the state structure", False, (), (12,))
    # |
    # - Template options
    USE_PRAGMA_ONCE = (10, "Use #pragma once instead of an #ifndef guard in the header")
    USE_CPLUSPLUS_GUARD = (14, "Include a __cplusplus extern C guard", True)

    # Debug options
    DEBUG_DFA_HIDE_ERROR_HANDLING = (100, "", True)
    DEBUG_DFA_BINARY_LABELS = (101, "Show all transition labels as hex points", False)
    DEBUG_STRICT_PROGRAM_DATA_ERRORS = (102, "Throw an error if the ProgramData tree is updated in an invalid way", False)
    DEBUG_DTREE_HIDE_GC = (103, "Don't show entries in dtree if they've been garbage collected", True)

class ProgramOption(enum.Enum):
    def __init__(self, default, helpstr):
        self.default = default
        self.helpstr = helpstr

    # DFA optimization options
    MAX_SHORTCIRCUIT_FALLTHROUGH = (20, "Maximum number of equivalent fallthrough transitions to inline")
    MAX_SHORTCIRCUIT_ACTION_PENALTY = (3, "How much actions affect the max shortcircuit falloff amount")

    # Codegen options
    COLLAPSED_RANGE_LENGTH = (4, "Minimum length of range to collapse into range comparison")

    # Debug options
    DEBUG_DFA_HIDE_THRESHOLD = (15, "")
    DEBUG_GRAPH_DUMP_FORMAT = ("pdf", "Output format for graphviz dumpers, use 'dot' to get raw dot file")

class HasDefaultDebugInfo:
    def debug_lookup(self, tag: DTAG): # pragma: no cover
        return None

class DebugDumpable(enum.Enum):
    AST = "ast"
    DFA = "dfa"
    TRACEBACK = "traceback"
    PARSE = "parse"
    DTREE = "dtree"

class ProgramData:
    _collection = defaultdict(dict)
    _children = defaultdict(list)
    _refmap = {}
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
        2: (ProgramFlag.COLLAPSE_TRANSITION_RANGES, ProgramFlag.USE_DELETE_FOR_EMPTY_STRING),     #- O2 (adds to 1, 2)
        3: (ProgramFlag.SHORTCIRCUIT_FALLTHROUGHS,),
    }

    @classmethod
    def load_source(cls, src: str):
        """
        Load the currently processing source code
        """

        cls._current_source = src.splitlines(keepends=False)

    @classmethod
    def _ensure_refmapped(cls, obj: object):
        if id(obj) not in cls._refmap:
            cls._refmap[id(obj)] = weakref.ref(obj)
        else:
            if cls._refmap[id(obj)]() is not obj:
                cls._refmap[id(obj)] = weakref.ref(obj)
                cls._children[id(obj)] = []
                cls._collection[id(obj)] = {}

    @classmethod
    def imbue(cls, obj: object, tag: DTAG, value: object, *extra_tags):
        """
        Imbue this object with this debug information
        """

        cls._ensure_refmapped(obj)

        bad = False

        if tag == DTAG.PARENT:
            cls._ensure_refmapped(value)
            # Ensure this node is not trying to mark itself as a parent
            if obj is value:
                if cls.do(ProgramFlag.DEBUG_STRICT_PROGRAM_DATA_ERRORS):
                    raise NMFUError([obj], "Attempt to set object as its own parent")
                else:
                    bad = True
            else:
                par = cls.lookup(value, DTAG.PARENT)
                while par is not None:
                    if par is obj:
                        if cls.do(ProgramFlag.DEBUG_STRICT_PROGRAM_DATA_ERRORS):
                            raise NMFUError([obj, par, value], "Attempt to set object as its own parent")
                        else:
                            bad = True
                        break
                    par = cls.lookup(par, DTAG.PARENT)

            if not bad:
                cls._children[id(value.__repr__.__self__)].append(id(obj))

        if not bad:
            cls._collection[id(obj)][tag] = value
        if extra_tags:
            return cls.imbue(obj, *extra_tags)
        return obj

    @classmethod
    def lookup(cls, obj: object, tag: DTAG, recurse_upwards=True, default=None, recurse_downwards=True):
        """
        Find the imbued object's data, or -- if none exists -- find it's parents
        """

        if type(obj) is lark.Token:
            if tag == DTAG.SOURCE_LINE:
                return obj.line
            elif tag == DTAG.SOURCE_COLUMN:
                return obj.column
            else:
                return default

        if type(obj) is lark.Tree:
            if tag == DTAG.SOURCE_LINE:
                return obj.meta.line
            elif tag == DTAG.SOURCE_COLUMN:
                return obj.meta.column
            else:
                return default

        id_obj = id(obj) if type(obj) is not int else obj

        # Ensure consistency
        if type(obj) is int and obj in cls._refmap and cls._refmap[obj]() is None:
            del cls._refmap[id_obj]
            cls._collection[id_obj] = {}

        if tag not in ProgramData._collection[id_obj]:
            if DTAG.PARENT in ProgramData._collection[id_obj] and recurse_upwards:
                val = cls.lookup(ProgramData._collection[id_obj][DTAG.PARENT], tag)
                if val is not None:
                    return val
            if isinstance(obj, HasDefaultDebugInfo):
                val = obj.debug_lookup(tag)
                if val is not None:
                    return val
            elif isinstance(obj, int) and id_obj in cls._refmap:
                v_obj = cls._refmap[id_obj]()
                if isinstance(v_obj, HasDefaultDebugInfo):
                    val = v_obj.debug_lookup(tag)
                    if val is not None:
                        return val
            if tag in (DTAG.SOURCE_COLUMN, DTAG.SOURCE_LINE) and recurse_downwards:
                for i in cls._children[id_obj]:
                    val = cls.lookup(i, tag, recurse_upwards=False)
                    if val is not None:
                        return val
            return default
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
        print("Copyright (C) 2020-2023 Matthew Mirvish")
        print("This is free software; see the source for copying conditions.  There is NO")
        print("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.")

    @classmethod
    def _print_help(cls, show_all=False):
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
        print("  --help-all                                 Show this help screen; showing hidden debug options")
        print("  --version                                  Show the version of nmfu")
        print("")
        print("Generation Options:")
        
        def filter_options_for_all(options):
            for i in options:
                if show_all:
                    yield i
                else:
                    if i.name.startswith("VERBOSE") or i.name.startswith("DEBUG"):
                        continue
                    yield i

        pad_length = 4 + len(max(filter_options_for_all(ProgramOption), key=lambda x: len(x.name)).name) + 6
        for option in filter_options_for_all(ProgramOption):
            flag_name = option.name.replace("_", "-").lower()
            opt_str = f"  --{flag_name} <arg>"
            print(f"{opt_str: <{pad_length}} {option.helpstr} (default: {option.default})")
        print("")
        print("Flags:")
        pad_length = 2 + len(max((y for y in filter_options_for_all(ProgramFlag) if cls._is_optimization_flag(y) == -1), key=lambda x: len(x.name)).name)
        for flag in filter_options_for_all(ProgramFlag):
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
    def _reset_flags(cls):
        cls._flags = {
                x: x.default for x in ProgramFlag
        }
        cls._options = {
                x: x.default for x in ProgramOption
        }
        cls._dump = []
        cls.dry_run = False
        cls.dump_prefix = None
        del cls._collection
        del cls._children
        del cls._refmap
        cls._collection = defaultdict(dict)
        cls._children = defaultdict(list)
        cls._refmap = {}

    @classmethod
    def load_commandline_flags(cls, all_cmd_options: List[str]):
        """
        Load the command line flags passed in. Returns a tuple of (input_filename, program_output_name)
        """

        cls._reset_flags()

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
                    program_output_name = "".join(x if (
                        x in string.ascii_letters or x == '_' or (i > 0 and x in string.digits)
                    ) else '_' for i, x in enumerate(program_output_name))
                    continue
                elif option[1] == "-":
                    option_name = option[2:]
                    if option_name not in ["help", "dry-run", "version", "help-all"]:
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
            elif option_name == "help-all":
                cls._print_help(show_all=True)
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
        if ProgramData.do(self.condition): # pragma: no cover
            print(*args, **kwargs)


# ===========
# ERROR TYPES
# ===========

class NMFUError(Exception):
    def __init__(self, reasons, message=None):
        self.reasons = reasons
        self.message = message

    @classmethod
    def _generate_whitespace_marker(cls, line, column):
        marker = ""
        for i in range(column):
            if i == column - 1:
                marker += "^"
            elif ProgramData.get_source_line(line)[i] == "\t":
                marker += "\t"
            else:
                marker += " "
        return marker

    def _get_message(self, show_potential_reasons=True, reasons_header="Potential reasons include:", subset=None, additional_info=None):
        if subset is None:
            subset = self.reasons
        if additional_info is None:
            additional_info = lambda reason: None 
        info_strs = []
        for reason in subset:
            name, line, column = (ProgramData.lookup(reason, tag) for tag in (DTAG.NAME, DTAG.SOURCE_LINE, DTAG.SOURCE_COLUMN))
            info_str = ""
            if name:
                info_str += f"- {name}:"
                if line is not None:
                    info_str += f"\n  at line {line}:\n{ProgramData.get_source_line(line)}"
                    if column is not None:
                        info_str += "\n" + NMFUError._generate_whitespace_marker(line, column)
                else:
                    info_str = info_str[-1]
            else:
                if line is not None:
                    info_str += f"- line {line}:\n{ProgramData.get_source_line(line)}"
                    if column is not None:
                        info_str += "\n" + NMFUError._generate_whitespace_marker(line, column)
                else:
                    continue
            if additional_info(reason):
                info_str += "\n  {}".format(additional_info(reason))
            info_strs.append(info_str)

        if info_strs:
            return (f"{reasons_header}\n" if show_potential_reasons else "") + "\n".join(info_strs)
        else:
            return ""

    def __str__(self):
        if self.message:
            return self.message + " " + self._get_message()
        return self._get_message()

class IllegalASTStateError(NMFUError):
    def __init__(self, msg, *source):
        super().__init__([*source])
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

class UnableToScheduleActionError(NMFUError):
    def __init__(self, states, actions):
        self.states = list(states)
        self.actions = list(actions)
        super().__init__([*self.states, *self.actions])
    
    def __str__(self):
        def add_info(act):
            rsn = ProgramData.lookup(act, DTAG.STRICT_TIMING_REASON)
            if rsn:
                return f"(strict timing because {rsn})"
            return None
        return f"Unable to schedule finish actions only once on\n{self._get_message(show_potential_reasons=False, subset=self.states)}\nfor\n{self._get_message(show_potential_reasons=False, subset=self.actions, additional_info=add_info)}"

class UndefinedReferenceError(NMFUError):
    def __init__(self, objtype, source: lark.Token):
        super().__init__([source])
        self.objtype = objtype
        self.source = source

    def __str__(self):
        if self.objtype is None:
            return f"Undefined reference to {self.source.value}:\n" + self._get_message(show_potential_reasons=False)
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
    MAY_GOTO_TARGET = 4 # action may goto a different state
    MAY_GOTO_UNDEFINED = 3 # action may goto an undefined state (like the finish action)
    ALWAYS_GOTO_OTHER = 2 # action will always go to a defined state
    ALWAYS_GOTO_UNDEFINED = 1 # action will always go to an undefined state (usually something like the global "PREMATURE_EXIT" state or whatever)

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

    def get_target_override_targets(self) -> List[DFState]:
        return []

    def modifies(self) -> List["OutputStorage"]:
        """
        Return which outputs are modified by executing this action
        """

        return []

    def embeddable(self):
        """
        Returns false if this action should not be wrapped in another action, like ConditionalAction
        """

        return True

    def embeds(self) -> List["Action"]:
        """
        Returns list of child embedded actions
        """

        return []

    def all_subactions(self):
        children = [self]
        for i in self.embeds():
            children.extend(i.all_subactions())
        return children

    def may_return_early(self):
        """
        Will this action return from _feed but still expect the state to be valid.
        """

        return False

class ConditionalAction(Action, HasDefaultDebugInfo):
    """
    Implements an entire if-else node chain
    """

    def __init__(self, conditions: List["DFCondition"], actions: Dict["DFCondition", List[Action]]):
        self.conditions = conditions
        self.sub_actions = actions

        for cond, acts in actions.items():
            ProgramData.imbue(cond, DTAG.PARENT, self)
            for act in acts:
                ProgramData.imbue(act, DTAG.PARENT, cond)

    def debug_lookup(self, tag):
        if tag == DTAG.NAME:
            return "if action"

    def get_mode(self):
        mode = None
        for act in itertools.chain(*self.sub_actions.values()):
            if mode is None:
                mode = act.get_mode()
            elif mode != act.get_mode():
                raise IllegalASTStateError("Action-only conditional node must be of only one type", act)
        return mode

    def is_timing_strict(self):
        # If the contained action could change something with the condition, this must be true
        for cond in self.conditions:
            if isinstance(cond, IntegerCondition):
                for act in self.sub_actions[cond]:
                    for potential in act.modifies():
                        for child in cond.expr.all_children():
                            if potential in child.accesses():
                                ProgramData.imbue(self, DTAG.STRICT_TIMING_REASON, f"expression could change containing conditional's outcome")
                                return True
        return any(x.is_timing_strict() for x in itertools.chain(*self.sub_actions.values()))

    def may_return_early(self):
        return any(x.may_return_early() for x in self.embeds())

    def get_target_override_targets(self):
        tgts = set()

        for act in itertools.chain(*self.sub_actions.values()):
            tgts.update(act.get_target_override_targets())

        return list(tgts)

    def embeds(self):
        return list(itertools.chain(*self.sub_actions.values()))

    def get_target_override_mode(self):
        mode = ActionOverrideMode.NONE
        for act in itertools.chain(*self.sub_actions.values()):
            submode = act.get_target_override_mode()

            if submode == ActionOverrideMode.ALWAYS_GOTO_UNDEFINED:
                submode = ActionOverrideMode.MAY_GOTO_UNDEFINED
            elif submode == ActionOverrideMode.ALWAYS_GOTO_OTHER:
                submode = ActionOverrideMode.MAY_GOTO_TARGET

            if submode.value > mode.value:
                mode = submode

        return mode

class FinishAction(Action, HasDefaultDebugInfo):
    def get_mode(self):
        return ActionMode.AT_FINISH

    def is_timing_strict(self):
        return True

    def get_target_override_mode(self):
        return ActionOverrideMode.ALWAYS_GOTO_UNDEFINED

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return "exit action"

class CustomFinishAction(FinishAction):
    def __init__(self, result_code: str):
        super().__init__()
        self.result_code = result_code

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return "exit action code {}".format(self.result_code)

class CustomYieldAction(Action, HasDefaultDebugInfo):
    def __init__(self, result_code: str):
        self.result_code = result_code

    def get_mode(self):
        return ActionMode.AT_FINISH

    def is_timing_strict(self):
        return True

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return "yield action code {}".format(self.result_code)

    def may_return_early(self):
        return True

class CallHook(Action, HasDefaultDebugInfo):
    def __init__(self, name):
        self.name = name

    def get_mode(self):
        return ActionMode.AT_FINISH

    def is_timing_strict(self):
        return True

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return f"call to hook {self.name}"

class BreakAction(Action, HasDefaultDebugInfo):
    def __init__(self, refers_to):
        self.refers_to: "LoopNode" = refers_to

    def get_mode(self):
        return ActionMode.AT_FINISH

    def is_timing_strict(self):
        return True

    def get_target_override_targets(self):
        return [self.refers_to.end_state]

    def get_target_override_mode(self):
        return ActionOverrideMode.ALWAYS_GOTO_OTHER

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return "break action for {}".format(ProgramData.lookup(self.refers_to, DTAG.NAME))

    def replacement_actions(self):
        return self.refers_to.after_break_actions

class AppendTo(Action, HasDefaultDebugInfo):
    def __init__(self, end_target, into_storage: "OutputStorage"):
        self.end_target = end_target
        self.into_storage = into_storage

    def get_target_override_mode(self):
        return ActionOverrideMode.MAY_GOTO_TARGET

    def is_timing_strict(self):
        return True

    def get_target_override_targets(self):
        return [self.end_target]

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return "append action ({})".format(ProgramData.lookup(self.into_storage, DTAG.NAME))

    def modifies(self):
        return [self.into_storage]

class AppendCharTo(Action, HasDefaultDebugInfo):
    def __init__(self, end_target, append_value: "IntegerExpr", into_storage: "OutputStorage"):
        self.end_target = end_target
        self.append_value = append_value
        self.into_storage = into_storage

    def get_mode(self):
        return ActionMode.AT_FINISH

    def get_target_override_mode(self):
        return ActionOverrideMode.MAY_GOTO_TARGET

    def get_target_override_targets(self):
        return [self.end_target]

    def is_timing_strict(self):
        return True

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return "append character action ({})".format(ProgramData.lookup(self.into_storage, DTAG.NAME))

    def modifies(self):
        return [self.into_storage]

class SetTo(Action, HasDefaultDebugInfo):
    def __init__(self, value_expr: "IntegerExpr", into_storage: "OutputStorage"):
        self.into_storage = into_storage
        self.value_expr = value_expr
        ProgramData.imbue(value_expr, DTAG.PARENT, self)

    def get_mode(self):
        return ActionMode.AT_FINISH

    def is_timing_strict(self):
        return any(self.into_storage in x.accesses() for x in self.value_expr.all_children())

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            if self.value_expr.is_literal():
                return "set into {} {}".format(ProgramData.lookup(self.into_storage, DTAG.NAME), self.value_expr.get_literal_result())
            else:
                return "set into {}".format(ProgramData.lookup(self.into_storage, DTAG.NAME))
        elif tag == DTAG.STRICT_TIMING_REASON:
            if not self.is_timing_strict():
                return None
            return "expression depends on previous stored value"

    def modifies(self):
        return [self.into_storage]

class SetToStr(Action, HasDefaultDebugInfo):
    def __init__(self, value_expr: str, into_storage: "OutputStorage"):
        self.into_storage = into_storage
        if into_storage.type != OutputStorageType.STR:
            raise IllegalASTStateError("SetToStr used without a string", self)
        self.value_expr = value_expr

    def get_mode(self):
        return ActionMode.AT_FINISH

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return "set into {} {!r}".format(ProgramData.lookup(self.into_storage, DTAG.NAME), self.value_expr)

    def modifies(self):
        return [self.into_storage]

class DeleteBuf(Action, HasDefaultDebugInfo):
    def __init__(self, into_storage: "OutputStorage"):
        self.into_storage = into_storage
        if not into_storage.holds_buflike():
            raise IllegalASTStateError("DeleteBuf used without a buflike", self)

    def get_mode(self):
        return ActionMode.AT_FINISH

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return "clear {}".format(ProgramData.lookup(self.into_storage, DTAG.NAME))

    def modifies(self):
        return [self.into_storage]



class Node(abc.ABC):
    """
    Represents an arbitrary AST node:
    - contains a pointer to the next node
    - processed bottom up
    - convertable to a full DFA
    """

    @abc.abstractmethod
    def set_next(self, next_node: "Node"):  # pragma: no cover
        """
        Set the next node that follows this node.
        _MUST_ be called before convert _UNLESS_ there is no next node
        """

        pass

    @abc.abstractmethod
    def get_next(self) -> "Node":  # pragma: no cover
        """
        Get the next node
        """
        pass

    @abc.abstractmethod
    def convert(self, current_error_handlers: dict) -> DFA:
        return None

class ActionSourceNode:
    def adopt_actions_from(self) -> Tuple[List[Action], Node]:  # pragma: no cover
        """
        Adopt the actions from this node, returning the actions that should be adopted and
        the new node which should be used instead of this node. If adopting is _not_ destructive,
        this should just return self
        """

        return [], self

class ActionNode(Node, ActionSourceNode):
    """
    An ephemeral node which represents a single node. There is guaranteed to only be one of these at any time and
    only as the topmost node in a stack
    """

    def __init__(self, *actions):
        self.actions = list(actions)
        for act in self.actions:
            ProgramData.imbue(act, DTAG.PARENT, self)
        self.next = None

    def set_next(self, next_node):
        if isinstance(next_node, ActionSourceNode):
            # adopt this node
            new_actions, self.next = next_node.adopt_actions_from()
            self.actions.extend(new_actions)
        else:
            # just set next
            self.next = next_node

    def get_next(self):
        return self.next

    def convert(self, *args):
        raise IllegalASTStateError("Action not inherited by a match left in AST", self)

    def adopt_actions_from(self):
        actions = self.actions
        return actions, self.next

class ActionSinkNode(Node):
    """
    Represents a node that can accept/adopt ActionNodes into itself, distributing them to appropriate
    Match objects (or deferring that process to sometime before conversion and hiding the ActionNodes from the sequence
    """

    @abc.abstractmethod
    def _set_next(self, actual_next_node: Node):  # pragma: no cover
        """
        Called by our set_next when the next node is not an action type
        """
        pass

    @abc.abstractmethod
    def _adopt_actions(self, actions: List[Action]):  # pragma: no cover
        pass

    def set_next(self, next_node):
        if isinstance(next_node, ActionSourceNode):
            actions, new_next = next_node.adopt_actions_from()
            self._adopt_actions(actions)
            self._set_next(new_next)
        else:
            self._set_next(next_node)

class InterruptableActionNode(ActionSinkNode):
    """
    Like an action node, but all actions after the first are "deferred" via a fallthrough else transition such that the first action
    can interrupt the main parser without causing subsequent actions to be missed.
    """

    def __init__(self, important_action: Optional[Action]):
        self.important_action = important_action  # if null, this is valid to construct
        self.following_actions = []
        self.next = None

    def _as_cleaned(self):
        newobj = InterruptableActionNode(None)
        newobj.following_actions = self.following_actions[:]
        newobj.next = self.next
        return newobj

    def get_next(self):
        return self.next

    def _set_next(self, actual_next_node):
        self.next = actual_next_node

    def _adopt_actions(self, act):
        self.following_actions.extend(act)

    def convert(self, current_error_handlers: dict):
        #if self.important_action is not None:
        #    raise IllegalASTStateError("Action not inherited by node in AST", self.important_action)

        dfa = DFA()
        start_node = DFProxyState()
        interrupt_node = DFProxyState()
        dfa.add(start_node)
        start_node[DFTransition.Else] = interrupt_node
        start_node[DFTransition.Else].fallthrough().attach(self.important_action)
        dfa.add(interrupt_node)
        trans = DFTransition(on_values=[DFTransition.Else]).fallthrough().attach(*self.following_actions)

        if self.next is None:
            fake_final = DFState()
            dfa.add(fake_final)
            dfa.mark_accepting(fake_final)
            trans.handles_else().to(fake_final)
            start_node[DFTransition.Else].handles_else()
        else:
            after = self.next.convert(current_error_handlers)
            for state in after.states:
                dfa.add(state)
                if state in after.accepting_states:
                    dfa.mark_accepting(state)
            trans.to(after.starting_state)
        interrupt_node.transition(trans)

        return dfa

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
    def convert(self, current_error_handlers: dict) -> DFA:  # pragma: no cover
        return None

    def attach(self, action: Action):
        if action.get_mode() == ActionMode.AT_FINISH:
            self.finish_actions.append(action)
        elif action.get_mode() == ActionMode.EACH_CHARACTER:
            ProgramData.imbue(action, DTAG.PARENT, self)
            self.char_actions.append(action)
        else:
            self.start_actions.append(action)

class DirectMatch(Match, HasDefaultDebugInfo):
    """A direct string match"""
    def __init__(self, match_contents):
        super().__init__()
        self.match_contents = match_contents

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return f"direct match {self.match_contents!r}"

    def convert(self, current_error_handlers: dict):
        sm = DFA() # Create a new SM
        ProgramData.imbue(sm, DTAG.PARENT, self) # Mark us as the parent of this SM
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
            state[DFTransition.Else] = DFTransition().to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*start_action_holder).fallthrough().handles_else()
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

    def debug_lookup(self, tag: DTAG):
        if tag == DTAG.NAME:
            return f"casei match {self.match_contents!r}"

    def _create_casei_from(self, character: str):
        # TODO: UNICODE handling
        if character in string.ascii_letters:
            return [character, string.ascii_letters[(string.ascii_letters.index(character) + 26) % len(string.ascii_letters)]]
        else:
            return [character]

    def convert(self, current_error_handlers: dict):
        sm = DFA() # Create a new SM
        ProgramData.imbue(sm, DTAG.PARENT, self) # Mark us as the parent of this SM
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
            state[DFTransition.Else] = DFTransition().to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*start_action_holder).fallthrough().handles_else()
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
    RAW = 4

class OutputStorage(HasDefaultDebugInfo):
    def __init__(self, typ: OutputStorageType, name, default_value=None, enum_values: List[str] = [], str_size=0, str_null=True, int_signed=True, int_width=None, raw_underlying=""):
        self.type = typ
        self.name = name
        self.default_value = default_value
        self.enum_values = enum_values
        self.str_size = str_size
        self.str_null = str_null
        self.int_signed = int_signed
        self.int_width = int_width
        self.raw_underlying = raw_underlying

    def holds_a(self, typ):
        return self.type == typ

    def holds_buflike(self):
        return self.type in [OutputStorageType.STR, OutputStorageType.RAW]

    def debug_lookup(self, tag):
        if tag == DTAG.NAME:
            return f"output '{self.name}'"

    def __repr__(self):
        return f"<OutputStorage name={self.name} type={self.type}>"

    def effective_string_size(self):
        if self.str_null:
            return self.str_size - 1
        return self.str_size

# ==================
# INTEGER EXPR TYPES
# ==================

class IntegerExprUseContext(enum.Enum):
    ASSIGN_ON_MATCH = 0
    ASSIGN_INITIAL = 1
    ASSIGN_ON_END = 2
    CONDITION_PREDICATE = 3
    CONDITION_PREDICATE_END = 4
    CONDITION_PREDICATE_ACTION = 5
    CONDITION_PREDICATE_ACTION_END = 6

class IntegerExpr(abc.ABC):
    @abc.abstractmethod
    def result_type(self) -> OutputStorageType:  # pragma: no cover
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

    def get_literal_result(self):  # pragma: no cover
        """
        Evaluate to a native python representation
        """
        return None

    def get_invalid_contexts(self):
        """
        Return an iterable of all the contexts in which this expression is invalid
        """
        return []

    def get_child_expressions(self):
        """
        Return an interable of all the children expressions contained within this expression
        """
        return []

    def all_children(self):
        """
        Recursively find child expressions
        """

        children = [self]
        for i in self.get_child_expressions():
            children.extend(i.all_children())
        return children

    def accesses(self) -> List["OutputStorage"]:
        """
        Return which outputs this expression accesses
        """

        return []

class LiteralIntegerExpr(IntegerExpr):
    def __init__(self, value, typ: OutputStorageType = OutputStorageType.INT, model_ref: OutputStorage=None):
        self.value = value
        self.typ = typ
        self.model_ref = model_ref

    def result_type(self):
        return self.typ

    def is_literal(self):
        return True

    def get_literal_result(self):
        return self.value

    def __eq__(self, other):
        if not isinstance(other, LiteralIntegerExpr): return False
        return self.value == other.value and self.typ == other.typ

class OutIntegerExpr(IntegerExpr):
    def __init__(self, ref: "OutputStorage"):
        self.ref = ref

    def result_type(self):
        return self.ref.type

    def __eq__(self, other):
        if not isinstance(other, OutIntegerExpr): return False
        return other.ref == self.ref

    def accesses(self):
        return [self.ref]

class StringRefIntegerExpr(IntegerExpr):
    def __init__(self, ref: "OutputStorage", index: IntegerExpr):
        if not ref.holds_buflike():
            raise IllegalParseTree("Index expression must be indexing a string", self)

        if not index.result_type() == OutputStorageType.INT:
            raise IllegalParseTree("Index expression must be an integer", index)

        ProgramData.imbue(index, DTAG.PARENT, self)

        self.ref = ref
        self.index = index

    def result_type(self):
        return OutputStorageType.INT

    def __eq__(self, other):
        if not isinstance(other, StringRefIntegerExpr): return False
        return self.ref == other.ref and self.index == other.index

    def accesses(self):
        return [self.ref, *self.index.accesses()]

    def get_child_expressions(self):
        return [self.index]

class StringLengthIntegerExpr(IntegerExpr):
    def __init__(self, ref: "OutputStorage"):
        if not ref.holds_buflike():
            raise IllegalParseTree("Index expression must be indexing a buffer-like object", self)

        self.ref = ref

    def __eq__(self, other):
        if not isinstance(other, StringLengthIntegerExpr): return False
        return self.ref == other.ref

    def accesses(self):
        return [self.ref]

    def result_type(self):
        return OutputStorageType.INT

class LastCharIntegerExpr(IntegerExpr):
    def result_type(self):
        return OutputStorageType.INT

    def get_invalid_contexts(self):
        return [IntegerExprUseContext.ASSIGN_INITIAL, IntegerExprUseContext.ASSIGN_ON_END, IntegerExprUseContext.CONDITION_PREDICATE_END, IntegerExprUseContext.CONDITION_PREDICATE, IntegerExprUseContext.CONDITION_PREDICATE_ACTION_END]

    def __eq__(self, other):
        return isinstance(other, LastCharIntegerExpr)

class MathIntegerExpr(IntegerExpr):
    def __init__(self, children: List[IntegerExpr], valid_type=OutputStorageType.INT):
        self.children = children
        
        for child in self.children:
            ProgramData.imbue(child, DTAG.PARENT, self)

        if not self.children:
            raise IllegalParseTree("Empty math expression", self)
    
        if valid_type is not None:
            if not all(x.result_type() == valid_type for x in self.children):
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

    def __eq__(self, other):
        return type(other) == type(self) and self.children == other.children

    def get_child_expressions(self):
        return self.children

    def accesses(self):
        acc = []
        for i in self.children:
            acc.extend(i.accesses())
        return acc

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

    def __eq__(self, other):
        if not isinstance(other, SumIntegerExpr): return False
        return set(zip(self.children, self.negate)) == set(zip(other.children, other.negate))

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

    def __eq__(self, other):
        if not isinstance(other, MulIntegerExpr): return False
        return self.children == other.children and self.divide == other.divide

class CompareIntegerExprOp(enum.Enum):
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    EQ = "=="
    NE = "!="

class CompareIntegerExpr(MathIntegerExpr):
    def __init__(self, left: IntegerExpr, right: IntegerExpr, op: CompareIntegerExprOp):
        super().__init__([left, right], valid_type=None)
        self.left = left
        self.right = right
        self.op = op

    def __eq__(self, other):
        if not isinstance(other, CompareIntegerExpr): return False
        return self.left == other.left and self.right == other.right and self.op == other.op

    def result_type(self):
        return OutputStorageType.BOOL

    def get_literal_result(self):
        left = self.left.get_literal_result()
        right = self.right.get_literal_result()

        if self.op == CompareIntegerExprOp.LT:
            return left < right
        elif self.op == CompareIntegerExprOp.GT:
            return left > right
        elif self.op == CompareIntegerExprOp.LE:
            return left <= right
        elif self.op == CompareIntegerExprOp.LE:
            return left >= right
        elif self.op == CompareIntegerExprOp.EQ:
            return left == right
        elif self.op == CompareIntegerExprOp.NE:
            return left != right
        else:
            return False

class BitShiftIntegerExpr(MathIntegerExpr):
    def __init__(self, left: IntegerExpr, right: IntegerExpr, towards_left: bool):
        super().__init__([left, right])
        self.left = left
        self.right = right
        self.towards_left = towards_left

    def __eq__(self, other):
        if not isinstance(other, BitShiftIntegerExpr): return False
        return self.left == other.left and self.right == other.right and self.towards_left == other.towards_left

    def get_literal_result(self):
        left = self.left.get_literal_result()
        right = self.right.get_literal_result()

        if self.towards_left:
            return left << right
        else:
            return left >> right

class BitwiseIntegerExprOp(enum.Enum):
    OR = "|"
    XOR = "^"
    AND = "&"

class BitwiseIntegerExpr(MathIntegerExpr):
    def __init__(self, children: List[IntegerExpr], operation: BitwiseIntegerExprOp):
        super().__init__(children)
        self.op = operation

    def __eq__(self, other):
        if not isinstance(other, BitwiseIntegerExpr): return False
        return self.children == other.children and self.op == other.op

    def get_literal_result(self):
        total = self.children[0].get_literal_result()

        for child in self.children[1:]:
            if self.op == BitwiseIntegerExprOp.OR:
                total |= child.get_literal_result()
            elif self.op == BitwiseIntegerExprOp.XOR:
                total ^= child.get_literal_result()
            else:
                total &= child.get_literal_result()

        return total

class DisjunctionIntegerExpr(MathIntegerExpr):
    def __init__(self, children):
        super().__init__(children, valid_type=OutputStorageType.BOOL)

    def get_literal_result(self):
        return any(x.get_literal_result() for x in self.children)

class ConjunctionIntegerExpr(MathIntegerExpr):
    def __init__(self, children):
        super().__init__(children, valid_type=OutputStorageType.BOOL)

    def get_literal_result(self):
        return all(x.get_literal_result() for x in self.children)

# ==========
# CONDITIONS
# ==========

class DFCondition:
    def __init__(self):
        pass

    def is_literal(self):
        """
        Is this condition's value computable now
        """

        return False

    def get_literal_result(self):
        """
        Is this condition active now?
        """

        return False

    def conflicts_with(self, other):
        return False

    def make_nonconflicting(self, other):
        """
        Returns (only_self, both, only_other)
        """

        return (self, ConstantCondition(False), other)

class ConstantCondition(DFCondition):
    def __init__(self, value: bool):
        self.value = value

    def is_literal(self): return True
    def get_literal_result(self): return self.value

    def conflicts_with(self, other):
        return self.value

    def make_nonconflicting(self, other):
        if self.value:
            return (ConstantCondition(False), other, ConstantCondition(False))
        else:
            return super().make_nonconflicting(other)

class ElseCondition(ConstantCondition, HasDefaultDebugInfo):
    def __init__(self):
        super().__init__(True)

    def debug_lookup(self, tag):
        if tag == DTAG.NAME:
            return "else"
        return None

class IntegerCondition(DFCondition, HasDefaultDebugInfo):
    def __init__(self, expr: IntegerExpr):
        if expr.result_type() != OutputStorageType.BOOL:
            if expr.result_type() == OutputStorageType.INT:
                expr = CompareIntegerExpr(expr, LiteralIntegerExpr(0), CompareIntegerExprOp.NE)
            else:
                raise IllegalParseTree("Condition must be convertable to bool", expr)

        if IntegerExprUseContext.CONDITION_PREDICATE in expr.get_invalid_contexts() and IntegerExprUseContext.CONDITION_PREDICATE_ACTION in expr.get_invalid_contexts():
            raise IllegalParseTree("Expression not valid for use as a predicate", expr)

        self.expr = expr

    def is_literal(self): return self.expr.is_literal()
    def get_literal_result(self): return self.expr.get_literal_result()

    def debug_lookup(self, tag):
        if tag == DTAG.NAME:
            return "if condition"
        return None


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
        ProgramData.imbue(sub_match, DTAG.PARENT, self)
        self.sub_match = sub_match

class RegexOptional:
    def __init__(self, sub_match):
        ProgramData.imbue(sub_match, DTAG.PARENT, self)
        self.sub_match = sub_match

class RegexAlternation:
    def __init__(self, sub_matches):
        self.sub_matches = set(sub_matches)
        for sub_match in self.sub_matches:
            ProgramData.imbue(sub_match, DTAG.PARENT, self)

class RegexSequence:
    def __init__(self, sub_matches):
        self.sub_matches = list(sub_matches)
        for sub_match in self.sub_matches:
            ProgramData.imbue(sub_match, DTAG.PARENT, self)

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
        ProgramData.imbue(visited_states[start_dfa_state], DTAG.PARENT, self.states[next(iter(start_dfa_state))])

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
                        ProgramData.imbue(visited_states[new_state], DTAG.PARENT, self.states[next(iter(new_state))])
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
            ProgramData.imbue(new_state, DTAG.PARENT, state)

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
        ProgramData.imbue(self.regex_tree, DTAG.PARENT, self)

        # Variables used during construction (similarly to how mlang works, to reduce arguments in recursive methods)
        self.nfa: Optional[RegexNFA] = None
        self.dfa_1: Optional[RegexNFA] = None
        self.dfa_2: Optional[RegexNFA] = None
        self.out_dfa_cache = {}

    def _convert_to_nfa(self, r, start_state: RegexNFState):
        """
        Convert the regex tree object into the NFA using Thompson construction. Return the finish state
        """
        ProgramData.imbue(start_state, DTAG.PARENT, r)

        if isinstance(r, RegexCharClass):
            # Simply convert to a boring form
            end_state = RegexNFState()
            ProgramData.imbue(end_state, DTAG.PARENT, start_state)
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
            ProgramData.imbue(end_state, DTAG.PARENT, start_state)
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
            ProgramData.imbue(end_state, DTAG.PARENT, start_state)
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
            ProgramData.imbue(end_state, DTAG.PARENT, start_state)
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
        tree_data = regex_tree.data
        if tree_data.startswith("binary_"):
            tree_data = tree_data[len("binary_"):]
        if tree_data in ("regex", "regex_group"):
            return RegexSequence(self._interpret_parse_tree(x) for x in regex_tree.children)
        elif tree_data in ("regex_any", "regex_char_class", "regex_set", "regex_inverted_set"):
            return ProgramData.imbue(
                RegexAlternation(self.character_class_mappings[list(self._visit_all_char_classes(regex_tree))[0]]),
                DTAG.SOURCE_LINE, regex_tree.meta.line,
                DTAG.SOURCE_COLUMN, regex_tree.meta.column
            )
        elif tree_data == "regex_alternation":
            return ProgramData.imbue(
                RegexAlternation(self._interpret_parse_tree(x) for x in regex_tree.children),
                DTAG.SOURCE_LINE, regex_tree.meta.line,
                DTAG.SOURCE_COLUMN, regex_tree.meta.column
            )
        elif tree_data == "regex_raw_match":
            return ProgramData.imbue(
                RegexAlternation(self.character_class_mappings[self._convert_raw_regex_unimportant(regex_tree.children[0])]),
                DTAG.SOURCE_LINE, regex_tree.meta.line,
                DTAG.SOURCE_COLUMN, regex_tree.meta.column
            )
        elif tree_data == "regex_operation":
            sub_match = self._interpret_parse_tree(regex_tree.children[0])
            if regex_tree.children[1].value == "+":
                val = RegexSequence((sub_match, RegexKleene(sub_match)))
            else:
                val = {"*": RegexKleene, "?": RegexOptional}[regex_tree.children[1].value](sub_match)
            ProgramData.imbue(val, DTAG.SOURCE_LINE, regex_tree.children[1].line)
            ProgramData.imbue(val, DTAG.SOURCE_COLUMN, regex_tree.children[1].column)
            return val
        elif tree_data == "regex_exact_repeat":
            repeated_match = self._interpret_parse_tree(regex_tree.children[0])
            repeat_times   = int(regex_tree.children[1].value)
            return ProgramData.imbue(
                RegexSequence(itertools.repeat(repeated_match, repeat_times)),
                DTAG.SOURCE_LINE, regex_tree.children[1].line,
                DTAG.SOURCE_COLUMN, regex_tree.children[1].column
            )
        elif tree_data == "regex_at_least_repeat":
            repeated_match = self._interpret_parse_tree(regex_tree.children[0])
            repeat_times   = int(regex_tree.children[1].value)
            return ProgramData.imbue(
                RegexSequence([repeated_match for x in range(repeat_times)] + [RegexKleene(repeated_match)]),
                DTAG.SOURCE_LINE, regex_tree.children[1].line,
                DTAG.SOURCE_COLUMN, regex_tree.children[1].column
            )
        elif tree_data == "regex_range_repeat":
            repeated_match = self._interpret_parse_tree(regex_tree.children[0])
            repeat_times_min = int(regex_tree.children[1].value)
            repeat_times_max = int(regex_tree.children[2].value)
            return ProgramData.imbue(RegexSequence(itertools.chain(
                itertools.repeat(repeated_match, repeat_times_min),
                itertools.repeat(RegexOptional(repeated_match), repeat_times_max - repeat_times_min)
            )),
                DTAG.SOURCE_LINE, regex_tree.children[1].line,
                DTAG.SOURCE_COLUMN, regex_tree.children[1].column,
            )
        else:
            raise NotImplementedError("don't handle {} yet".format(tree_data))

    def _convert_raw_regex_unimportant(self, regex_tree: lark.Token):
        if regex_tree.value[0] == '\\':
            v = RegexCharClass((regex_tree.value[1],))
        else:
            v = RegexCharClass((regex_tree.value[0],))
        ProgramData.imbue(v, DTAG.SOURCE_LINE, regex_tree.line)
        ProgramData.imbue(v, DTAG.SOURCE_COLUMN, regex_tree.column)
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
        ProgramData.imbue(val, DTAG.SOURCE_LINE, regex_char_class.children[0].line)
        ProgramData.imbue(val, DTAG.SOURCE_COLUMN, regex_char_class.children[0].column)
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
        ProgramData.imbue(new_state, DTAG.PARENT, nfdfa_state)
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

            new_state.transition(DFTransition(list(source)).to(target).attach(*actions).fallthrough(target == else_path).handles_else(target == else_path))

        return new_state


    def convert(self, current_error_handlers: dict):
        # First convert to an NFA
        self.nfa = RegexNFA()
        start_state = RegexNFState()
        self.nfa.add(start_state)
        self.nfa.mark_finishing(self._convert_to_nfa(self.regex_tree, start_state))
        # Then convert to a DFA

        new_dfa = RegexNFA()
        ProgramData.imbue(new_dfa, DTAG.PARENT, self)
        self.dfa_1 = self.nfa.convert_to_dfa(self.alphabet, new_dfa)
        # Minimize the DFA

        new_dfa = RegexNFA()
        ProgramData.imbue(new_dfa, DTAG.PARENT, self)
        self.dfa_2 = self.dfa_1.minimize_dfa(self.alphabet, new_dfa)
        
        # Check if it's possible to schedule the finish actions. TODO: instead of throwing an error, attempt to move them to the next thing's start
        if any(x.is_timing_strict() for x in self.finish_actions) and any(x.transitions for x in self.dfa_2.finishing_states):
            # Complain early
            raise UnableToScheduleActionError((x for x in self.dfa_2.finishing_states if x.transitions), (x for x in self.finish_actions if x.is_timing_strict()))

        # Create a normal SM
        out_dfa = DFA()
        self._create_dfa_state(self.dfa_2.start_state, out_dfa, True, current_error_handlers[ErrorReasons.NO_MATCH])
        return ProgramData.imbue(out_dfa, DTAG.PARENT, self)

class BinaryRegexMatch(RegexMatch):
    def _visit_all_char_classes(self, regex_tree: lark.Tree):
        if regex_tree.data in ("binary_regex", "binary_regex_group", "binary_regex_alternation"):
            result = set()
            for child in regex_tree.children:
                result |= self._visit_all_char_classes(child)
            return result
        if regex_tree.data == "binary_regex_raw_match":
            return set(self._convert_raw_regex_unimportant(x) for x in regex_tree.children)
        if regex_tree.data == "binary_regex_any":
            return set((InvertedRegexCharClass(()),))
        if regex_tree.data in ("binary_regex_operation", "binary_regex_exact_repeat", "binary_regex_range_repeat", "binary_regex_at_least_repeat"):
            return self._visit_all_char_classes(regex_tree.children[0])
        if regex_tree.data in ("binary_regex_set", "binary_regex_inverted_set"):
            inverted = regex_tree.data != "binary_regex_set"
            incoming_set = RegexCharClass(())
            for child in regex_tree.children:
                if isinstance(child, lark.Token):
                    new_set = self._convert_raw_regex_unimportant(child)
                elif child.data == "binary_regex_set_range":
                    start = list(self._convert_raw_regex_unimportant(child.children[0]).chars)[0]
                    end = list(self._convert_raw_regex_unimportant(child.children[1]).chars)[0]
                    new_set = RegexCharClass(chr(x) for x in range(ord(start), ord(end)+1))
                incoming_set = incoming_set.union(new_set)
            if not inverted:
                return set((incoming_set,))
            else:
                return set((incoming_set.invert(),))

    def _convert_raw_regex_unimportant(self, byte: lark.Token):
        v = RegexCharClass((chr(int(byte.value, base=16)),))
        ProgramData.imbue(v, DTAG.SOURCE_LINE, byte.line)
        ProgramData.imbue(v, DTAG.SOURCE_COLUMN, byte.column)
        return v

class WaitMatch(Match):
    def __init__(self, sub_match: Match):
        super().__init__()
        ProgramData.imbue(sub_match, DTAG.PARENT, self)
        self.match_contents = sub_match

    def attach(self, action: Action):
        self.match_contents.attach(action)
        super().attach(action)

    def convert(self, current_error_handlers: dict):
        sm = self.match_contents.convert(current_error_handlers)
        for state, trans in sm.transitions_pointing_to(current_error_handlers[ErrorReasons.NO_MATCH], True):
            trans.to(sm.starting_state).handles_else()  # we make these error handling since that makes semantic sense for the usual use case for a wait node
            if state == sm.starting_state:
                trans.fallthrough(False).attach(*self.char_actions)
        return sm

class EndMatch(Match):
    def convert(self, current_error_handlers: dict):
        """
            END
        s   -->  s
         0        1
        """
        sm = DFA()
        ProgramData.imbue(sm, DTAG.PARENT, self)
        start_state = DFState()
        sm.add(start_state)
        ok_state = DFState()
        sm.add(ok_state)
        sm.mark_accepting(ok_state)
        start_state.transition(DFTransition([DFTransition.End]).attach(*self.start_actions, *self.char_actions, *self.finish_actions).to(ok_state))
        start_state.transition(DFTransition([DFTransition.Else]).to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*self.start_actions).fallthrough().handles_else())
        return sm

class ConcatMatch(Match):
    def __init__(self, sub_matches: List[Match]):
        super().__init__()
        self.sub_matches = sub_matches
        for i in sub_matches:
            ProgramData.imbue(i, DTAG.PARENT, self)

    def convert(self, current_error_handlers: dict):
        # distribute actions
        self.sub_matches[0].start_actions.extend(self.start_actions)
        self.sub_matches[-1].finish_actions.extend(self.finish_actions)
        for i in self.sub_matches:
            i.char_actions.extend(self.char_actions.copy())
        # convert in order
        sm = self.sub_matches[0].convert(current_error_handlers)
        for i in self.sub_matches[1:]:
            sm.append_after(i.convert(current_error_handlers))
        return sm

class MatchNode(ActionSinkNode):
    """
    Node which executes a match.
    """

    def __init__(self, match: Match):
        ProgramData.imbue(match, DTAG.PARENT, self)
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
            base_dfa.append_after(self.next.convert(current_error_handlers))
        return base_dfa

class CaseNode(Node):
    """
    Handles cases.
    """

    def __init__(self, sub_matches: Dict[Set[Optional[Match]], Node], greedy: bool = False, priorities: Dict[Set[Optional[Match]], int] = None):
        super().__init__()
        self.sub_matches = {k: v for k, v in sub_matches.items() if v is not None}
        self.empty_matches = [k for k, v in sub_matches.items() if v is None]
        self.priorities = defaultdict(int)
        if priorities:
            self.priorities.update(priorities)
        self.case_match_actions = defaultdict(list)
        self.next = None
        self.greedy = greedy

        self._find_case_actions()

    def _find_case_actions(self):
        """
        Find all the case actions and delete ActionNodes
        """

        to_replace = []

        for sub_matches, target in self.sub_matches.items():
            if not isinstance(target, ActionSourceNode):
                continue

            actions, new_next = target.adopt_actions_from()

            # adopt into our internal list
            self.case_match_actions.update({sub_matches: actions})
            if new_next is not None:
                self.sub_matches[sub_matches] = new_next
            else:
                to_replace.append(sub_matches)

        for i in to_replace:
            del self.sub_matches[i]
            self.empty_matches.append(i)

    def _merge(self, ds: Iterable[DFA], error_handling_state: DFState, priorities: Dict[DFA, int]):
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

        In order to deal with the complexities of the DFA representation in NMFU, we do a simplification pass on the input alphabet. For each "superstate" -- that is, an NFA state (set
        of DFA states), we consider a "local alphabet", which is the set of all sets of symbols matched by all transitions in all substates. We then do a pass on this to convert those
        sets into disjoint sets, forming the true "local alphabet". We then use these to generate the transitions on the new DFA state, considering elements of the local alphabet
        which no substates match to form the set of error symbols -- or the "Else" transition, perhaps.

        We also ignore all transitions that go to the entry marked as error-handling, as we re-create all the else transitions later.

        We start in state (D1,D1_start),(D2,D2_start),...,(Dn,Dn_start)
        """

        new_dfa = DFA()

        converted_states = {}
        corresponding_finish_states = {dfa: [] for dfa in ds}
        to_process = queue.Queue()

        def create_real_state_of(state):
            new_state = DFState()
            ProgramData.imbue(new_state, DTAG.PARENT, next(iter(state))[1])

            corresponds_to_finishes_in = set() 
            is_part_of = set()
            for sub_dfa, sub_state in state:
                if sub_state in sub_dfa.accepting_states:
                    corresponds_to_finishes_in.add(sub_dfa)
                is_part_of.add(sub_dfa)

            if len(corresponds_to_finishes_in) > 1:
                if not self.greedy:
                    raise IllegalDFAStateConflictsError("Ambigious case label: multiple possible finishes", *corresponds_to_finishes_in)
                target = max(corresponds_to_finishes_in, key=lambda x: priorities[x])
                if sum(1 for x in corresponds_to_finishes_in if priorities[x] == priorities[target]) > 1:
                    raise IllegalDFAStateConflictsError("Ambigious case label: multiple possible finishes with same priority {}".format(priorities[target]), 
                            *(x for x in corresponds_to_finishes_in if priorities[x] == priorities[target]))
                corresponding_finish_states[target].append(new_state)
                new_dfa.mark_accepting(new_state)
            elif len(corresponds_to_finishes_in) == 1:
                if len(is_part_of) != 1 and not self.greedy:
                    raise IllegalDFAStateConflictsError("Ambigious case label: should finish or check next. If you mean to finish, use a greedy case.", *is_part_of)
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

            # set of all symbols this dfa state needs to match against. should be disjoint sets
            local_alphabet = set()
            # set of all symbols which none of the states in this superstate match on (and should therefore
            # be directed to the else state)
            actual_else = set()

            # find all possible symbols that this superstate needs to check for
            for _, sub_state in processing:
                for trans in sub_state.transitions:
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

            # process all moves with those sets as the alphabet

            for symbol in local_alphabet:
                # construct the set of next states
                next_state = set()
                # did we find a transition that marked this symbol as being an error?
                # since all symbols are now disjoint, two situations occur if this is set:
                #   a) at least one state correctly matches this symbol, and the others do not
                #      in this case, we should just discard those other states (since there is no
                #      reason to "speculatively execute" the error state, obviously)
                #   b) none of the states correctly match this symbol, in which case it is an overall error.
                #      in this case, we should add the symbol to the list of symbols that we will assign
                #      an error transition to
                found_else_transition = False
                for (sub_dfa, sub_state) in processing:
                    potential_transition = sub_state[symbol]
                    if potential_transition is None:
                        continue
                    if potential_transition.error_handling:
                        found_else_transition = True
                        continue
                    else:
                        next_state.add((sub_dfa, potential_transition.target))

                # if case b) is met 
                if not next_state:
                    actual_else |= symbol
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

            if DFTransition.Else in actual_else:
                # just use it
                actual_else = set((DFTransition.Else,))
            
            if actual_else: # sometimes you actually don't need one
                converted_states[processing].transition(DFTransition(list(actual_else)).to(error_handling_state).fallthrough().handles_else(), allow_replace=True)

        return new_dfa, corresponding_finish_states

    def get_next(self):
        return self.next

    def set_next(self, next_node):
        if isinstance(next_node, ActionSourceNode):
            actions, our_next_node = next_node.adopt_actions_from()
            for sub_ast in self.sub_matches.values():
                # Go to end of sub_ast
                while sub_ast.get_next() is not None:
                    sub_ast = sub_ast.get_next()

                # Adopt it
                new_action_node = ActionNode(*actions)
                sub_ast.set_next(new_action_node)
            for empty_match in self.empty_matches:
                self.case_match_actions[empty_match].extend(actions)
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
        # Their priorities
        priorities = {}
        # Flatten the sub_matches
        for sub_matches in self.sub_matches:
            for sub_match in sub_matches:
                if sub_match is not None:
                    converted = sub_match.convert(current_error_handlers)
                    original_backreference[converted] = sub_matches
                    mergeable_ds.add(converted)
                    priorities[converted] = self.priorities[sub_matches]
                else:
                    original_backreference[None] = sub_matches
        for empty_matches in self.empty_matches:
            for empty_match in empty_matches:
                if empty_match is not None:
                    converted = empty_match.convert(current_error_handlers)
                    original_backreference[converted] = None
                    empty_backreference[converted] = empty_matches
                    mergeable_ds.add(converted)
                    priorities[converted] = self.priorities[empty_matches]
                else:
                    original_backreference[None] = None
                    empty_backreference[None] = None

        # Create the merged acceptor
        decider_dfa, corresponding_finish_states = self._merge(mergeable_ds, current_error_handlers[ErrorReasons.NO_MATCH], priorities)

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
                    trans.to(sub_dfas[original_backreference[None]].starting_state).attach(*else_actions).handles_else()
                    ProgramData.imbue(trans, DTAG.NAME, "else action on case node")
                    ProgramData.imbue(trans, DTAG.PARENT, self)
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
                    trans.to(new_state).attach(*else_actions, prepend=True).fallthrough().handles_else()  # this is an error handler, make sure it's a fallthrough
                    # give the transition better debug info
                    ProgramData.imbue(trans, DTAG.NAME, "else action on case node")
                    ProgramData.imbue(trans, DTAG.PARENT, self)

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
                        raise UnableToScheduleActionError([i], [x for x in self.case_match_actions[true_backref] if x.is_timing_strict()])
                    # Add actions
                    for j in all_transitions_empty:
                        j.attach(*self.case_match_actions[true_backref], prepend=True)
            else:
                refers_to = sub_dfas[original_backreference[i]]
                decider_dfa.append_after(refers_to, sub_states=corresponding_finish_states[i], chain_actions=self.case_match_actions[original_backreference[i]])

        ProgramData.imbue(decider_dfa, DTAG.PARENT, self)

        if self.next is not None:
            decider_dfa.append_after(self.next.convert(current_error_handlers))
        
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
            sub_dfa.append_after(self.next.convert(current_error_handlers), chain_actions=self.finish_actions)
        else:
            sub_dfa.chain_actions_at_end(self.finish_actions)

        return sub_dfa

class LoopNode(ActionSinkNode, ActionSourceNode):
    def __init__(self, name):
        self.name = name
        self.next = None
        self.after_break_actions = []
        self.loop_start_actions = []
        self.child_node = None
        self.end_state = DFState()
        ProgramData.imbue(self.end_state, DTAG.PARENT, self)

        self.break_action = BreakAction(self)
        ProgramData.imbue(self, DTAG.NAME, f"loop node {name}" if name else "anonymous loop")

    def _set_next(self, next_node):
        self.next = next_node

    def get_next(self):
        return self.next

    def _adopt_actions(self, actions):
        if any(x.get_mode() != ActionMode.AT_FINISH for x in actions):
            raise IllegalASTStateError("Invalid action type for loop", self)

        self.after_break_actions.extend(actions)

    def adopt_actions_from(self):
        # This will place the loop start actions in the right place for the first iteration. We add them to all iteration transitions too.
        return self.loop_start_actions, self

    def get_break_handler(self):
        return self.break_action

    def set_child(self, child: Node):
        if isinstance(child, ActionSourceNode):
            self.loop_start_actions, child = child.adopt_actions_from()
        self.child_node = child

    def convert(self, current_error_handlers):
        if self.child_node is None:
            raise IllegalDFAStateError("Empty loop body", self)

        parent_dfa = DFA()
        parent_dfa.add(self.end_state)
        parent_dfa.mark_accepting(self.end_state)

        if self.next:
            parent_dfa.append_after(self.next.convert(current_error_handlers))
        
        # First, create the sub_dfa 
        sub_dfa = self.child_node.convert(current_error_handlers)

        # don't mark it accepting yet

        should_try_to_append = False

        # Reroute all transitions with a BreakAction in them that corresponds to our break action to go to us immediately as an optimization.
        for transition in sub_dfa.transitions_that_do(self.break_action):
            transition.to(self.end_state)
            transition.actions.remove(self.break_action)
            transition.actions.extend(self.after_break_actions)
            should_try_to_append = True

        # Check if any actions are break actions
        for transition in sub_dfa.all_transitions():
            for action in transition.actions:
                for subaction in action.all_subactions():
                    if subaction == self.break_action:
                        should_try_to_append = True

        if self.break_action in self.loop_start_actions:
            should_try_to_append = True
        else:
            for x in self.loop_start_actions:
                if self.break_action in x.all_subactions():
                    should_try_to_append = True
                    break

        # Verify that the accepting states are all distinct TODO: make this properly do ambiguity resolution
        for accept_state in sub_dfa.accepting_states:
            for transition in accept_state.all_transitions():
                if transition.target in sub_dfa.accepting_states:
                    raise IllegalDFAStateConflictsError("Ambigious loop: should loop or continue matching", accept_state, transition.target)

        # If there are error-handling transitions on the accept node, point them to the starting node as fallthrough (so that anything that _isn't_ getting matched by 
        # the last node gets forwarded to the start, looping). If there are no transitions on the final node, point everything to the start.
        for accept_state in sub_dfa.accepting_states:
            for trans in accept_state.transitions:
                if trans.error_handling:
                    trans.handles_else(False).fallthrough().to(sub_dfa.starting_state).attach(*self.loop_start_actions)
            if not accept_state.transitions:
                accept_state[DFTransition.Else] = DFTransition(fallthrough=True).to(sub_dfa.starting_state).attach(*self.loop_start_actions)

        for state in sub_dfa.states:
            parent_dfa.add(state)

        parent_dfa.starting_state = sub_dfa.starting_state

        if self.next and not should_try_to_append:
            raise IllegalASTStateError("Unreachable states after loop", self.next)

        return parent_dfa

class TryExceptNode(ActionSinkNode, ActionSourceNode):
    def __init__(self, handles):
        self.handler_node = DFState()
        self.handles = handles
        self.after_actions = []
        self.incoming_handler_actions = []
        self.incoming_body_actions = []

        self.body = None
        self.handler = None

        self.next = None

    def adopt_actions_from(self):
        actions = self.incoming_body_actions
        return actions, self

    def get_handler(self):
        return self.handler_node

    def set_body(self, body):
        if isinstance(body, ActionSourceNode):
            self.incoming_body_actions, body = body.adopt_actions_from()
        self.body = body

    def set_handler(self, handler):
        if isinstance(handler, ActionSourceNode):
            self.incoming_handler_actions, handler = handler.adopt_actions_from()
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

        # This _should_ have transitions going to the handler node. Add it to the tree now
        sub_dfa.add(self.handler_node)

        # If there _is_ a handler, add it after
        if self.handler is not None:
            handler_dfa = self.handler.convert(current_error_handlers)
            sub_dfa.append_after(handler_dfa, sub_states=[self.handler_node], chain_actions=self.incoming_handler_actions)
        else:
            # Otherwise, create a fallthrough dummy transition to hook up the handler actions.
            dummy_end_node = DFState()
            # And add the finish actions to everything that pointed at it
            self.handler_node.transition(DFTransition([DFTransition.Else], fallthrough=True).to(dummy_end_node).attach(*self.incoming_handler_actions))
            sub_dfa.add(dummy_end_node)
            sub_dfa.mark_accepting(dummy_end_node)

        # If there is a next node, append it
        if self.next is not None:
            sub_dfa.append_after(self.next.convert(current_error_handlers), chain_actions=self.after_actions)
        else:
            sub_dfa.chain_actions_at_end(self.after_actions)

        return sub_dfa

class ForeachNode(ActionSinkNode, ActionSourceNode):
    def __init__(self, child_node: Node, each_actions: List[Action]):
        self.each_actions = each_actions
        self.child_node = child_node
        self.after_actions: List[Action] = []
        self.incoming_body_actions: List[Action] = []
        self.next: Node = None

        if isinstance(child_node, ActionSourceNode):
            self.incoming_body_actions, self.child_node = child_node.adopt_actions_from()

    def _adopt_actions(self, actions):
        if any(x.get_mode() != ActionMode.AT_FINISH for x in actions):
            raise IllegalASTStateError("Invalid action type following foreach", self)
            
        self.after_actions.extend(actions)

    def adopt_actions_from(self):
        actions = self.incoming_body_actions
        return actions, self

    def _set_next(self, node):
        self.next = node

    def get_next(self):
        return self.next

    def convert(self, current_error_handlers):
        if self.child_node is None:
            raise IllegalASTStateError("Empty foreach body", self)

        # Generate the DFA for the content of the foreach
        sub_dfa: DFA = self.child_node.convert(current_error_handlers)

        # Visit all transitions contained therein, ignoring ones going to error handlers,
        # and attach our each eactions to them
        # We specifically do this _before_ trying to attach after actions / the next node
        ignored_targets = set(current_error_handlers.values())
        for state in sub_dfa.states:
            for transition in state.all_transitions():
                if transition.target in ignored_targets or transition.is_fallthrough:
                    continue
                transition.attach(*self.each_actions, prepend=True)

        if self.next is not None:
            sub_dfa.append_after(self.next.convert(current_error_handlers), chain_actions=self.after_actions)
        else:
            sub_dfa.chain_actions_at_end(self.after_actions)

        return sub_dfa

class IfElseNode(ActionSinkNode, ActionSourceNode):
    def __init__(self, branches: List[DFCondition], branch_bodies: Dict[DFCondition, Node]):
        self.branches = branches 
        self.branch_bodies = branch_bodies
        self.branch_actions = {}

        self.after_actions = []
        self.next = None

        # If nonempty, we replace the if-node with a set of ConditionalActions
        self.equivalent_actions = []

        # Adopt all branch actions
        for condition in self.branches:
            if isinstance(self.branch_bodies[condition], ActionSourceNode):
                self.branch_actions[condition], self.branch_bodies[condition] = self.branch_bodies[condition].adopt_actions_from()
            else:
                self.branch_actions[condition] = []

        # Check if we can replace with an action-only node
        if all(x is None for x in self.branch_bodies.values()):
            for actions in self.branch_actions.values():
                for action in actions:
                    if not action.embeddable():
                        raise IllegalASTStateError("Unembeddable action inside action-only if/else node; place a match somewhere after the action?", action)
            self.equivalent_actions = [ConditionalAction(self.branches, self.branch_actions)]

    def _set_next(self, next_node):
        self.next = next_node

    def get_next(self):
        return self.next

    def adopt_actions_from(self):
        if self.equivalent_actions:
            return tuple(self.equivalent_actions), self.next
        else:
            return (), self

    def _adopt_actions(self, actions):
        if any(x.get_mode() != ActionMode.AT_FINISH for x in actions):
            raise IllegalASTStateError("Invalid action after if condition", self)

        self.after_actions.extend(actions)
        if self.equivalent_actions:
            self.equivalent_actions.extend(actions)

    def convert(self, current_error_handlers):
        # Generate condition dfa + start node
        dfa = DFA()
        ProgramData.imbue(dfa, DTAG.PARENT, self)
        cond_point = DFConditionPoint()
        dfa.add(cond_point)
        dfa.starting_state = cond_point

        sub_dfas = {x: y.convert(current_error_handlers) for x, y in self.branch_bodies.items() if y is not None}

        if any(y is None for y in self.branch_bodies.values()):
            dummy_target = DFState()
            dfa.add(dummy_target)
            dfa.mark_accepting(dummy_target)

        for x in self.branches:
            if x in sub_dfas:
                for i in sub_dfas[x].states:
                    dfa.add(i)
                    if i in sub_dfas[x].accepting_states:
                        dfa.mark_accepting(i)
                cond_point.transition(DFConditionalTransition(x).attach(*self.branch_actions[x]).to(sub_dfas[x].starting_state))
            else:
                cond_point.transition(DFConditionalTransition(x).attach(*self.branch_actions[x]).to(dummy_target))

        if self.next is not None:
            dfa.append_after(self.next.convert(current_error_handlers), chain_actions=self.after_actions)
        else:
            dfa.chain_actions_at_end(self.after_actions)
        return dfa

class MacroArgumentKind(enum.Enum):
    MACRO = 0
    OUT = 1
    MATCH = 2
    INTEXPR = 3
    HOOK = 4
    LOOP = 5
    FINISHCODE = 6
    YIELDCODE = 7

    EXPR = 10

class MacroArgument:
    def __init__(self, name: str, kind: MacroArgumentKind):
        self.name = name
        self.kind = kind

    def get_lookup_type(self):
        if self.kind in (MacroArgumentKind.MATCH, MacroArgumentKind.INTEXPR):
            return MacroArgumentKind.EXPR
        else:
            return self.kind

    def should_early_bind(self):
        """
        Should this argument be looked up immediately at macro entry, instead of just recording the parse tree for later use. Generally
        only valid for globally-scoped identifying arguments
        """

        return self.kind in (MacroArgumentKind.MACRO, MacroArgumentKind.OUT, MacroArgumentKind.HOOK, MacroArgumentKind.LOOP, MacroArgumentKind.FINISHCODE, MacroArgumentKind.YIELDCODE)
    
class Macro:
    def __init__(self, name_token: lark.Token, parse_tree: lark.Tree, arguments: List[MacroArgument]):
        self.name = name_token.value
        self.parse_tree = parse_tree
        self.arguments = arguments
        ProgramData.imbue(self, DTAG.SOURCE_LINE, name_token.line)
        ProgramData.imbue(self, DTAG.NAME, "macro " + self.name)

    def bind_arguments_for(self, input_trees: List[lark.Tree], parse_ctx: "ParseCtx"):
        bound_arguments = {}
        for argspec, value in zip(self.arguments, input_trees):
            allowed_types = {
                MacroArgumentKind.MACRO: ("identifier_const",),
                MacroArgumentKind.OUT: ("identifier_const",),
                MacroArgumentKind.HOOK: ("identifier_const",),
                MacroArgumentKind.LOOP: ("identifier_const",),
                MacroArgumentKind.FINISHCODE: ("identifier_const",),
                MacroArgumentKind.YIELDCODE: ("identifier_const",),
                MacroArgumentKind.MATCH: ("regex", "end_expr", "concat_expr", "string_const", "string_case_const", "binary_regex", "binary_string_const"),
                MacroArgumentKind.INTEXPR: ("string_const", "bool_const", "number_const", "char_const", "identifier_const", *all_sum_expr_nodes)
            }[argspec.kind]
            if value.data not in allowed_types:
                raise IllegalParseTree("Invalid argument type for argument " + argspec.name, value)
            if argspec.should_early_bind():
                value = parse_ctx._lookup_named_entity(argspec.kind, value.children[0])
            bound_arguments[(argspec.get_lookup_type(), argspec.name)] = value
        return bound_arguments

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
        
        self.bound_argument_stack: List[Dict[Tuple[MacroArgumentKind, str], lark.Tree]] = []

        self.yield_codes = []
        self.finish_codes = []
    
    def parse(self):
        # Parse state_object_spec
        for out in self._parse_tree.find_data("out_decl"):
            out_obj = self._parse_out_decl(out)
            if out_obj.name in self.state_object_spec:
                raise DuplicateDefinitionError("output variable", out_obj, out_obj.name)
            if out_obj.holds_a(OutputStorageType.ENUM):
                if any(x.upper() == out_obj.name.upper() for x in self.state_object_spec):
                    raise DuplicateDefinitionError("enum header name", out_obj, out_obj.name.upper())
                if out_obj.name == "finish":
                    raise IllegalParseTree("Enumerations cannot be named 'finish'", out)
            self.state_object_spec[out_obj.name] = out_obj
        # Parse macros
        for macro in self._parse_tree.find_data("macro_decl"):
            macro_obj = Macro(macro.children[0], macro.children[2:], self._parse_macro_arguments(macro.children[1]))
            if macro_obj.name in self.macros:
                raise DuplicateDefinitionError("macro", macro_obj, macro_obj.name)
            self.macros[macro_obj.name] = macro_obj

        for hook in self._parse_tree.find_data("hook_decl"):
            if hook.children[0].value in self.hooks:
                raise DuplicateDefinitionError("hook", hook.children[0], hook.children[0].value)
            self.hooks.append(hook.children[0].value)

        for code in self._parse_tree.find_data("code_decl"):
            if code.children[0].value == "yieldcode" and not ProgramData.do(ProgramFlag.YIELD_SUPPORT):
                raise IllegalParseTree("Yield support not enabled", code)

            target = {
                "yieldcode": self.yield_codes,
                "finishcode": self.finish_codes
            }[code.children[0].value]
            for i in code.children[1:]:
                val = i.value
                if val in target:
                    raise DuplicateDefinitionError("code", i, val)
                target.append(val)

        # Parse main
        parser_decl = next(self._parse_tree.find_data("parser_decl"))
        self.ast = self._parse_stmt_seq(parser_decl.children)

        if isinstance(self.ast, ActionSourceNode):
            self.start_actions, self.ast = self.ast.adopt_actions_from()

    def _lookup_named_entity(self, context: Union[MacroArgumentKind, Iterable[MacroArgumentKind]], from_tree: lark.Token):
        assert from_tree.type == "IDENTIFIER"
        name = from_tree.value

        if type(context) is not MacroArgumentKind:
            for attempt in context:
                try:
                    return self._lookup_named_entity(attempt, from_tree), attempt
                except UndefinedReferenceError:
                    continue
            raise UndefinedReferenceError(None, from_tree)

        # check if in bound argument stack
        for entry in reversed(self.bound_argument_stack):
            if (context, name) in entry:
                return entry[(context, name)]
        # otherwise, try and find globally 
        if context not in [MacroArgumentKind.MACRO, MacroArgumentKind.LOOP, MacroArgumentKind.HOOK, MacroArgumentKind.OUT, MacroArgumentKind.FINISHCODE, MacroArgumentKind.YIELDCODE]:
            raise UndefinedReferenceError("named expression", from_tree)

        if context in (MacroArgumentKind.HOOK, MacroArgumentKind.FINISHCODE, MacroArgumentKind.YIELDCODE):
            storage = {
                MacroArgumentKind.HOOK: self.hooks,
                MacroArgumentKind.FINISHCODE: self.finish_codes,
                MacroArgumentKind.YIELDCODE: self.yield_codes
            }[context]

            if name not in storage:
                raise UndefinedReferenceError({
                    MacroArgumentKind.HOOK: "hook",
                    MacroArgumentKind.YIELDCODE: "yield code",
                    MacroArgumentKind.FINISHCODE: "finish code"
                }[context], from_tree)

            return name
        else:
            storage = {
                MacroArgumentKind.MACRO: self.macros,
                MacroArgumentKind.LOOP: self.break_handlers,
                MacroArgumentKind.OUT: self.state_object_spec,
            }[context]

            if name not in storage:
                raise UndefinedReferenceError({
                    MacroArgumentKind.MACRO: "macro",
                    MacroArgumentKind.OUT: "variable",
                    MacroArgumentKind.LOOP: "break target"
                }, from_tree)

            return storage[name]

    def _parse_macro_arguments(self, args: lark.Tree):
        if args.data == "macro_arg_empty":
            return []
        defined = set()
        parsed_args = []
        for i in args.children:
            if i.data == "macro_rescode_arg":
                name = i.children[1].value
                kind = {"yieldcode": MacroArgumentKind.YIELDCODE, "finishcode": MacroArgumentKind.FINISHCODE}[i.children[0].value]
            else:
                name = i.children[0].value
                kind = {
                    "macro_macro_arg": MacroArgumentKind.MACRO,
                    "macro_out_arg": MacroArgumentKind.OUT,
                    "macro_match_expr_arg": MacroArgumentKind.MATCH,
                    "macro_int_expr_arg": MacroArgumentKind.INTEXPR,
                    "macro_hook_arg": MacroArgumentKind.HOOK,
                    "macro_breaktgt_arg": MacroArgumentKind.LOOP,
                }[i.data]
            if name in defined:
                raise DuplicateDefinitionError("macro argument", i, name)
            parsed_args.append(MacroArgument(name, kind))
            defined.add(name)
        return parsed_args

    def _convert_char_const(self, char_const: str):
        if len(char_const) == 3:
            return char_const[1]
        else:
            return {
                'n': '\n',
                'r': '\r',
                't': '\t',
                'b': '\b',
                '\'': '\''
            }.get(char_const[2], char_const[2])

    def _convert_int(self, text: str):
        sign = 1
        if text[0] == "+":
            text = text[1:]
        elif text[0] == "-":
            sign = -1
            text = text[1:]

        if text[0:2] == "0x":
            return sign * int(text[2:], base=16)
        elif text[0:2] == "0b":
            return sign * int(text[2:], base=2)
        else:
            return sign * int(text)

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
                    if contents[i] == "u":
                        raise NotImplementedError("don't support uescapes yet")
                    code = contents[i+1:i+3]
                    result += chr(int(code, base=16))
                    i += 3
                else:
                    result += {
                        'n': '\n',
                        'r': '\r',
                        't': '\t',
                        'b': '\b',
                        '0': '\x00',
                        '"': '"',
                        '\\': '\\'
                    }[contents[i]]
                    i += 1
        return result

    def _convert_binary_string(self, binary_string: str):
        """
        Parse the hex binary string (with quotes too)
        """

        contents = [x for x in binary_string[1:-1] if x in string.hexdigits]
        
        if len(contents) % 2 != 0:
            raise ValueError("binary literal must have even number of characters")

        result = ""

        for word in zip(contents[::2], contents[1::2]):
            result += chr(int(word[0] + word[1], base=16))

        return result

    def _parse_out_decl(self, decl: lark.Tree) -> OutputStorage:
        """
        Parse an output declaration
        """

        type_obj = decl.children[0]
        name = decl.children[1].value
        if len(decl.children) == 3:
            if type_obj.data not in ["str_type", "unterm_str_type"]:
                default_value = self._parse_integer_expr(decl.children[2])
                if not default_value.is_literal():
                    raise IllegalParseTree("Default value for out-decl must be constant", decl.children[2])
            elif type_obj.data == "raw_type":
                raise IllegalParseTree("Default values are not allowed for raw types, set them manually if necessary.", decl.children[2])
            else:
                if decl.children[2].data == "string_const":
                    default_value = self._convert_string(decl.children[2].children[0].value)
                elif decl.children[2].data == "binary_string_const":
                    try:
                        default_value = self._convert_binary_string(decl.children[2].children[0].value).encode('latin-1')
                    except ValueError as e:
                        raise IllegalParseTree(e.args[0], decl.children[2].children[0])
                else:
                    raise IllegalParseTree("Default value for string must be a string constant", decl.children[2])
        else:
            default_value = None

        if type_obj.data == "bool_type":
            return OutputStorage(OutputStorageType.BOOL, name, default_value=default_value)
        elif type_obj.data == "int_type":
            kwargs = {}
            for attr in type_obj.children:
                if attr.data == "signed_attr":
                    kwargs["int_signed"] = attr.children[0].value == "signed"
                elif attr.data == "width_attr":
                    kwargs["int_width"] = int(attr.children[0].value)
                else:
                    raise NotImplementedError(attr)
            return OutputStorage(OutputStorageType.INT, name, default_value=default_value, **kwargs)
        elif type_obj.data == "enum_type":
            return OutputStorage(OutputStorageType.ENUM, name, default_value=default_value, enum_values=list(x.value for
                x in type_obj.children))
        elif type_obj.data == "str_type":
            return OutputStorage(OutputStorageType.STR, name, default_value=default_value, str_size=self._convert_int(type_obj.children[0].value))
        elif type_obj.data == "unterm_str_type":
            return OutputStorage(OutputStorageType.STR, name, default_value=default_value, str_size=self._convert_int(type_obj.children[0].value), str_null=False)
        elif type_obj.data == "raw_type":
            return OutputStorage(OutputStorageType.RAW, name, raw_underlying=type_obj.children[0].value)
        else:
            raise NotImplementedError(type_obj.data)

    def _parse_math_expr(self, expr: lark.Tree, into_storage: OutputStorage=None):
        """
        Parse a math expr [(something)]
        """

        if expr.data == "math_num":
            return ProgramData.imbue(ProgramData.imbue(LiteralIntegerExpr(self._convert_int(expr.children[0].value)), DTAG.SOURCE_LINE, expr.meta.line), DTAG.SOURCE_COLUMN, expr.meta.column)
        elif expr.data == "math_char_const":
            return ProgramData.imbue(ProgramData.imbue(LiteralIntegerExpr(ord(self._convert_char_const(expr.children[0].value))), DTAG.SOURCE_LINE, expr.meta.line), DTAG.SOURCE_COLUMN, expr.meta.column)
        elif expr.data == "math_var":
            # Try to handle enums too
            if into_storage is not None and into_storage.type == OutputStorageType.ENUM and expr.children[0].value in into_storage.enum_values:
                val = LiteralIntegerExpr(expr.children[0].value, OutputStorageType.ENUM, model_ref=into_storage)
                ProgramData.imbue(val, DTAG.SOURCE_LINE, expr.children[0].line)
                ProgramData.imbue(val, DTAG.SOURCE_COLUMN, expr.children[0].column)
                return val
            out_spec, kind = self._lookup_named_entity((MacroArgumentKind.EXPR, MacroArgumentKind.OUT), expr.children[0])
            if kind == MacroArgumentKind.EXPR:
                return self._parse_integer_expr(out_spec, into_storage=into_storage)
            return ProgramData.imbue(ProgramData.imbue(OutIntegerExpr(out_spec), DTAG.SOURCE_LINE, expr.meta.line), DTAG.SOURCE_COLUMN, expr.meta.column)
        elif expr.data in ["math_str_len", "math_str_index"]:
            out_spec = self._lookup_named_entity(MacroArgumentKind.OUT, expr.children[0])

            if not out_spec.holds_buflike():
                raise IllegalParseTree("Variable must be string", expr.children[0])

            if expr.data == "math_str_len":
                val = StringLengthIntegerExpr(out_spec)
            else:
                val = StringRefIntegerExpr(out_spec, self._parse_integer_expr(expr.children[1]))
            
            return ProgramData.imbue(val,
                    DTAG.SOURCE_LINE, expr.meta.line,
                    DTAG.SOURCE_COLUMN, expr.meta.column
            )
        elif expr.data == "builtin_math_var":
            ref = expr.children[0]
            if ref.value == "last":
                return ProgramData.imbue(ProgramData.imbue(LastCharIntegerExpr(), DTAG.SOURCE_LINE, expr.meta.line), DTAG.SOURCE_COLUMN, expr.meta.column)
            else:
                raise UndefinedReferenceError("builtin math variable", ref)
        elif expr.data == "sum_expr":
            return SumIntegerExpr([self._parse_integer_expr(x, into_storage=into_storage) for x in expr.children[::2]], [False, *(x.value == "-" for x in expr.children[1::2])])
        elif expr.data == "mul_expr":
            return MulIntegerExpr([self._parse_integer_expr(x, into_storage=into_storage) for x in expr.children[::2]], [MulIntegerExprOp.MUL, *(MulIntegerExprOp(x.value) for x in expr.children[1::2])])
        elif expr.data == "comp_expr":
            left = self._parse_integer_expr(expr.children[0])
            if isinstance(left, OutIntegerExpr):
                ref = left.ref
            else:
                ref = None
            return CompareIntegerExpr(left, self._parse_integer_expr(expr.children[2], into_storage=ref), CompareIntegerExprOp(expr.children[1].value))
        elif expr.data == "shift_expr":
            return BitShiftIntegerExpr(self._parse_integer_expr(expr.children[0], into_storage=into_storage), self._parse_integer_expr(expr.children[2], into_storage=into_storage), expr.children[1].value == "<<")
        elif expr.data in ["bit_or_expr", "bit_xor_expr", "bit_and_expr"]:
            op = {
                "bit_or_expr": BitwiseIntegerExprOp.OR,
                "bit_xor_expr": BitwiseIntegerExprOp.XOR,
                "bit_and_expr": BitwiseIntegerExprOp.AND,
            }[expr.data]
            return BitwiseIntegerExpr([self._parse_integer_expr(x, into_storage=into_storage) for x in expr.children], op)
        elif expr.data == "conjunction_expr":
            return ConjunctionIntegerExpr([self._parse_integer_expr(x, into_storage=into_storage) for x in expr.children])
        elif expr.data == "disjunction_expr":
            return DisjunctionIntegerExpr([self._parse_integer_expr(x, into_storage=into_storage) for x in expr.children])
        elif expr.data == "not_expr":
            return CompareIntegerExpr(self._parse_integer_expr(expr.children[0], into_storage=into_storage), LiteralIntegerExpr(False, OutputStorageType.BOOL), CompareIntegerExprOp.EQ)
        elif expr.data == "negate_expr":
            return SumIntegerExpr([LiteralIntegerExpr(0), self._parse_integer_expr(expr.children[0], into_storage=into_storage)], [False, True])
        else:
            raise IllegalParseTree("Invalid math expression", expr)

    def _parse_integer_expr(self, expr: lark.Tree, into_storage: OutputStorage=None) -> IntegerExpr:
        """
        Parse an integer type expr (also has bool/etc.)
        """

        BANNED_TYPES = ["end_expr", "concat_expr", "regex", "string_const", "string_case_const", "binary_regex", "binary_string_const"]
        if expr.data in BANNED_TYPES:
            raise IllegalParseTree("String-typed value encountered for integer-typed expression", expr)

        if expr.data == "number_const":
            val = LiteralIntegerExpr(self._convert_int(expr.children[0].value))
            ProgramData.imbue(val, DTAG.SOURCE_LINE, expr.children[0].line)
            ProgramData.imbue(val, DTAG.SOURCE_COLUMN, expr.children[0].column)
            return val
        elif expr.data == "char_const":
            val = LiteralIntegerExpr(ord(self._convert_char_const(expr.children[0].value)))
            ProgramData.imbue(val, DTAG.SOURCE_LINE, expr.children[0].line)
            ProgramData.imbue(val, DTAG.SOURCE_COLUMN, expr.children[0].column)
            return val
        elif expr.data == "identifier_const":
            try:
                expr = self._lookup_named_entity(MacroArgumentKind.EXPR, expr.children[0])
                return self._parse_integer_expr(expr, into_storage=into_storage)
            except UndefinedReferenceError:
                pass

            if into_storage is None:
                raise IllegalParseTree("Undefined enumeration value, no into_storage", expr)

            if into_storage.type != OutputStorageType.ENUM:
                raise IllegalParseTree("Use of enumeration constant for non-enumeration type output (perhaps you meant to use []?)", expr)

            if expr.children[0].value not in into_storage.enum_values:
                raise UndefinedReferenceError("enumeration constant", expr)
            
            val = LiteralIntegerExpr(expr.children[0].value, OutputStorageType.ENUM, model_ref=into_storage)
            ProgramData.imbue(val, DTAG.SOURCE_LINE, expr.children[0].line)
            ProgramData.imbue(val, DTAG.SOURCE_COLUMN, expr.children[0].column)
            return val
        elif expr.data == "bool_const":
            if into_storage is None:
                result_type = OutputStorageType.BOOL
            else:
                if into_storage.type not in [OutputStorageType.BOOL, OutputStorageType.INT]:
                    raise IllegalParseTree("Use of boolean expression in non integral-type", expr)
                result_type = into_storage.type

            try:
                return ProgramData.imbue(ProgramData.imbue(LiteralIntegerExpr({"true": 1, "false": 0}[expr.children[0].value], result_type), DTAG.SOURCE_LINE, expr.meta.line), DTAG.SOURCE_COLUMN, expr.meta.column)
            except KeyError as e:
                raise UndefinedReferenceError("boolean constant", expr.children[0]) from e
        elif expr.data in all_sum_expr_nodes:
            return self._parse_math_expr(expr, into_storage)
        else:
            raise IllegalParseTree("Invalid expression in integer expr", expr)


    def _parse_match_expr(self, expr: lark.Tree) -> Match:
        """
        Parse a match expression into a match object
        """
        if expr.data in ["string_const", "binary_string_const"]:
            actual_content = expr.children[0]
            if expr.data == "string_const":
                match = DirectMatch(self._convert_string(actual_content.value))
            else:
                try:
                    match = DirectMatch(self._convert_binary_string(actual_content.value))
                except ValueError as e:
                    raise IllegalParseTree(e.args[0], actual_content)
                ProgramData.imbue(match, DTAG.NAME, f"binary match {expr.children[0].value}")
            ProgramData.imbue(match, DTAG.SOURCE_LINE, actual_content.line)
            ProgramData.imbue(match, DTAG.SOURCE_COLUMN, actual_content.column)
            return match
        elif expr.data == "string_case_const":
            actual_content = expr.children[0]
            match = CaseDirectMatch(self._convert_string(actual_content.value))
            ProgramData.imbue(match, DTAG.SOURCE_LINE, actual_content.line)
            ProgramData.imbue(match, DTAG.SOURCE_COLUMN, actual_content.column)
            return match
        elif expr.data == "regex":
            match = RegexMatch(expr)
            ProgramData.imbue(match, DTAG.SOURCE_LINE, expr.meta.line)
            ProgramData.imbue(match, DTAG.SOURCE_COLUMN, expr.meta.column)
            return match
        elif expr.data == "binary_regex":
            match = BinaryRegexMatch(expr)
            ProgramData.imbue(match, DTAG.SOURCE_LINE, expr.meta.line)
            ProgramData.imbue(match, DTAG.SOURCE_COLUMN, expr.meta.column)
            return match
        elif expr.data == "end_expr":
            if not ProgramData.do(ProgramFlag.EOF_SUPPORT):
                raise IllegalParseTree("end match but EOF support is not enabled", expr)
            match = EndMatch()
            ProgramData.imbue(match, DTAG.SOURCE_LINE, expr.meta.line)
            ProgramData.imbue(match, DTAG.SOURCE_COLUMN, expr.meta.column)
            return match
        elif expr.data == "concat_expr":
            return ConcatMatch(list(self._parse_match_expr(x) for x in expr.children))
        elif expr.data == "identifier_const":
            return self._parse_match_expr(self._lookup_named_entity(MacroArgumentKind.EXPR, expr.children[0]))
        else:
            raise IllegalParseTree("Invalid expression in match expression context", expr)

    def _parse_assign_stmt(self, stmt: lark.Tree, is_append) -> Node:
        """
        Parse an assignment into its underlying action
        """

        # Resolve the target
        targeted = self._lookup_named_entity(MacroArgumentKind.OUT, stmt.children[0])

        if targeted.holds_buflike() and is_append:
            # Handle arguments
            if stmt.children[1].data == "identifier_const":
                sub_expr = self._lookup_named_entity(MacroArgumentKind.EXPR, stmt.children[1].children[0])
            else:
                sub_expr = stmt.children[1]
            # Check if this is a math expression (only valid append type other than match)
            if sub_expr.data in all_sum_expr_nodes:
                # Create an AppendCharTo action
                return ActionNode(AppendCharTo(self.exception_handlers[ErrorReasons.OUT_OF_SPACE], self._parse_math_expr(sub_expr), targeted))
            # Create an append expression
            match_node = MatchNode(self._parse_match_expr(sub_expr))
            match_node.match.attach(AppendTo(self.exception_handlers[ErrorReasons.OUT_OF_SPACE], targeted))
            return match_node
        elif targeted.type == OutputStorageType.STR:
            # Handle arguments
            if stmt.children[1].data == "identifier_const":
                sub_expr = self._lookup_named_entity(MacroArgumentKind.EXPR, stmt.children[1].children[0])
            else:
                sub_expr = stmt.children[1]

            if sub_expr.data != "string_const":
                raise IllegalParseTree("String assignment only supports string constants, did you mean +=?", sub_expr)
            result = self._convert_string(sub_expr.children[0].value)
            if len(result) == 0 and ProgramData.do(ProgramFlag.USE_DELETE_FOR_EMPTY_STRING):
                return ActionNode(DeleteBuf(targeted))
            return ActionNode(SetToStr(result, targeted))
        elif not is_append:
            if targeted.type == OutputStorageType.RAW:
                raise IllegalParseTree("Raw types only support append expressions, did you mean +=?", sub_expr)
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

    def _parse_macro_call(self, lark_node_for_error: lark.Tree, macro: Macro, arguments: List[lark.Tree]):
        if len(arguments) != len(macro.arguments):
            raise IllegalParseTree("Incorrect number of arguments", lark_node_for_error)
        self.bound_argument_stack.append(
            macro.bind_arguments_for(arguments, self)
        )
        node = ProgramData.imbue(self._parse_stmt_seq(macro.parse_tree), DTAG.PARENT, macro)
        del self.bound_argument_stack[-1]
        return node

    def _parse_stmt(self, stmt: lark.Tree) -> Node:
        """
        Parse a statement into a node
        """
        if stmt.data == "match_stmt":
            return MatchNode(self._parse_match_expr(stmt.children[0]))
        elif stmt.data == "wait_stmt":
            return MatchNode(WaitMatch(self._parse_match_expr(stmt.children[0])))
        elif stmt.data == "call_stmt":
            referenced, kind = self._lookup_named_entity((MacroArgumentKind.MACRO, MacroArgumentKind.HOOK), stmt.children[0])
            if kind == MacroArgumentKind.MACRO:
                return self._parse_macro_call(stmt, referenced, stmt.children[1:])
            else:
                return ActionNode(ProgramData.imbue(ProgramData.imbue(CallHook(referenced), DTAG.SOURCE_LINE, stmt.meta.line), DTAG.SOURCE_COLUMN, stmt.meta.column))
        elif stmt.data == "finish_stmt":
            act = FinishAction()
            ProgramData.imbue(act, DTAG.SOURCE_LINE, stmt.meta.line)
            ProgramData.imbue(act, DTAG.SOURCE_COLUMN, stmt.meta.column)
            return ActionNode(act)
        elif stmt.data == "custom_finish_stmt":
            act = CustomFinishAction(self._lookup_named_entity(MacroArgumentKind.FINISHCODE, stmt.children[0]))
            ProgramData.imbue(act, DTAG.SOURCE_LINE, stmt.meta.line)
            ProgramData.imbue(act, DTAG.SOURCE_COLUMN, stmt.meta.column)
            return ActionNode(act)
        elif stmt.data == "custom_yield_stmt":
            if not ProgramData.do(ProgramFlag.YIELD_SUPPORT):
                raise IllegalParseTree("Yield support not enabled", stmt)
            act = CustomYieldAction(self._lookup_named_entity(MacroArgumentKind.YIELDCODE, stmt.children[0]))
            ProgramData.imbue(act, DTAG.SOURCE_LINE, stmt.meta.line)
            ProgramData.imbue(act, DTAG.SOURCE_COLUMN, stmt.meta.column)
            return InterruptableActionNode(act)
        elif stmt.data == "break_stmt":
            if stmt.children:
                act = self._lookup_named_entity(MacroArgumentKind.LOOP, stmt.children[0])
            else:
                act = self.innermost_break_handler
            if act is None:
                raise IllegalParseTree("Break outside of loop", stmt)
            ProgramData.imbue(act, DTAG.SOURCE_LINE, stmt.meta.line)
            ProgramData.imbue(act, DTAG.SOURCE_COLUMN, stmt.meta.column)
            return ActionNode(act)
        elif stmt.data == "delete_stmt":
            targeted = self._lookup_named_entity(MacroArgumentKind.OUT, stmt.children[0])
            
            act = DeleteBuf(targeted)
            return ProgramData.imbue(ActionNode(act),
                DTAG.SOURCE_LINE, stmt.meta.line,
                DTAG.SOURCE_COLUMN, stmt.meta.column
            )
        elif stmt.data in ("assign_stmt", "append_stmt"):
            return ProgramData.imbue(self._parse_assign_stmt(stmt, stmt.data == "append_stmt"), DTAG.SOURCE_LINE, stmt.meta.line, DTAG.SOURCE_COLUMN, stmt.meta.column)
        elif stmt.data == "case_stmt":
            # Find all of the matches
            return ProgramData.imbue(ProgramData.imbue(CaseNode({k: v for k, v in (self._parse_case_clause(x) for x in stmt.children)}), 
                DTAG.SOURCE_LINE, stmt.meta.line),
                DTAG.SOURCE_COLUMN, stmt.meta.column
            )
        elif stmt.data == "greedy_case_stmt":
            case_blocks = {}
            priorities = {}
            for block in stmt.children:
                if block.data == "case_clause":
                    k, v = self._parse_case_clause(block)
                    case_blocks[k] = v
                else:
                    for clause in block.children[1:]:
                        k, v = self._parse_case_clause(clause)
                        case_blocks[k] = v
                        priorities[k] = int(block.children[0].value)

            return ProgramData.imbue(ProgramData.imbue(CaseNode(case_blocks, greedy=True, priorities=priorities), 
                DTAG.SOURCE_LINE, stmt.meta.line),
                DTAG.SOURCE_COLUMN, stmt.meta.column
            )

        elif stmt.data == "if_stmt":
            conditions_ordered = []
            condition_map = {}
            for condition_tree in stmt.children:
                if condition_tree.data == "else_condition":
                    condition = ElseCondition()
                    condition_map[condition] = self._parse_stmt_seq(condition_tree.children)
                else:
                    condition = IntegerCondition(self._parse_integer_expr(condition_tree.children[0]))
                    condition_map[condition] = self._parse_stmt_seq(condition_tree.children[1:])

                ProgramData.imbue(condition,
                    DTAG.SOURCE_LINE, condition_tree.meta.line,
                    DTAG.SOURCE_COLUMN, condition_tree.meta.column
                )
                conditions_ordered.append(condition)
            if not isinstance(conditions_ordered[-1], ElseCondition):
                conditions_ordered.append(ElseCondition())
                condition_map[conditions_ordered[-1]] = None
            return ProgramData.imbue(
                IfElseNode(conditions_ordered, condition_map),
                DTAG.SOURCE_LINE, stmt.meta.line,
                DTAG.SOURCE_COLUMN, stmt.meta.column
            )
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
        elif stmt.data == "foreach_stmt":
            action_block = stmt.children[-1]
            contents     = stmt.children[:-1]

            content_node = self._parse_stmt_seq(contents)
            action_node  = self._parse_stmt_seq(action_block.children)

            if not isinstance(action_node, ActionSourceNode):
                raise IllegalASTStateError("Invalid statement in foreach actions; must only be an action source", action_node)

            each_actions, action_node = action_node.adopt_actions_from()
            if action_node is not None:
                raise IllegalASTStateError("Invalid statement in foreach actions; must only be action source with no DFA", action_node)

            return ForeachNode(content_node, each_actions)
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
        self.finish_codes = parse_ctx.finish_codes
        self.yield_codes = parse_ctx.yield_codes
        self.ast = parse_ctx.ast
        self.start_actions = parse_ctx.start_actions
        self.generic_fail_state = parse_ctx.generic_fail_state
        self.dfa = None

    def _optimize_remove_inaccessible(self):
        if not ProgramData.do(ProgramFlag.REMOVE_INACCESIBLE_STATES):
            return 0
        accessible = set(self.dfa.dfs())
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

    def _optimize_shortcircuit_fallthroughs(self):
        """
        Simplify fallthrough transitions such as:

        a -=-=-> b -<h>-> c

        to just

        a --<h>-->c

        by following all transitions pointing at fallthrough transitions. We also impose a limit such that if a transition to be shortcircuited is referred to by enough
        other states (and has at least one action) we won't replace it.

        Currently, we don't handle action side effect since there's no good way to retarget those right now.
        """

        if not ProgramData.do(ProgramFlag.SHORTCIRCUIT_FALLTHROUGHS):
            return 0

        # Count all transitions
        ignore_map_counter = Counter()
        for transition in self.dfa.all_transitions():
            ignore_map_counter[(frozenset(transition.on_values), transition.target)] += 1

        mod = 0

        all_transitions = list(self.dfa.all_transitions(include_states=True))

        for orig_state, transition in all_transitions:
            if not transition.is_fallthrough:
                continue

            # Check if the target has a matching fallthrough
            if isinstance(transition.target, DFProxyState) and not transition.target.can_eliminate() or isinstance(orig_state, DFProxyState) and not orig_state.can_eliminate(): continue

            effective = set(transition.on_values)
            if DFTransition.Else in transition.on_values:
                effective.update(transition.target.compute_foreign_else_definition(orig_state))

            next_target = transition.target[effective]

            if next_target is None or next_target.is_fallthrough:
                continue

            # Are there actions? If so, does this violate the threshold
            if len(next_target.actions) > 0:
                max_count = ProgramData.option(ProgramOption.MAX_SHORTCIRCUIT_FALLTHROUGH) - ProgramData.option(ProgramOption.MAX_SHORTCIRCUIT_ACTION_PENALTY)*(len(next_target.actions)-1)
                if ignore_map_counter[(frozenset(next_target.on_values), next_target.target)] > max_count:
                    continue

            # Shortcircuit the transition
            if next_target.error_handling:
                transition.handles_else()

            transition.attach(*next_target.actions)
            transition.to(next_target.target)
            transition.fallthrough(False)

            mod += 1

        dprint[ProgramFlag.VERBOSE_OPTIMIZE_RESULTS]("shortcircuited {} transitions".format(mod))

        mod2 = 0

        # Now, we'll try to remove "dummy states" -- ones that _only_ have an Else fallthrough on them.
        all_transitions = list(self.dfa.all_transitions(include_states=True))
        for orig_state, transition in all_transitions:
            if isinstance(transition.target, DFProxyState) and not transition.target.can_eliminate() or isinstance(orig_state, DFProxyState) and not orig_state.can_eliminate(): continue

            if len(transition.target.transitions) != 1 or DFTransition.Else not in transition.target.transitions[0].on_values:
                continue

            to_replace = transition.target.transitions[0]

            if not to_replace.is_fallthrough:
                continue

            if len(to_replace.actions) > 0:
                max_count = ProgramData.option(ProgramOption.MAX_SHORTCIRCUIT_FALLTHROUGH) - ProgramData.option(ProgramOption.MAX_SHORTCIRCUIT_ACTION_PENALTY)*(len(to_replace.actions)-1)
                if ignore_map_counter[(frozenset(to_replace.on_values), to_replace.target)] > max_count:
                    continue

            # Shortcircuit the transition
            if to_replace.error_handling:
                transition.handles_else()

            transition.attach(*to_replace.actions)
            transition.to(to_replace.target)

            mod2 += 1

        dprint[ProgramFlag.VERBOSE_OPTIMIZE_RESULTS]("removed {} dummy transitions".format(mod2))
        return mod + mod2

    def _verify_fallthrough_loop(self):
        """
        Check if any transitions pointing to their own state are fallthrough, which is invalid
        since it would cause an infinite loop
        """

        for state in self.dfa.states:
            for transition in state.transitions:
                if not transition.is_fallthrough:
                    continue

                visited = set()
                def consider(transition):
                    return not any(x.get_target_override_mode() in [ActionOverrideMode.ALWAYS_GOTO_OTHER, ActionOverrideMode.ALWAYS_GOTO_UNDEFINED] and transition.target not in x.get_target_override_targets() for x in transition.actions)

                def aux(x):
                    if isinstance(x, DFConditionPoint):
                        for i in x.transitions:
                            if i.target in visited:
                                continue
                            if consider(i):
                                visited.add(i.target)
                                aux(i.target)
                    else:
                        real_target = x[transition.on_values]
                        if real_target and real_target.is_fallthrough and consider(real_target):
                            if real_target.target not in visited:
                                visited.add(real_target.target)
                                aux(real_target.target)
                
                aux(state)

                if state in visited:
                    raise IllegalDFAStateError("Infinite loop due to self-referential fallthrough", transition)
        

    def compile(self):
        """
        Convert the AST into a (potentially optimized) DFA.
        """

        self.dfa = self.ast.convert(defaultdict(lambda: self.generic_fail_state))
        self.dfa.add(self.generic_fail_state)

        while self._optimize_remove_inaccessible() + self._optimize_simplify_transition_matches() + self._optimize_shortcircuit_fallthroughs():
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
        self.finish_codes = dfa_compile_ctx.finish_codes
        self.yield_codes = dfa_compile_ctx.yield_codes
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
            for fc in self.finish_codes:
                enum_content.add(f"{self.program_name.upper()}_FINISH_{fc},")
            for yc in self.yield_codes:
                enum_content.add(f"{self.program_name.upper()}_YIELD_{yc},")

        result.add("};")
        result.add(f"typedef enum {self.program_name}_result {self.program_name}_result_t;")
        result.add()

        result.add(f"{self.program_name}_result_t {self.program_name}_start({self.program_name}_state_t *state);")
        start_typename = "const uint8_t *" if not ProgramData.do(ProgramFlag.INDIRECT_START_PTR) else "const uint8_t **";
        result.add(f"{self.program_name}_result_t {self.program_name}_feed({start_typename}start, const uint8_t *end, {self.program_name}_state_t *state);")
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

    def _integer_containing(self, maxval=None, signed=True, width=None):
        if width is not None:
            maxval = {
                1: 127,
                2: 32767,
                4: 2147483647,
                8: (1 << 63)-1
            }.get(width, None)
            if not signed:
                maxval += 1
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

    def _get_maxval_hint_for_raw_type(self, typename: str):
        """
        Guess the size of an arbitrary c type.
        """

        return {
            "int8_t": 1,
            "uint8_t": 1,
            "int16_t": 2,
            "uint16_t": 2,
            "int32_t": 4,
            "uint32_t": 4,
            "int64_t": 8,
            "uint64_t": 8,
            "float": 4,
            "double": 8
        }.get(typename, None)

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

            if any(x.holds_buflike() for x in self.state_object_spec):
                for out_str in (x for x in self.state_object_spec if x.type == OutputStorageType.STR):
                    contents.add(self._integer_containing(out_str.str_size, signed=False), f"{out_str.name}_counter;")
                for out_raw in (x for x in self.state_object_spec if x.type == OutputStorageType.RAW):
                    contents.add(self._integer_containing(self._get_maxval_hint_for_raw_type(out_raw.raw_underlying), signed=False), f"{out_raw.name}_counter;")

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

    def _get_string_char_type(self):
        return "uint8_t" if ProgramData.do(ProgramFlag.STRINGS_AS_U8) else "char"

    def _get_state_object_out_declaration(self, out_decl: OutputStorage):
        """
        Get the type of a state object out decl
        """
        
        if out_decl.type != OutputStorageType.STR:
            return {
                OutputStorageType.INT: self._integer_containing(signed=out_decl.int_signed, width=out_decl.int_width),
                OutputStorageType.ENUM: f"{self.program_name}_out_{out_decl.name}_t",
                OutputStorageType.BOOL: "bool",
                OutputStorageType.RAW: out_decl.raw_underlying
            }[out_decl.type] + " " + out_decl.name
        else:
            typ = self._get_string_char_type()
            if not self._is_dynamic(out_decl):
                return f"{typ} {out_decl.name}[{out_decl.str_size}]"
            else:
                return f"{typ} * {out_decl.name}"

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

    def _convert_literal_value(self, literal: LiteralIntegerExpr):
        if literal.result_type() == OutputStorageType.ENUM:
            return f"{self.program_name.upper()}_{literal.model_ref.name.upper()}_{literal.get_literal_result().upper()}"
        elif literal.result_type() == OutputStorageType.BOOL:
            return "true" if literal.get_literal_result() else "false"
        elif literal.result_type() == OutputStorageType.INT:
            return str(literal.get_literal_result())
        elif literal.result_type() == OutputStorageType.STR:
            return '"{}"'.format(self._escape_string(literal.get_literal_result()))
        else:
            raise NotImplementedError(literal.result_type())

    def _generate_buflike_index_expr(self, out_expr: OutputStorage, index_expr: str):
        if out_expr.holds_a(OutputStorageType.RAW):
            return f"((uint8_t *)state->c.{out_expr.name})[{index_expr}]"
        else:
            return f"state->c.{out_expr.name}[{index_expr}]"

    def _generate_buflike_length_expr(self, out_expr: OutputStorage, include_null=False):
        if out_expr.holds_a(OutputStorageType.RAW):
            return f"sizeof(state->c.{out_expr.name})"
        else:
            if include_null:
                return str(out_expr.effective_string_size())
            else:
                return str(out_expr.str_size)

    def _generate_code_for_int_expr(self, intexpr: IntegerExpr, ctx: IntegerExprUseContext, out_expr: OutputStorage=None):
        """
        Generate C expression that evaluates to the value of intexpr, used in context ctx and which will fill out_expr (or have its type)
        """

        if ctx in intexpr.get_invalid_contexts():
            raise IllegalIntExpr("Illegal use of integer expression in context {}".format(ctx.name), intexpr)

        if out_expr is not None:
            if intexpr.result_type() != out_expr.type:
                raise IllegalIntExpr("Mismatched types for target storage", intexpr)

        if isinstance(intexpr, LiteralIntegerExpr):
            return self._convert_literal_value(intexpr)
        elif isinstance(intexpr, OutIntegerExpr):
            return f"state->c.{intexpr.ref.name}"
        elif isinstance(intexpr, StringLengthIntegerExpr):
            return f"state->{intexpr.ref.name}_counter";
        elif isinstance(intexpr, StringRefIntegerExpr):
            index = self._generate_code_for_int_expr(intexpr.index, ctx)
            text = self._generate_buflike_index_expr(intexpr.ref, index)
            size_str = self._generate_buflike_length_expr(intexpr.ref)
            if ProgramData.do(ProgramFlag.UNSAFE_STRING_INDEXING):
                return text
            return f"((({index}) >= 0 && ({index}) < {size_str}) ? {text} : 0)"
        elif isinstance(intexpr, LastCharIntegerExpr):
            return f"(inval)" # name of the last character value
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
        elif isinstance(intexpr, CompareIntegerExpr):
            return f"({self._generate_code_for_int_expr(intexpr.left, ctx, out_expr)}) {intexpr.op.value} ({self._generate_code_for_int_expr(intexpr.right, ctx, out_expr)})"
        elif isinstance(intexpr, (DisjunctionIntegerExpr, ConjunctionIntegerExpr)):
            result = f"({self._generate_code_for_int_expr(intexpr.children[0], ctx, out_expr)})"
            for child in intexpr.children[1:]:
                result += " "
                result += "||" if isinstance(intexpr, DisjunctionIntegerExpr) else "&&"
                result += " "
                result += f"({self._generate_code_for_int_expr(child, ctx, out_expr)})"
            return result
        elif isinstance(intexpr, BitwiseIntegerExpr):
            result = f"({self._generate_code_for_int_expr(intexpr.children[0], ctx, out_expr)})"
            for child in intexpr.children[1:]:
                result += " "
                result += intexpr.op.value
                result += " "
                result += f"({self._generate_code_for_int_expr(child, ctx, out_expr)})"
            return result
        elif isinstance(intexpr, BitShiftIntegerExpr):
            return f"({self._generate_code_for_int_expr(intexpr.left, ctx, out_expr)}) {'<<' if intexpr.towards_left else '>>'} ({self._generate_code_for_int_expr(intexpr.right, ctx, out_expr)})"
        else:
            raise NotImplementedError("unsupported intexpr type", intexpr)

    def _escape_string(self, value: Union[bytes, str]):
        result = ""
        if type(value) is str:
            bytes_value = value.encode('utf-8')
        else:
            bytes_value = value
        for i in bytes_value:
            if chr(i) in ["\\", '"']:
                result += "\\" + chr(i)
            elif not (32 <= i < 127):
                result += "\\x{:02x}".format(i)
            else:
                result += chr(i)
        return result

    def _generate_set_string(self, value: Union[bytes, str], into: OutputStorage):
        """
        Generate code that sets value into into

        : strncpy(state->c.into, "value", into.str_size)

        Must ensure value is short enough first.
        """

        if isinstance(value, str):
            escaped_length = len(value.encode('utf-8'))
        else:
            escaped_length = len(value)

        return f"memcpy(state->c.{into.name}, \"{self._escape_string(value)}\", {escaped_length if not into.str_null else escaped_length+1});"

    def _generate_action_implementation(self, action: Action, is_start: bool = False, is_end: bool = False, transition=None):
        result = Outputter()
        ctx = IntegerExprUseContext.ASSIGN_ON_MATCH
        if is_start:
            ctx = IntegerExprUseContext.ASSIGN_INITIAL
        elif is_end:
            ctx = IntegerExprUseContext.ASSIGN_ON_END
        if isinstance(action, CustomFinishAction):
            result.add(f"return {self.program_name.upper()}_FINISH_{action.result_code};")
        elif isinstance(action, FinishAction):
            result.add(f"return {self.program_name.upper()}_DONE;")
        elif isinstance(action, CustomYieldAction):
            if not ProgramData.do(ProgramFlag.YIELD_SUPPORT):
                raise IllegalDFAStateError("Yield support not enabled", action)
            result.add(f"return {self.program_name.upper()}_YIELD_{action.result_code};")
        elif isinstance(action, SetTo):
            target = action.into_storage
            value = self._generate_code_for_int_expr(action.value_expr, ctx, target)
            result.add(f"state->c.{target.name} = {value};")
        elif isinstance(action, SetToStr):
            assert action.into_storage.holds_a(OutputStorageType.STR)
            # Check if we need to allocate
            if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC_ON_DEMAND) and action.into_storage.default_value is None:  # if it wasn't None it'd be allocated in the start()
                if is_start:
                    # if we're at the start, and there's no default value, and on demand is in effect, there's no possible way for state->c to have any value other than NULL
                    result.add(f"state->c.{action.into_storage.name} = malloc({action.into_storage.str_size});")
                else:
                    result.add(f"if (!state->c.{action.into_storage.name}) state->c.{action.into_storage.name} = malloc({action.into_storage.str_size});")
            if len(action.value_expr) > action.into_storage.effective_string_size():
                raise IllegalDFAStateError("Literal is too long for output", action)
            result.add(self._generate_set_string(action.value_expr, action.into_storage))
            result.add(f"state->{action.into_storage.name}_counter = {len(action.value_expr)};")
        elif isinstance(action, DeleteBuf):
            assert action.into_storage.holds_buflike()

            # free buffer if required
            if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC_ON_DEMAND) and ProgramData.do(ProgramFlag.DELETE_STRING_FREE_MEMORY) and not is_start and self._is_dynamic(action.into_storage):
                result.add(f"free(state->c.{action.into_storage.name});")
                result.add(f"state->c.{action.into_storage.name} = NULL;")
            else:
                # if buffer is not freed, ensure strings are made empty
                if action.into_storage.holds_a(OutputStorageType.STR) and action.into_storage.str_null:
                    result.add(f"state->c.{action.into_storage.name}[0] = 0;")

            result.add(f"state->{action.into_storage.name}_counter = 0;");
        elif isinstance(action, (AppendTo, AppendCharTo)):
            assert action.into_storage.holds_buflike()
            output_length_expr = self._generate_buflike_length_expr(action.into_storage)
            # Check if we need to allocate
            if ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC_ON_DEMAND) and action.into_storage.default_value is None and self._is_dynamic(action.into_storage):  # if it wasn't None it'd be allocated in the start()
                result.add(f"if (!state->c.{action.into_storage.name}) state->c.{action.into_storage.name} = malloc({output_length_expr});")
            # We treat the size given in by the user as including a terminating null (if requested, anyways)
            max_length_expr = self._generate_buflike_length_expr(action.into_storage, include_null=True)
            result.add(f"if (state->{action.into_storage.name}_counter == {max_length_expr}) {{")
            with result as body:
                body.add(f"state->state = {self.dfa.states.index(action.end_target)};")
                if transition is not None:
                    body.add(f"goto repeatswitch;") # Fallthrough via switch
                else:
                    body.add(f"return {self.program_name.upper()}_OK;") # end processing instructions
            result.add("}")
            result.add("else {")
            with result as body:
                target_expression = "inval" if isinstance(action, AppendTo) else self._generate_code_for_int_expr(
                    action.append_value, ctx, OutputStorage(OutputStorageType.INT, "$appendctx")
                )
                char_type = self._get_string_char_type()
                if action.into_storage.holds_a(OutputStorageType.RAW):
                    char_type = "uint8_t"
                body.add(f"{self._generate_buflike_index_expr(action.into_storage, f'state->{action.into_storage.name}_counter++')} = ({char_type})({target_expression});")
                if action.into_storage.holds_a(OutputStorageType.STR) and action.into_storage.str_null:
                    body.add(f"{self._generate_buflike_index_expr(action.into_storage, f'state->{action.into_storage.name}_counter')} = 0;")
            result.add("}")
        elif isinstance(action, CallHook):
            parm = "0" if is_start else "inval"
            if ProgramData.do(ProgramFlag.HOOK_GLOBAL):
                result.add(f"{self.program_name}_{action.name}_hook(state, {parm});")
            elif ProgramData.do(ProgramFlag.HOOK_PER_STATE):
                result.add(f"(state->{action.name}_hook)(state, {parm});")
            else:
                raise IllegalDFAStateError("Attempt to use hook while no hook method is enabled", action)
        elif isinstance(action, ConditionalAction):
            generated_if = False
            for condition in action.conditions:
                if isinstance(condition, ElseCondition):
                    if not generated_if:
                        raise IllegalDFAStateError("Else condition is not last in list", action)
                    result.add("else {")
                else:
                    if generated_if:
                        cond_name = "else if"
                    else:
                        generated_if = True
                        cond_name = "if"

                    result.add(f"{cond_name} ({self._generate_condition(condition, is_start or is_end, True)}) {{")

                # Generate condition body like normal
                with result as body:
                    for sub_act in action.sub_actions[condition]:
                        body += self._generate_action_implementation(sub_act, is_start=is_start, is_end=is_end, transition=transition)

                result.add("}")
        elif isinstance(action, BreakAction):
            # And add the finish actions to everything that pointed at it
            # Generate all subactions
            result.add("// break subactions")
            for subaction in action.replacement_actions():
                result += self._generate_action_implementation(subaction, is_start=is_start, is_end=is_end, transition=transition)
            result.add(f"state->state = {self.dfa.states.index(action.refers_to.end_state)};")
            if transition is not None:
                result.add(f"goto {self._transition_skip_action_label(transition)};")
            else:
                result.add(f"return {self.program_name.upper()}_OK;")
            ProgramData.imbue(action, DTAG.ACTION_MAY_SKIP, True)
        else:
            raise NotImplementedError(action)
        return result.value()

    def _is_dynamic(self, out_expr: OutputStorage):
        return out_expr.holds_a(OutputStorageType.STR) and ProgramData.do(ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC)

    def _generate_start_implementation(self):
        result = Outputter()

        result.add(f"{self.program_name}_result_t {self.program_name}_start({self.program_name}_state_t *state)", "{")

        with result as contents:
            # Initialize all state variables
            # First, any (if specified) default values.
            for out_expr in self.state_object_spec:
                # If a string, first init the counter value
                if out_expr.holds_buflike():
                    counter_val = 0
                    if out_expr.default_value is not None:
                        assert out_expr.holds_a(OutputStorageType.STR)
                        counter_val = len(out_expr.default_value)
                    contents.add("// initialize append counter for", out_expr.name)
                    contents.add(f"state->{out_expr.name}_counter = {counter_val};")
                if out_expr.default_value is not None:
                    contents.add("// initialize default for", out_expr.name)
                    if out_expr.holds_buflike():
                        assert out_expr.holds_a(OutputStorageType.STR)
                        # Also allocate the data if not included
                        if self._is_dynamic(out_expr):
                            contents.add(f"state->c.{out_expr.name} = malloc({self._generate_buflike_length_expr(out_expr)});")
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

        return result

    def _transition_will_directly_jump(self, transition: DFTransition, excl_fall=False):
        if transition.is_fallthrough and not excl_fall:
            return False
        if transition.target in self.dfa.accepting_states and not ProgramData.do(ProgramFlag.STRICT_DONE_TOKEN_GENERATION):
            return False
        return all(x.get_target_override_mode() == ActionOverrideMode.NONE for x in transition.actions)

    def _transition_skip_action_label(self, transition: DFTransition):
        return f"skipaction_{id(transition)}"

    def _generate_transition_body(self, transition: DFTransition, from_end=False):
        transition_body = Outputter()
        # Set the next state
        try:
            transition_body.add(f"state->state = {self.dfa.states.index(transition.target)};")
        except ValueError:
            transition_body.add("// terminating state")
        target_overriden = False
        needs_early_advance = any(x.may_return_early() for x in transition.actions)
        immediate_done = transition.target in self.dfa.accepting_states and not ProgramData.do(ProgramFlag.STRICT_DONE_TOKEN_GENERATION) and all(x.error_handling for x in transition.target.transitions)
        if needs_early_advance and not from_end and not transition.is_fallthrough and not immediate_done:
            if ProgramData.do(ProgramFlag.INDIRECT_START_PTR):
                transition_body.add(f"++(*start);");
            else:
                transition_body.add(f"++start;");
        # Generate actions
        for action in transition.actions:
            transition_body.add()
            transition_body.add(f"// action {action!r} ")
            transition_body += self._generate_action_implementation(action, is_end=from_end, transition=transition)
        if any(
            any(
                ProgramData.lookup(subact, DTAG.ACTION_MAY_SKIP, recurse_upwards=False, default=False) for subact in action.all_subactions()
            ) for action in transition.actions
        ):
            transition_body.add("// skip action label")
            transition_body.add(f"{self._transition_skip_action_label(transition)}:")
        # Check if we should fallthrough and generate a goto
        if transition.is_fallthrough:
            if transition.target in self.dfa.states:
                transition_body.add(f"// fallthrough")
                if self._transition_will_directly_jump(transition, excl_fall=True):
                    transition_body.add(f"goto fall_{self.dfa.states.index(transition.target)};")
                else:
                    transition_body.add(f"goto repeatswitch;");
            else:
                transition_body.add("// fallthrough to terminate")
        # Otherwise, if this state is targeting an accept state, return DONE instead of OK
        elif immediate_done:
            transition_body.add("// immediately return DONE")
            transition_body.add(f"return {self.program_name.upper()}_DONE;")
        # Normally, though, just generate a jump to the next jpto
        elif not from_end:
            if transition.target in self.dfa.states:
                if ProgramData.do(ProgramFlag.INDIRECT_START_PTR):
                    transition_body.add(f"if ({'++' if not needs_early_advance else ''}(*start) == end) return {self.program_name.upper()}_OK;");
                    transition_body.add("inval = **start;")
                else:
                    transition_body.add(f"if ({'++' if not needs_early_advance else ''}start == end) return {self.program_name.upper()}_OK;");
                    transition_body.add("inval = *start;")
                if self._transition_will_directly_jump(transition):
                    transition_body.add(f"goto jpto_{self.dfa.states.index(transition.target)};");
                else:
                    # use the repeatswitch case
                    transition_body.add(f"goto repeatswitch;");
            else:
                pass # terminating state
        return transition_body.value()

    def _generate_condition(self, condition: DFCondition, from_end=False, from_action=False):
        use_ctx = {
            (False, False): IntegerExprUseContext.CONDITION_PREDICATE,
            (True, False): IntegerExprUseContext.CONDITION_PREDICATE_END,
            (False, True): IntegerExprUseContext.CONDITION_PREDICATE_ACTION,
            (True, True): IntegerExprUseContext.CONDITION_PREDICATE_ACTION_END,
        }[(from_end, from_action)]
        if isinstance(condition, ConstantCondition):
            return str(condition.get_literal_result()).lower()
        elif isinstance(condition, IntegerCondition):
            return self._generate_code_for_int_expr(condition.expr, use_ctx)

    def _generate_condition_point_body(self, state: DFConditionPoint, from_end=False):
        result = Outputter()

        result.add("// conditions")

        generated_if = False
        for condition in state.transitions:
            if isinstance(condition.condition, ElseCondition):
                if not generated_if:
                    raise IllegalDFAStateError("Else condition is not last in list", state)
                result.add("else {")
            else:
                if generated_if:
                    cond_name = "else if"
                else:
                    generated_if = True
                    cond_name = "if"

                result.add(f"{cond_name} ({self._generate_condition(condition.condition, from_end)}) {{")

            # Generate condition body like normal
            with result as body:
                body += self._generate_transition_body(condition, from_end)

            result.add("}")

        result.add("// fallback for invalid state")
        result.add(f"return {self.program_name.upper()}_FAIL;")

        return result.value()

    def _generate_switch_body(self, state: DFState):
        if isinstance(state, DFConditionPoint):
            return self._generate_condition_point_body(state)
        result = Outputter()

        # Split transitions into else groups
        try:
            actual_else_transition = next(state.all_transitions_for((DFTransition.Else,)))
        except StopIteration:
            actual_else_transition = None

        result.add("// transitions")
        generated_if = False
        
        # Create all transition if cases
        for j, transition in enumerate((x for x in state.transitions if x != actual_else_transition)):
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

    def _needs_end_check(self):
        if ProgramData.do(ProgramFlag.ZERO_LEN_INPUT_SUPPORT):
            return True

        for trans in self.dfa.all_transitions():
            if any(x.may_return_early() for x in trans.actions):
                return True
        return False

    def _generate_feed_implementation(self):
        result = Outputter()

        start_typename = "const uint8_t *" if not ProgramData.do(ProgramFlag.INDIRECT_START_PTR) else "const uint8_t **";
        result.add(f"{self.program_name}_result_t {self.program_name}_feed({start_typename}start, const uint8_t *end, {self.program_name}_state_t *state) {{")
        with result as contents:
            if self._needs_end_check():
                contents.add(f"if ({'*start' if ProgramData.do(ProgramFlag.INDIRECT_START_PTR) else 'start'} == end) return {self.program_name.upper()}_OK;")
                contents.add()
                # Generate an explicit input check 
            # Generate the `inval` variable
            contents.add("uint8_t inval = " + ("**start" if ProgramData.do(ProgramFlag.INDIRECT_START_PTR) else "*start") + ";")
            contents.add()
            # Generate a target for states with actions that modify the state in an unpredictable way
            contents.add("repeatswitch:");
            # The body of feed is a massive switch statement that has a bunch of internal gotos
            contents.add("switch (state->state) {")
            for idx, state in enumerate(self.dfa.states):
                # Emit the case label
                contents.add(f"case {idx}:")
                # Emit goto target for fallthroughs if anything falls here (these are separate to make it slightly easier to read)
                if any(x.is_fallthrough and self._transition_will_directly_jump(x, excl_fall=True) for x in self.dfa.transitions_pointing_to(state)):
                    contents.add(f"fall_{idx}:")
                # If any transition can directly jump into this case, emit a label for it to do so. We don't really _need_ these checks
                # but gcc complains about unused labels in -Wall.
                if any(self._transition_will_directly_jump(x) for x in self.dfa.transitions_pointing_to(state) if x.on_values != {DFTransition.End}):
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
        if isinstance(state, DFConditionPoint):
            return self._generate_condition_point_body(state, True)
        result = Outputter()

        # Find all transitions that operate on End
        unconditional_end_transition = state[DFTransition.End]

        result.add("// possible end transitions")
        
        # Create all transitions for possible conditions
        if unconditional_end_transition:
            result += self._generate_transition_body(unconditional_end_transition, True)

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
                        contents.add(f"state->c.{out_expr.name} = NULL;")

        result.add("}")
        return result.value()

# =============
# DEBUG DUMPERS
# =============

def debug_dump_dfa(dfa: DFA, out_name="dfa", highlight=None): # pragma: no cover
    if not debug_enabled:
        raise RuntimeError("Debugging was disabled! You probably need to install graphviz")

    g = graphviz.Digraph(name='dfa', comment=ProgramData.lookup(dfa, DTAG.NAME))

    nodes = []
    edges = []

    def build_label_condition(condition: DFCondition):
        val = ProgramData.lookup(condition, DTAG.NAME)
        if not val:
            val = repr(condition)
        return graphviz.escape(val)

    def build_label_single(value):
        if ProgramData.do(ProgramFlag.DEBUG_DFA_BINARY_LABELS) and type(value) in (str, bytes):
            return f"{ord(value):02x}"
        else: return repr(value)

    def build_label_onvalues(on_values):
        on_values_remaining = list(on_values)
        on_values_remaining.sort(key=lambda x: x if isinstance(x, str) else '')

        used = []
        start_idx = sum(int(not isinstance(x, str)) for x in on_values_remaining)

        range_start = start_idx
        range_end = start_idx

        label = ""

        for i in range(start_idx + 1, len(on_values_remaining)):
            if ord(on_values_remaining[i-1]) + 1 == ord(on_values_remaining[i]):
                range_end = i
            else:
                if range_end - range_start >= ProgramData.option(ProgramOption.COLLAPSED_RANGE_LENGTH):
                    # this is a valid range
                    for j in range(range_start, range_end+1):
                        used.append(on_values_remaining[j])
                    
                    label += f"{build_label_single(on_values_remaining[range_start])}-{build_label_single(on_values_remaining[range_end])},"
                range_start = i
                range_end = i

        if range_end - range_start >= ProgramData.option(ProgramOption.COLLAPSED_RANGE_LENGTH):
            # this is a valid range
            for j in range(range_start, range_end+1):
                used.append(on_values_remaining[j])
            label += f"{build_label_single(on_values_remaining[range_start])}-{build_label_single(on_values_remaining[range_end])},"

        for x in used:
            on_values_remaining.remove(x)

        if on_values_remaining:
            label += ",".join(build_label_single(x) for x in on_values_remaining)
        elif label:
            label = label[:-1]
        label = graphviz.escape(label)
        return label

    transition_similar_count = Counter()
    replaced_actions = {}
    for state in dfa.states:
        for transition in state.all_transitions():
            transition_similar_count[(frozenset(transition.on_values), transition.target)] += 1
            for action in transition.actions:
                if action.get_target_override_mode() in [ActionOverrideMode.ALWAYS_GOTO_OTHER, ActionOverrideMode.MAY_GOTO_TARGET]:
                    replaced_actions[id(action)] = action
                    for tgt in action.get_target_override_targets():
                        transition_similar_count[(frozenset([id(action)]), tgt)] += 1

    ignored_transitions = []
    if ProgramData.do(ProgramFlag.DEBUG_DFA_HIDE_ERROR_HANDLING):
        for i, count in transition_similar_count.items():
            if count >= ProgramData.option(ProgramOption.DEBUG_DFA_HIDE_THRESHOLD):
                ignored_transitions.append(i)

    for j, state in enumerate(dfa.states):
        shape = "circle"
        if state in dfa.accepting_states:
            shape = "doublecircle"
        elif state == dfa.starting_state:
            shape = "square"
        if state == highlight:
            shape = "triangle"

        parent_thing = ProgramData.lookup(state, DTAG.PARENT)
        while parent_thing and not isinstance(parent_thing, (Node, Match)):
            parent_thing = ProgramData.lookup(parent_thing, DTAG.PARENT)
        parent_id = (id(parent_thing) * 11400714819323198485) & ((1 << 64)-1)
        parent_id >>= (64-24)
        parent_id |= 0x808080
        parent_color = "#" + format(parent_id, "06x")
        g.node(str(id(state)), shape=shape, label=str(j), style="filled,dashed" if isinstance(state, DFConditionPoint) else "filled", fillcolor=parent_color)
        for transition in state.all_transitions():
            if isinstance(transition, DFConditionalTransition):
                if transition.condition.is_literal() and transition.condition.get_literal_result() == False:
                    continue
                label = build_label_condition(transition.condition)
            else:
                if not transition.target or not transition.on_values:
                    continue
                label = build_label_onvalues(transition.on_values)
                if (frozenset(transition.on_values), transition.target) in ignored_transitions:
                    continue
            is_real = True
            overriden_target = str(id(transition.target))

            for action in transition.actions:
                acname = ProgramData.lookup(action, DTAG.NAME, recurse_upwards=False)
                if acname:
                    label += "\n{}".format(acname)
                else:
                    acname = repr(action)
                    label += f"\n{action!r}"
                if action.get_target_override_mode() in [ActionOverrideMode.ALWAYS_GOTO_OTHER, ActionOverrideMode.MAY_GOTO_TARGET]:
                    for tgt in action.get_target_override_targets():
                        if (frozenset([id(action)]), tgt) not in ignored_transitions:
                            g.edge(str(id(state)), str(id(tgt)), label=f"{acname} side effect")
                if action.get_target_override_mode() in [ActionOverrideMode.ALWAYS_GOTO_OTHER, ActionOverrideMode.ALWAYS_GOTO_UNDEFINED]:
                    if action.get_target_override_mode() == ActionOverrideMode.ALWAYS_GOTO_UNDEFINED:
                        # add fake invisible node
                        g.node("_asn" + str(id(action)), style="invis")
                        overriden_target = "_asn" + str(id(action))
                    else:
                        is_real = False
                    break
            if is_real:
                g.edge(str(id(state)), overriden_target, label=label, style="dashed" if transition.is_fallthrough else "solid", color = "orange" if transition.error_handling else "black")

    for v in ignored_transitions:
        values, target = v
        if not target or not values:
            continue
        # make a node
        g.node(str(id(v)), shape="rectangle", label=f"{transition_similar_count[v]} others", color="blue", fontsize="10")
        if type(next(iter(values))) is int:
            acid = next(iter(values))
            label = f"{ProgramData.lookup(replaced_actions[acid], DTAG.NAME, recurse_upwards=False)} side effect"
        else:
            label = build_label_onvalues(values)
        g.edge(str(id(v)), str(id(target)), label=label)

    if ProgramData.option(ProgramOption.DEBUG_GRAPH_DUMP_FORMAT) == "dot":
        g.save(out_name + ".dot")
    else:
        g.render(out_name, format=ProgramData.option(ProgramOption.DEBUG_GRAPH_DUMP_FORMAT), cleanup=True)

def debug_dump_regexnfa(nfa: RegexNFA, out_name="nfa"): # pragma: no cover
    if not debug_enabled:
        raise RuntimeError("Debugging was disabled! You probably need to install graphviz")

    g = graphviz.Digraph(name='nfa', comment=ProgramData.lookup(nfa, DTAG.NAME))

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

    if ProgramData.option(ProgramOption.DEBUG_GRAPH_DUMP_FORMAT) == "dot":
        g.save(out_name + ".dot")
    else:
        g.render(out_name, format=ProgramData.option(ProgramOption.DEBUG_GRAPH_DUMP_FORMAT), cleanup=True)

def debug_dump_ast(ast, out_name="ast", into=None, coming_from=None, make_id=None, make_subgraph=None): # pragma: no cover
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

        if ProgramData.option(ProgramOption.DEBUG_GRAPH_DUMP_FORMAT) == "dot":
            g.save(out_name + ".dot")
        else:
            g.render(out_name, format=ProgramData.option(ProgramOption.DEBUG_GRAPH_DUMP_FORMAT), cleanup=True)
        return

    def label_of(x):
        base = type(x).__name__
        if type(x).__repr__ != object.__repr__:
            base = repr(x)

        name = ProgramData.lookup(x, DTAG.NAME, recurse_upwards=False)
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

        elif isinstance(ast, IfElseNode):
            start_node = make_id()
            c.node(start_node, "", style="filled", shape="circle", color="grey")
            tie_to(start_node)

            next_coming_from = []

            for j, condition in enumerate(ast.branches):
                sub_ast = ast.branch_bodies[condition]
                sub_coming_from = []
                ematch_node = make_id()
                c.node(ematch_node, f"{j}: {label_of(condition)}")

                tie_to(ematch_node, [start_node], extra_kwargs={"label": "\n".join(label_of(x) for x in ast.branch_actions[condition])})
                sub_coming_from.append(ematch_node)

                if sub_ast:
                    next_coming_from.extend(debug_dump_ast(sub_ast, into=c, coming_from=sub_coming_from, make_id=make_id, make_subgraph=make_subgraph))
                else:
                    next_coming_from.append(ematch_node)

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

            if ast.incoming_handler_actions:
                c.node(make_id(), f"hstart: " + ",".join(label_of(x) for x in ast.incoming_handler_actions), style="filled", color="white")

            if ast.after_actions:
                c.node(make_id(), f"finish: " + ",".join(label_of(x) for x in ast.after_actions), style="filled", color="white")

            if not ast.handles:
                next_coming_from.append(handler_start)
            elif ast.handler is not None:
                next_coming_from.extend(debug_dump_ast(ast.handler, into=c, coming_from=[handler_start], make_id=make_id, make_subgraph=make_subgraph))

            coming_from = next_coming_from

        elif isinstance(ast, ForeachNode):
            next_coming_from = debug_dump_ast(ast.child_node, into=c, coming_from=coming_from, make_id=make_id, make_subgraph=make_subgraph)

            handler_start = make_id()
            if ast.each_actions:
                c.node(make_id(), f"each: " + ",".join(label_of(x) for x in ast.each_actions), style="filled", color="white")

            if ast.after_actions:
                c.node(make_id(), f"finish: " + ",".join(label_of(x) for x in ast.after_actions), style="filled", color="white")

            coming_from = next_coming_from

        else:
            node = make_id()
            c.node(node, "?", color="blue")
            tie_to(node)
            coming_from = [node]
            
    if ast.get_next():
        return debug_dump_ast(ast.get_next(), into=into, coming_from=coming_from, make_id=make_id, make_subgraph=make_subgraph)
    return coming_from

def debug_dump_regextree(rx, indent=0): # pragma: no cover
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

def debug_dump_datatree(v: object, indent=0, disallow=None, target=sys.stdout): # pragma: no cover
    if v is None:
        for key in ProgramData._children.copy():
            if ProgramData.lookup(key, DTAG.PARENT, recurse_upwards=False) is None:
                if ProgramData.do(ProgramFlag.DEBUG_DTREE_HIDE_GC) and (key not in ProgramData._refmap or ProgramData._refmap[key]() is None):
                    continue
                debug_dump_datatree(key, target=target)
        return

    def lprint(*args, **kwargs):
        nonlocal indent, target

        if target is None:
            return

        print(" "*indent, end="", file=target)
        print(*args, **kwargs, file=target)

    if disallow is not None:
        if v in disallow or id(v) in disallow:
            lprint("<recerr>")
            return
        disallow = [*disallow, v]
    else:
        disallow = [v]

    if type(v) is int and v in ProgramData._refmap and ProgramData._refmap[v]() is not None:
        v = ProgramData._refmap[v]()

    name = ProgramData.lookup(v, DTAG.NAME, recurse_upwards=False)
    if name is None:
        if type(v) is int:
            if ProgramData.do(ProgramFlag.DEBUG_DTREE_HIDE_GC):
                lprint("<gcd>")
                return
            lprint("<unk>")
        else:
            lprint(repr(v))
    else:
        lprint(f"{name} ({v!r})")

    if type(v) is not int:
        v = id(v)

    for tag in DTAG:
        if tag in (DTAG.PARENT, DTAG.NAME): continue
        aux = ProgramData.lookup(v, tag, recurse_upwards=False, recurse_downwards=False)
        if aux is not None:
            lprint(f" {tag}: {aux}")

    for i in ProgramData._children[v]:
        debug_dump_datatree(i, indent+2, disallow, target=target)

def main(): # pragma: no cover
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
        parse_tree = parser.parse(contents, start="start")
    except lark.LarkError as e:
        print("Syntax error:", str(e), file=sys.stderr)
        exit(3)

    if ProgramData.dump(DebugDumpable.PARSE): lark.tree.pydot__tree_to_png(parse_tree, ProgramData.dump_prefix + ".parse.png")

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
        if ProgramData.dump(DebugDumpable.DTREE): debug_dump_datatree(None)
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

    if ProgramData.dump(DebugDumpable.DTREE): debug_dump_datatree(None)

    with open(program_name + ".h", "w") as f:
        f.write(header)
    with open(program_name + ".c", "w") as f:
        f.write(source)

if __name__ == "__main__":
    main()
