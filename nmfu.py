#!/usr/bin/env python
"""
NMFU - the "no memory for you" "parser" generator.

designed to create what a compsci major would yell at me for calling a dfa to parse files/protocols character by character while using as little
RAM as possible.
"""

import lark
import click
import abc
import enum
import string
import itertools
import queue
from collections import defaultdict
from typing import List, Optional, Iterable, Dict, Union, Set
try:
    import graphviz
    debug_enabled = True
except ImportError:
    debug_enabled = False

grammar = r"""
start: top_decl* parser_decl top_decl*

?top_decl: out_decl
         | macro_decl

out_decl: "out" out_type IDENTIFIER ";"
        | "out" out_type IDENTIFIER "=" atom ";"

out_type: "bool" -> bool_type
        | "int" -> int_type
        | "enum" "{" IDENTIFIER ("," IDENTIFIER)+ "}" -> enum_type
        | "str" "[" NUMBER "]" -> str_type

macro_decl: "macro" IDENTIFIER "{" statement* "}"

parser_decl: "parser" "{" statement+ "}"

?statement: block_stmt
          | simple_stmt ";"

simple_stmt: expr -> match_stmt
           | IDENTIFIER "=" expr -> assign_stmt
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

// mode types
MODE_STR: /ascii|utf8|raw/

// catch options
CATCH_OPTION: /nomatch|outofspace/

%import common.CNAME -> IDENTIFIER
%import common.NUMBER

STRING: /"(?:[^"\\]|\\.)*"/

// regex internals
REGEX_UNIMPORTANT: /[^.?*()\[\]\\+{}|\/]|\\\.|\\\*|\\\(|\\\)|\\\[|\\\]|\\\+|\\\\|\\\{|\\\}|\\\||\\\//
REGEX_OP: /[+*?]/
REGEX_CHARGROUP_ELEMENT_RAW: /[^\-\]\\\/]|\\-|\\\]|\\\\|\\\//
REGEX_CHARCLASS: /[wWdDsSntr]/

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
- 2. conversion to state machine recursively with abstract actions
- 3. make actions concrete (remove append instrs.)
- 4. (optional) state machine optimizing
- 5. codegen

The overall architecture is similar to MLang, with various context classes which each steal the resources from a
predecessor ctx.
"""

# ===========
# STATE TYPES
# ===========

class ForwardStateRef:
    def __init__(self):
        self._dependents = []

class DFTransition:
    """
    A transition

    on_values is a set of characters (or integers)
    """

    # Special tokens
    Else = type('_DFElse', (), {'__repr__': lambda x: "Else"})()
    End = type('_DFEnd', (), {'__repr__': lambda x: "End"})()

    def __init__(self, on_values=None, conditions=None):
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
        self.actions = []

    def copy(self):
        my_copy = DFTransition()
        my_copy.on_values = self.on_values.copy()
        my_copy.conditions = self.conditions.copy()
        my_copy.actions = self.actions.copy()
        my_copy.target = self.target
        return my_copy

    def __repr__(self):
        return f"<DFTransition on={self.on_values} to={self.target}>"

    def restrict(self, *conditions):
        self.conditions.extend(conditions)
        return self

    def attach(self, *actions):
        self.actions.extend(actions)
        return self
    
    def to(self, target):
        if isinstance(target, int):
            self.target = DFState.all_states[target]
        else:
            self.target = target
        return self

    @classmethod
    def from_key(cls, on_values, conditions, inherited):
        return cls(on_values, conditions).to(inherited.target).attach(*inherited.actions)

class DFState:
    all_states = {}

    def __init__(self):
        self.transitions = []
        DFState.all_states[id(self)] = self

    def transition(self, transition, allow_replace=False):
        contained = None
        for i in self.all_transitions():
            if i.conditions == transition.conditions and (set(i.on_values) & set(transition.on_values)):
                contained = i
                break
        if contained is not None and contained.target != transition.target:
            if allow_replace:
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
            if transition_in.conditions == transition.conditions and transition_in.actions == transition.actions and transition_in.target == transition.target:
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
            self.transition(DFTransition(on_values, conditions).to(data))

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
        self.states = []

    def add(self, state):
        if DebugData.lookup(state, DebugTag.PARENT) is None:
            DebugData.imbue(state, DebugTag.PARENT, self)
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
                position = position[action].target
            except AttributeError:
                return None
            else:
                if not position:
                    return position

        return position

    def transitions_pointing_to(self, target_state: DFState):
        """
        Return a set of all transitions that point to the target state that are reachable
        from the start state
        """

        visited = set()
        result = set()

        def aux(state):
            if not state:
                return
            if state in visited:
                return
            visited.add(state)

            for t in state.all_transitions():
                if t.target == target_state:
                    result.add(t)
                aux(t.target)

        aux(self.starting_state)
        return result

    def is_valid(self):
        """
        Can we still reach at least one accept state?
        """

        visited = set() 

        def aux(state):
            if not state:
                return False
            if state in visited:
                return False
            if state in self.accepting_states:
                return True
            visited.add(state)
            for t in state.all_transitions():
                if aux(t.target):
                    return True
            return False

        return aux(self.starting_state)

    def append_after(self, chained_dfa: "DFA", sub_states=None, mark_accept=True, chain_actions=None):
        """
        Add the chained_dfa in such a way that its start state "becomes" all of the `sub_states` (or if unspecified, the accept states of _this_ dfa)
        """

        if sub_states is None:
            sub_states = self.accepting_states

        # First, add transitions between each of the sub_states corresponding to the transitions of the original starting node
        for sub_state in sub_states:
            for transition in chained_dfa.starting_state.transitions: # specifically exclude the else transition
                try:
                    if not chain_actions:
                        sub_state.transition(transition.copy(), DFTransition.Else in transition.on_values)
                    else:
                        sub_state.transition(transition.copy().attach(*chain_actions), DFTransition.Else in transition.on_values)
                except IllegalDFAStateError as e:
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

class DebugFlag(enum.Enum):
    VERBOSE_REGEX_CCLASS = 0
    VERBOSE_NFA_CONV = 1
    VERBOSE_CASE_MERGE = 2

class HasDefaultDebugInfo:
    def debug_lookup(self, tag: DebugTag):
        return None

class DebugData:
    _collection = defaultdict(dict)
    _children = defaultdict(list)
    _current_source = []
    _flags = {
            x: False for x in DebugFlag # todo: make this 0
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
            DebugData._children[id(value.__repr__.__self__)].append(obj)
        DebugData._collection[id(obj)][tag] = value
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

        if tag not in DebugData._collection[id(obj)]:
            if DebugTag.PARENT in DebugData._collection[id(obj)] and recurse_upwards:
                val = cls.lookup(DebugData._collection[id(obj)][DebugTag.PARENT], tag)
                if val is not None:
                    return val
            if isinstance(obj, HasDefaultDebugInfo):
                val = obj.debug_lookup(tag)
                if val is not None:
                    return val
            if tag in (DebugTag.SOURCE_COLUMN, DebugTag.SOURCE_LINE):
                for i in cls._children[id(obj)]:
                    val = cls.lookup(i, tag, recurse_upwards=False)
                    if val is not None:
                        return val
            return None
        return DebugData._collection[id(obj)][tag]

    @classmethod
    def get_source_line(cls, line: int):
        line -= 1
        if line >= len(cls._current_source):
            return None
        else:
            return cls._current_source[line]

    @classmethod
    def load_commandline_flags(cls, all_cmd_options: List[str]):
        pass

    @classmethod
    def do(self, flag):
        return self._flags[flag]

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
        if DebugData.do(self.condition):
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
            name, line, column = (DebugData.lookup(reason, tag) for tag in (DebugTag.NAME, DebugTag.SOURCE_LINE, DebugTag.SOURCE_COLUMN))
            info_str = ""
            if name:
                info_str += f"- {name}:"
                if line is not None:
                    info_str += f"\n  at line {line}:\n{DebugData.get_source_line(line)}"
                    if column is not None:
                        info_str += "\n" + " " * (column - 1) + "^"
                else:
                    info_str = info_str[-1]
            else:
                if line is not None:
                    info_str += f"- line {line}:\n{DebugData.get_source_line(line)}"
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

class Action():
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
        return ActionOverrideMode.ALWAYS_GOTO_OTHER

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return "exit action"

class AppendTo(Action, HasDefaultDebugInfo):
    def __init__(self, end_target, into_storage: "OutputStorage"):
        self.end_target = end_target
        self.into_storage = into_storage

    def get_target_override_mode(self):
        return ActionOverrideMode.MAY_GOTO_TARGET

    def debug_lookup(self, tag: DebugTag):
        if tag == DebugTag.NAME:
            return "append action ({})".format(DebugData.lookup(self.into_storage, DebugTag.NAME))

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
        DebugData.imbue(action, DebugTag.PARENT, self)
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
        DebugData.imbue(sm, DebugTag.PARENT, self) # Mark us as the parent of this SM
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
            state[DFTransition.Else] = DFTransition().to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*start_action_holder)
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
        DebugData.imbue(sm, DebugTag.PARENT, self) # Mark us as the parent of this SM
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
            state[DFTransition.Else] = DFTransition().to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*start_action_holder)
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

# ==================
# INTEGER EXPR TYPES
# ==================

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
        DebugData.imbue(sub_match, DebugTag.PARENT, self)
        self.sub_match = sub_match

class RegexOptional:
    def __init__(self, sub_match):
        DebugData.imbue(sub_match, DebugTag.PARENT, self)
        self.sub_match = sub_match

class RegexAlternation:
    def __init__(self, sub_matches):
        self.sub_matches = set(sub_matches)
        for sub_match in self.sub_matches:
            DebugData.imbue(sub_match, DebugTag.PARENT, self)

class RegexSequence:
    def __init__(self, sub_matches):
        self.sub_matches = list(sub_matches)
        for sub_match in self.sub_matches:
            DebugData.imbue(sub_match, DebugTag.PARENT, self)

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
        DebugData.imbue(visited_states[start_dfa_state], DebugTag.PARENT, self.states[next(iter(start_dfa_state))])

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
                        DebugData.imbue(visited_states[new_state], DebugTag.PARENT, self.states[next(iter(new_state))])
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
            DebugData.imbue(new_state, DebugTag.PARENT, state)

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
        DebugData.imbue(self.regex_tree, DebugTag.PARENT, self)

        # Variables used during construction (similarly to how mlang works, to reduce arguments in recursive methods)
        self.nfa: Optional[RegexNFA] = None
        self.dfa_1: Optional[RegexNFA] = None
        self.dfa_2: Optional[RegexNFA] = None
        self.out_dfa_cache = {}

    def _convert_to_nfa(self, r, start_state: RegexNFState):
        """
        Convert the regex tree object into the NFA using Thompson construction. Return the finish state
        """
        DebugData.imbue(r, DebugTag.PARENT, start_state)

        if isinstance(r, RegexCharClass):
            # Simply convert to a boring form
            end_state = RegexNFState()
            DebugData.imbue(end_state, DebugTag.PARENT, start_state)
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
            DebugData.imbue(end_state, DebugTag.PARENT, start_state)
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
            DebugData.imbue(end_state, DebugTag.PARENT, start_state)
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
            DebugData.imbue(end_state, DebugTag.PARENT, start_state)
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
            DebugData.imbue(val, DebugTag.SOURCE_LINE, regex_tree.children[1].line)
            DebugData.imbue(val, DebugTag.SOURCE_COLUMN, regex_tree.children[1].column)
            return val
        else:
            raise NotImplementedError("don't handle {} yet".format(regex_tree.data))

    def _convert_raw_regex_unimportant(self, regex_tree: lark.Token):
        if regex_tree.value[0] == '\\':
            v = RegexCharClass((regex_tree.value[1],))
        else:
            v = RegexCharClass((regex_tree.value[0],))
        DebugData.imbue(v, DebugTag.SOURCE_LINE, regex_tree.line)
        DebugData.imbue(v, DebugTag.SOURCE_COLUMN, regex_tree.column)
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
        }[regex_char_class.children[0].value[0]]
        DebugData.imbue(val, DebugTag.SOURCE_LINE, regex_char_class.children[0].line)
        DebugData.imbue(val, DebugTag.SOURCE_COLUMN, regex_char_class.children[0].column)
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
                    dprint[DebugFlag.VERBOSE_REGEX_CCLASS]("splitting", a, "and", b)
                    keepgoing = True
                    # Split
                    overlap, newa, newb = a.split(b)
                    dprint[DebugFlag.VERBOSE_REGEX_CCLASS]("into", overlap, newa, newb)
                    io = len(total_char_classes)
                    # Add overlap
                    total_char_classes.append(overlap)
                    # Search
                    for k in new_character_classes:
                        if ia in new_character_classes[k]:
                            dprint[DebugFlag.VERBOSE_REGEX_CCLASS]("adding overlap for a at", k)
                            new_character_classes[k].append(io)
                        if ib in new_character_classes[k]:
                            dprint[DebugFlag.VERBOSE_REGEX_CCLASS]("adding overlap for b at", k)
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
        DebugData.imbue(new_state, DebugTag.PARENT, nfdfa_state)
        self.out_dfa_cache[id(nfdfa_state)] = new_state
        into.add(new_state)

        if nfdfa_state in self.dfa_2.finishing_states:
            into.mark_accepting(new_state)

        for source, target in transitions.items():
            if isinstance(source, InvertedRegexCharClass):
                # TODO: handle multiple of these
                # Convert to a normal set
                new_transitions[source.chars] = (else_path, False)
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

            new_state.transition(DFTransition(list(source)).to(target).attach(*actions))

        return new_state


    def convert(self, current_error_handlers: dict):
        # First convert to an NFA
        self.nfa = RegexNFA()
        start_state = RegexNFState()
        self.nfa.add(start_state)
        self.nfa.mark_finishing(self._convert_to_nfa(self.regex_tree, start_state))
        # Then convert to a DFA

        new_dfa = RegexNFA()
        DebugData.imbue(new_dfa, DebugTag.PARENT, self)
        self.dfa_1 = self.nfa.convert_to_dfa(self.alphabet, new_dfa)
        # Minimize the DFA

        new_dfa = RegexNFA()
        DebugData.imbue(new_dfa, DebugTag.PARENT, self)
        self.dfa_2 = self.dfa_1.minimize_dfa(self.alphabet, new_dfa)
        
        # Check if it's possible to schedule the finish actions. TODO: instead of throwing an error, attempt to move them to the next thing's start
        if any(x.is_timing_strict() for x in self.finish_actions) and any(x.transitions for x in self.dfa_2.finishing_states):
            # Complain early
            raise IllegalDFAStateConflictsError("Unable to schedule finish actions only once", *(x for x in self.dfa_2.finishing_states if x.transitions), 
                    *(x for x in self.finish_actions if x.is_timing_strict()))

        # Create a normal SM
        out_dfa = DFA()
        self._create_dfa_state(self.dfa_2.start_state, out_dfa, True, current_error_handlers[ErrorReasons.NO_MATCH])
        return DebugData.imbue(out_dfa, DebugTag.PARENT, self)

class WaitMatch(Match):
    def __init__(self, sub_match: Match):
        super().__init__()
        DebugData.imbue(sub_match, DebugTag.PARENT, self)
        self.match_contents = sub_match

    def attach(self, action: Action):
        self.match_contents.attach(action)
        super().attach(action)

    def convert(self, current_error_handlers: dict):
        sm = self.match_contents.convert(current_error_handlers)
        sm.starting_state.or_else(sm.starting_state, *self.char_actions)
        return sm

class EndMatch(Match):
    def convert(self, current_error_handlers: dict):
        """
            END
        s   -->  s
         0        1
        """
        sm = DFA()
        DebugData.imbue(sm, DebugTag.PARENT, self)
        start_state = DFState()
        sm.add(start_state)
        ok_state = DFState()
        sm.add(ok_state)
        sm.mark_accepting(ok_state)
        start_state.transition(DFTransition([DFTransition.End]).attach(*self.start_actions, *self.char_actions, *self.finish_actions).to(ok_state))
        start_state.transition(DFTransition([DFTransition.Else]).to(current_error_handlers[ErrorReasons.NO_MATCH]).attach(*self.start_actions))
        return sm

class ConcatMatch(Match):
    def __init__(self, sub_matches: List[Match]):
        super().__init__()
        self.sub_matches = sub_matches
        for i in sub_matches:
            DebugData.imbue(i, DebugTag.PARENT, self)

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
        DebugData.imbue(match, DebugTag.PARENT, self)
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
            DebugData.imbue(new_state, DebugTag.PARENT, next(iter(state))[1])

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
            
            if actual_else: # sometimes you actually don't need one
                converted_states[processing].transition(DFTransition(list(actual_else)).to(treat_as_else), allow_replace=True)

        return new_dfa, corresponding_finish_states

    def get_next(self):
        return self.next

    def set_next(self, next_node):
        if isinstance(next_node, ActionNode):
            for sub_ast in self.sub_matches.values():
                # Go to end of sub_ast
                while sub_ast.get_next() is not None:
                    sub_ast = sub_ast.get_next()

                # Adopt it
                sub_ast.set_next(next_node)
            for empty_match in self.empty_matches:
                self.case_match_actions[empty_match].extend(next_node.actions)
        else:
            self.next = next_node

    def convert(self, current_error_handlers):
        # IN NEED OF REFACTORING (slightly unclear how it does it's fairly simple job)
        print(self.empty_matches, self.sub_matches)
        has_else = any(None in x for x in itertools.chain(self.sub_matches.keys(), self.empty_matches))

        # First, render out all of the sub_dfas
        sub_dfas = {x: y.convert(current_error_handlers) for x, y in self.sub_matches.items()}

        original_backreference = {}
        empty_backreference = {}
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
                    empty_backreference[converted] = None
                    empty_backreference[None] = None

        print(mergeable_ds)

        # Create the merged acceptor
        decider_dfa, corresponding_finish_states = self._merge(mergeable_ds, current_error_handlers[ErrorReasons.NO_MATCH])

        # Check if we need to handle else
        if has_else:
            try:
                else_actions = next(v for k, v in self.case_match_actions.items() if None in k)
            except StopIteration:
                else_actions = []

            if original_backreference[None] is not None:
                for trans in decider_dfa.transitions_pointing_to(current_error_handlers[ErrorReasons.NO_MATCH]):
                    trans.to(sub_dfas[original_backreference[None]].starting_state).attach(*else_actions)
            else:
                if empty_backreference[None] is not None:
                    for trans in decider_dfa.transitions_pointing_to(current_error_handlers[ErrorReasons.NO_MATCH]):
                        trans.to(sub_dfas[empty_backreference[None]].starting_state).attach(*else_actions)
                else:
                    new_state = DFState()
                    decider_dfa.mark_accepting(new_state)
                    decider_dfa.add(new_state)
                    for trans in decider_dfa.transitions_pointing_to(current_error_handlers[ErrorReasons.NO_MATCH]):
                        trans.to(new_state).attach(*else_actions)

        # Go through and link up all the states
        for i in mergeable_ds:
            if original_backreference[i] is None:
                true_backref = empty_backreference[i]
                if true_backref is not None:
                    # Handle empty matches
                    all_transitions_empty = set().union(*(decider_dfa.transitions_pointing_to(x) for x in corresponding_finish_states[i]))
                    print(all_transitions_empty)
                    if len(all_transitions_empty) != 1 and any(x.is_timing_strict() for x in self.case_match_actions[true_backref]):
                        raise IllegalDFAStateError("Unable to schedule strict finish action for case", i)
                    # Add actions
                    for j in all_transitions_empty:
                        j.attach(*self.case_match_actions[true_backref])
            else:
                refers_to = sub_dfas[original_backreference[i]]
                print(self.case_match_actions, i, refers_to, original_backreference[i])
                decider_dfa.append_after(refers_to, corresponding_finish_states[i], chain_actions=self.case_match_actions[original_backreference[i]])

        DebugData.imbue(decider_dfa, DebugTag.PARENT, self)

        # If we need to, add a boring after thing
        if self.next is not None:
            decider_dfa.append_after(self.next.convert(current_error_handlers))
        
        return decider_dfa

class Macro:
    def __init__(self, name_token: lark.Token, parse_tree: lark.Tree):
        self.name = name_token.value
        self.parse_tree = parse_tree
        DebugData.imbue(self, DebugTag.SOURCE_LINE, name_token.line)
        DebugData.imbue(self, DebugTag.NAME, "macro " + self.name)

class ParseCtx:
    def __init__(self, parse_tree: lark.Tree):
        self._parse_tree = parse_tree
        self.macros = {} # all macros, name --> AST
        self.state_object_spec = {}
        self.ast = None
    
    def parse(self):
        # Parse state_object_spec
        for out in self._parse_tree.find_data("out_decl"):
            out_obj = self._parse_out_decl(out)
            self.state_object_spec[out_obj.name] = out_obj
        # Parse macros
        for macro in self._parse_tree.find_data("macro_decl"):
            macro_obj = Macro(macro.children[0], macro.children[1:])
            self.macros[macro_obj.name] = macro_obj
        # Parse main
        parser_decl = next(self._parse_tree.find_data("parser_decl"))
        self.ast = self._parse_stmt_seq(parser_decl.children)

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
            default_value = self._parse_integer_expr(decl.children[2])
            if not default_value.is_literal():
                raise IllegalParseTree("Default value for out-decl must be constant", decl.children[2])
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

    def _parse_integer_expr(self, expr: lark.Tree) -> IntegerExpr:
        """
        Parse an integer type expr (also has bool/etc.)
        """

        BANNED_TYPES = ["end_expr", "concat_expr", "regex", "string_const", "string_case_const"]
        if expr.data in BANNED_TYPES:
            raise IllegalParseTree("String-typed value encountered for integer-typed expression", expr)

        if expr.data == "number_const":
            val = LiteralIntegerExpr(int(expr.children[0].value))
            DebugData.imbue(val, DebugTag.SOURCE_LINE, expr.children[0].line)
            DebugData.imbue(val, DebugTag.SOURCE_COLUMN, expr.children[0].column)
            return val
        else:
            raise IllegalParseTree("Invalid expression in integer expr", expr)


    def _parse_match_expr(self, expr: lark.Tree) -> Match:
        """
        Parse a match expression into a match object
        """
        if expr.data == "string_const":
            actual_content = expr.children[0]
            match = DirectMatch(self._convert_string(actual_content.value))
            DebugData.imbue(match, DebugTag.SOURCE_LINE, actual_content.line)
            DebugData.imbue(match, DebugTag.SOURCE_COLUMN, actual_content.column)
            return match
        elif expr.data == "string_case_const":
            actual_content = expr.children[0]
            match = CaseDirectMatch(self._convert_string(actual_content.value))
            DebugData.imbue(match, DebugTag.SOURCE_LINE, actual_content.line)
            DebugData.imbue(match, DebugTag.SOURCE_COLUMN, actual_content.column)
            return match
        elif expr.data == "regex":
            return RegexMatch(expr)
        elif expr.data == "end_expr":
            match = EndMatch()
            DebugData.imbue(match, DebugTag.SOURCE_LINE, expr.line)
            DebugData.imbue(match, DebugTag.SOURCE_COLUMN, expr.column)
            return match
        elif expr.data == "concat_expr":
            return ConcatMatch(list(self._parse_match_expr(x) for x in expr.children))
        else:
            raise IllegalParseTree("Invalid expression in match expression context", expr)

    def _parse_assign_stmt(self, stmt: lark.Tree) -> Node:
        """
        Parse an assignment into its underlying action
        """

        # Resolve the target
        targeted = stmt.children[0].value
        if targeted not in self.state_object_spec:
            raise UndefinedReferenceError("output", stmt.children[0])
        targeted = self.state_object_spec[targeted]

        if targeted.type == OutputStorageType.STR:
            # Create an append expression
            # TODO: end expr target (try/except?)
            match_node = MatchNode(self._parse_match_expr(stmt.children[1]))
            match_node.match.attach(AppendTo(None, targeted))
            return match_node
        else:
            return None # TODO: int exprs

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
            try:
                name = stmt.children[0].value
                return DebugData.imbue(self._parse_stmt_seq(self.macros[name].parse_tree), DebugTag.PARENT, self.macros[name])
            except KeyError:
                pass
            raise UndefinedReferenceError("macro", stmt.children[0])
        elif stmt.data == "finish_stmt":
            act = FinishAction()
            DebugData.imbue(act, DebugTag.SOURCE_LINE, stmt.line)
            DebugData.imbue(act, DebugTag.SOURCE_COLUMN, stmt.column)
            return ActionNode(act)
        elif stmt.data == "assign_stmt":
            return self._parse_assign_stmt(stmt)
        elif stmt.data == "case_stmt":
            # Find all of the matches
            return CaseNode({k: v for k, v in (self._parse_case_clause(x) for x in stmt.children)})
        else:
            raise IllegalParseTree("Unknown statement", stmt)

    def _parse_stmt_seq(self, stmts: List[lark.Tree]) -> Node:
        """
        Parse a set of statements into one big node
        """

        next_node = None
        for stmt in reversed(stmts):
            node = self._parse_stmt(stmt)
            node.set_next(next_node)
            next_node = node
        return next_node

# =============
# DEBUG DUMPERS
# =============

def debug_dump_dfa(dfa: DFA, out_name="dfa", highlight=None):
    if not debug_enabled:
        raise RuntimeError("Debugging was disabled! You probably need to install graphviz")

    g = graphviz.Digraph(name='dfa', comment=DebugData.lookup(dfa, DebugTag.NAME))

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
            for action in transition.actions:
                acname = DebugData.lookup(action, DebugTag.NAME, recurse_upwards=False)
                if acname:
                    label += "\n{}".format(acname)
            g.edge(str(id(state)), str(id(transition.target)), label=label)

    g.render(out_name, format="pdf", cleanup=True)

def debug_dump_regexnfa(nfa: RegexNFA, out_name="nfa"):
    if not debug_enabled:
        raise RuntimeError("Debugging was disabled! You probably need to install graphviz")

    g = graphviz.Digraph(name='nfa', comment=DebugData.lookup(nfa, DebugTag.NAME))

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

if __name__ == "__main__":
    with open("boring.nmfu") as f:
        contents = f.read()

    DebugData.load_source(contents)
    parse_tree = parser.parse(contents)

    ctx = ParseCtx(parse_tree)
    ctx.parse()

    total = ctx.ast.convert(defaultdict(lambda: None))
    debug_dump_dfa(total)

    #in_arr = []
    #while hasattr(pos, "match"):
    #    in_arr.append(pos.match.convert(defaultdict(lambda: None)))
    #    pos = pos.next

    #g = CaseNode(None)
    #total, cfs = g._merge(in_arr, None)
    #debug_dump_dfa(total, 'out')
