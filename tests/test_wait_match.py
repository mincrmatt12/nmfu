import nmfu
import pytest
from collections import defaultdict

def verify_fall(dfa):
    for state in dfa.states:
        for transition in state.transitions:
            assert not (transition.target == state and transition.is_fallthrough)

def test_simple_wait():
    sub_match = nmfu.DirectMatch("terminate")

    match = nmfu.WaitMatch(sub_match)
    dfa = match.convert(defaultdict(lambda: None))
    
    assert len(dfa.accepting_states) == 1
    verify_fall(dfa)
    assert dfa.simulate("terminate") in dfa.accepting_states
    assert dfa.simulate("junkterminate") in dfa.accepting_states
    assert dfa.simulate("termin") not in dfa.accepting_states
    assert dfa.simulate("terminterminate") in dfa.accepting_states

def test_wait_with_nonsimple_error_cases():
    sub_match = nmfu.RegexMatch(nmfu.parser.parse(r'/te[^e]+rm.!*/', start="regex"))

    match = nmfu.WaitMatch(sub_match)
    dfa = match.convert(defaultdict(lambda: None))

    assert len(dfa.accepting_states)
    verify_fall(dfa)
    assert dfa.simulate("teirma") in dfa.accepting_states 
    assert dfa.simulate("teeirma!") not in dfa.accepting_states 
    assert dfa.simulate("teeeteirma!!") in dfa.accepting_states 
