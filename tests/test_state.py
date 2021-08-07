import nmfu
import pytest
from collections import defaultdict

def test_illegal_transitions():
    s1 = nmfu.DFState()
    s2 = nmfu.DFState()

    s1.transition(nmfu.DFTransition(["a"]).to(s2))
    with pytest.raises(nmfu.IllegalDFAStateError):
        s1.transition(nmfu.DFTransition(["a"]).to(s1))
    with pytest.raises(nmfu.IllegalDFAStateError):
        s1.transition(nmfu.DFTransition(["a", "b"]).to(s1))
    with pytest.raises(nmfu.IllegalDFAStateError):
        s1.transition(nmfu.DFTransition(["a"]).to(s2).fallthrough())
    s1.transition(nmfu.DFTransition(["b"]).to(s2))

def test_set_get_delitem():
    s1 = nmfu.DFState()
    s2 = nmfu.DFState()

    s1["a"] = s2
    s1[["a", "b"]] = s1

    assert s1["a"].target == s1
    assert s1["b"].target == s1

    s1 = nmfu.DFState()
    s1[["a", "b", "c", "d"]] = s2
    s1[["b", "c"]] = s1

    assert s1["a"].target == s2
    assert s1["b"].target == s1
    assert s1["d"].target == s2
    assert s1[["b", "c"]].target == s1
    assert s1[["a", "b", "c", "d"]] is None

    del s1["b"]

    assert s1["b"] is None
    assert s1["c"].target == s1
    assert len(s1["c"].on_values) == 1

    del s1["c"]

    assert s1["c"] is None

def test_dfa_append_after_simple():
    error_state = nmfu.DFState()

    match1 = nmfu.DirectMatch("abc").convert(defaultdict(lambda: error_state))
    match2 = nmfu.DirectMatch("def").convert(defaultdict(lambda: error_state))
    
    match1.append_after(match2, error_state)

    assert match1.simulate("abcdef") in match1.accepting_states
    assert match1.simulate("abc") not in match1.accepting_states

    match1 = nmfu.DirectMatch("abc").convert(defaultdict(lambda: error_state))
    match2 = nmfu.DirectMatch("def").convert(defaultdict(lambda: error_state))

    # try with actions
    act = [nmfu.FinishAction()]

    match1.append_after(match2, error_state, chain_actions=act)

    target = match1.simulate("abc")
    assert act[0] in target["d"].actions

def test_dfa_transition_accessors():
    dfa = nmfu.DFA()

    s1 = nmfu.DFState()
    s2 = nmfu.DFState()
    s3 = nmfu.DFState()

    s1["a"] = s3
    s1["a"].handles_else()

    s1["b"] = s2
    s2["c"] = s2

    ch = nmfu.CallHook("dummy")

    s2["c"].attach(ch)

    dfa.add(s1)
    dfa.add(s2)
    dfa.add(s3)

    assert dfa.error_handling_transitions() == {s1["a"]}
    assert dfa.error_handling_transitions(include_states=True) == {(s1, s1["a"])}
    assert dfa.transitions_that_do(ch) == {s2["c"]}
    assert dfa.transitions_pointing_to(s2, include_states=True) == {(s1, s1["b"]), (s2, s2["c"])}

def test_dfa_is_valid():
    dfa = nmfu.DFA()

    s1 = nmfu.DFState()
    s2 = nmfu.DFState()
    s3 = nmfu.DFState()

    dfa.add(s1)
    dfa.add(s2)
    dfa.add(s3)

    s1["a"] = s2
    s2["b"] = s3

    dfa.mark_accepting(s3)
    
    assert dfa.is_valid()

    del s2["b"]

    assert not dfa.is_valid()

