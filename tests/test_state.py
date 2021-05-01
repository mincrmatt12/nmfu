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

def test_set_getitem():
    s1 = nmfu.DFState()
    s2 = nmfu.DFState()

    s1["a"] = s2
    s1[[["a", "b"], []]] = s1

    assert s1["a"].target == s1
    assert s1["b"].target == s1

    s1 = nmfu.DFState()
    s1[[["a", "b", "c", "d"], []]] = s2
    s1[[["b", "c"], []]] = s1

    assert s1["a"].target == s2
    assert s1["b"].target == s1
    assert s1["d"].target == s2

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
