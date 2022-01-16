import nmfu
import itertools
import pytest
import collections

def test_action_node_merge():
    each_tester = nmfu.ActionNode(nmfu.AppendTo(None, None))
    finish_tester = nmfu.ActionNode(nmfu.SetTo(nmfu.LiteralIntegerExpr(5), None), nmfu.BreakAction(None))

    each_tester.set_next(finish_tester)
    
    assert len(each_tester.actions) == 3
    assert each_tester.next is None

@pytest.mark.parametrize("subtype", [nmfu.DirectMatch, nmfu.CaseDirectMatch])
def test_direct_matches_action_binding(subtype):
    each_tester = nmfu.AppendTo(None, None)
    finish_tester = nmfu.FinishAction()

    string = "test"

    match = subtype(string)
    match.attach(each_tester)
    match.attach(finish_tester)

    errs = nmfu.DFState()
    dfa: nmfu.DFA = match.convert(collections.defaultdict(lambda: errs))

    for i in range(len(string)-1):
        target = dfa.simulate(string[:i])
        assert each_tester in target[string[i]].actions

    second_last_state = dfa.simulate(string[:-1])
    assert finish_tester in second_last_state[string[-1]].actions

    assert all(not x.actions for x in dfa.transitions_pointing_to(errs))

def test_wait_match_action_binding():
    act = nmfu.AppendTo(None, None)

    match = nmfu.WaitMatch(nmfu.DirectMatch("term"))
    match.attach(act)

    errs = nmfu.DFState()
    dfa: nmfu.DFA = match.convert(collections.defaultdict(lambda: errs))

    # fallthrough actions are ignored because they will hopefully wind up parsing something real
    assert all(act in x.actions or x.is_fallthrough for x in dfa.transitions_pointing_to(dfa.starting_state))

def test_dfa_dfs_with_actions():
    dfa = nmfu.DFA()

    s1 = nmfu.DFState()
    s2 = nmfu.DFState()
    s3 = nmfu.DFState()

    dfa.add(s1)
    dfa.add(s2)
    dfa.add(s3)

    dfa.starting_state = s1

    s1['a'] = s2
    s2['b'] = s1

    assert set(dfa.dfs()) == {s1, s2}

    s2['b'].attach(nmfu.AppendTo(s3, None))

    assert set(dfa.dfs()) == {s1, s2, s3}

    s1['a'].attach(nmfu.FinishAction())

    assert set(dfa.dfs()) == {s1}
