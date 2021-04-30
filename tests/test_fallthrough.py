import nmfu
from collections import defaultdict

def test_fallthrough_simulate():
    dfa = nmfu.DFA()

    s1 = nmfu.DFState()
    s2 = nmfu.DFState()
    s3 = nmfu.DFState()

    s1['a'] = s2
    s1['b'] = nmfu.DFTransition().to(s2).fallthrough()
    s1['c'] = s3

    dfa.mark_accepting(s3)

    s2['b'] = s3
    s2['a'] = s2

    dfa.starting_state = s1

    assert dfa.simulate("a") == s2
    assert dfa.simulate("b") == s3
    assert dfa.simulate("ab") == s3
    assert dfa.simulate("c") == s3

def test_case_else_fallthrough_generation():
    errs = nmfu.DFState()
    target_case_match = nmfu.DirectMatch("handle")

    complex_example_a = nmfu.RegexMatch(nmfu.parser.parse(r'/te[^h]adle+/', start="regex"))
    complex_example_b = nmfu.CaseDirectMatch("trap")
    
    node = nmfu.CaseNode({frozenset([None]): target_case_match, frozenset([complex_example_a, complex_example_b]): None})
    dfa = node.convert(defaultdict(lambda: errs))

    assert dfa.simulate("handle") in dfa.accepting_states
    assert dfa.simulate("trap") in dfa.accepting_states
    assert dfa.simulate("tehandle") in dfa.accepting_states
    assert dfa.simulate("tegadle") in dfa.accepting_states
    assert dfa.simulate("hand") not in dfa.accepting_states
    assert dfa.simulate("tehadle") not in dfa.accepting_states
