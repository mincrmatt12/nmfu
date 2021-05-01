"""
Use hypothesis-testing with a variety of input disjoint regexes
and input values to make sure the case match doesn't lose anything
"""

import nmfu
import itertools
from hypothesis import given, settings, example
import hypothesis.strategies as st
import pytest
from collections import defaultdict
import re

# First test: make sure the merging algorithm works correctly for a 
# variety of inputs

@given(st.sets(st.text(alphabet=st.characters(min_codepoint=0, max_codepoint=255), min_size=1), min_size=1, max_size=10))
def test_merge_with_dms(dm_prefixes):
    matches = [nmfu.DirectMatch(x).convert(defaultdict(lambda: None)) for x in dm_prefixes]
    obj = nmfu.CaseNode({})
    
    # Check if they are all disjoint
    if any(x.startswith(y) for x, y in itertools.permutations(dm_prefixes, 2)):
        with pytest.raises(nmfu.IllegalDFAStateConflictsError):
            obj._merge(matches, None)
    else:
        sub_dfa, cfs = obj._merge(matches, None)

        for j, i in enumerate(dm_prefixes):
            assert sub_dfa.simulate(i) in cfs[matches[j]]

def create_regex_match(x):
    return nmfu.RegexMatch(nmfu.parser.parse(x, start="regex"))

DISJOINT_REGEX_EXAMPLE_KEYS = [
    b"test",
    b"ter+t",
    br't[^ter"]*a',
    br't[^tea]r[^ter"]+"',
    br"oof",
    br"yargle",
    br"oor+t",
    br"(te|r)y"
]

@pytest.fixture(scope="module")
def disjoint_example_map():
    return {
        k: create_regex_match("/" + k.decode('ascii') + "/") for k in DISJOINT_REGEX_EXAMPLE_KEYS
    }

@pytest.fixture(scope="module")
def disjoint_case_merged(disjoint_example_map):
    obj = nmfu.CaseNode({})
    vals = {x: x.convert(defaultdict(lambda: None)) for x in disjoint_example_map.values()}
    sub_dfa, cfs = obj._merge(list(vals.values()), None)

    return sub_dfa, cfs, vals
    

@given(re_matches=st.sets(st.sampled_from(DISJOINT_REGEX_EXAMPLE_KEYS), min_size=1))
def test_merge_with_regexes(re_matches, disjoint_example_map):
    obj = nmfu.CaseNode({})
    obj._merge([disjoint_example_map[x].convert(defaultdict(lambda: None)) for x in re_matches], None)

@example(b"tfa")  # this caused a bug before version 0.2.0
@given(st.one_of(st.from_regex(x, fullmatch=True) for x in (DISJOINT_REGEX_EXAMPLE_KEYS + [b"..."])))
def test_valid_end_states(disjoint_case_merged, disjoint_example_map, input_str):
    sub_dfa, cfs, vals = disjoint_case_merged 
    corresponding = None

    # figure out which one it matches
    for i in DISJOINT_REGEX_EXAMPLE_KEYS:
        if re.fullmatch(i, input_str):
            corresponding = disjoint_example_map[i]
            break

    if corresponding is None:
        assert sub_dfa.simulate([chr(x) for x in input_str]) not in list(cfs.values())
    else:
        assert sub_dfa.simulate([chr(x) for x in input_str]) in cfs[vals[corresponding]]

def test_else_binding():
    errs = nmfu.DFState()
    target_case_match = nmfu.DirectMatch("handle")

    complex_example_a = nmfu.RegexMatch(nmfu.parser.parse(r'/te[^h]adle+/', start="regex"))
    complex_example_b = nmfu.RegexMatch(nmfu.parser.parse(r'/te[^g]bdle+/', start="regex"))    

    node = nmfu.CaseNode({frozenset([None]): target_case_match, frozenset([complex_example_a, complex_example_b]): None})
    dfa = node.convert(defaultdict(lambda: errs))

    assert dfa.simulate("handle") in dfa.accepting_states
    assert dfa.simulate("tehhandle") in dfa.accepting_states
    assert dfa.simulate("tegadle") in dfa.accepting_states
    assert dfa.simulate("hand") not in dfa.accepting_states
    assert dfa.simulate("tehbdleee") in dfa.accepting_states
