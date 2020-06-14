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

DISJOINT_EXAMPLES = {
        b"test": create_regex_match(r"/test/"),
        b"ter+t": create_regex_match(r"/ter+t/"),
        br't[^ter"]*a': create_regex_match(r'/t[^ter"]*a/'),
        br't"[^"]+"': create_regex_match(r'/t"[^"]+"/'),
        br"oof": create_regex_match(r"/oof/"),
        br"yargle": create_regex_match(r"/yargle/"),
        br"oor+t": create_regex_match(r"/oor+t/"),
        br"(te|r)y": create_regex_match(r"/(te)|ry/")
}

@given(st.sets(st.sampled_from(list(DISJOINT_EXAMPLES.keys())), min_size=1))
def test_merge_with_regexes(re_matches):
    obj = nmfu.CaseNode({})
    obj._merge([DISJOINT_EXAMPLES[x].convert(defaultdict(lambda: None)) for x in re_matches], None)

@given(st.one_of(st.from_regex(x, fullmatch=True) for x in DISJOINT_EXAMPLES))
@settings(max_examples=500)
@example(b"tfa")  # this caused a bug before version 0.2.0
def test_valid_end_states(input_str):
    obj = nmfu.CaseNode({})
    vals = {x: x.convert(defaultdict(lambda: None)) for x in DISJOINT_EXAMPLES.values()}
    sub_dfa, cfs = obj._merge(list(vals.values()), None)
    corresponding = None

    # figure out which one it matches
    for i in DISJOINT_EXAMPLES:
        if re.fullmatch(i, input_str):
            corresponding = DISJOINT_EXAMPLES[i]
            break

    if corresponding is None:
        return

    assert sub_dfa.simulate([chr(x) for x in input_str]) in cfs[vals[corresponding]]
