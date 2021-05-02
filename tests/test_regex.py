import re
import nmfu
import itertools
from hypothesis import given, assume
import hypothesis.strategies as st
import pytest
from collections import defaultdict

EXAMPLE_REGEXES = [
    b"test",
    b"ter+t",
    b"as?s?df",
    br't[^ter"]*a',
    br't"[^"]+"',
    br'"[^"]+"end?',
    br"yar.le",
    br"[abci-p][^def][^ghi][\w]\s",
    br"(te|r)y",
    br"asdf|test+(ab)*",
    br"\d{3}|[abc]{3,}|[def]{3,5}",
]

@pytest.fixture(scope="module", params=EXAMPLE_REGEXES)
def match_fixture(request):
    x = request.param
    result = nmfu.RegexMatch(nmfu.parser.parse('/' + x.decode("ascii") + '/', start="regex")).convert(defaultdict(lambda: None)), re.compile(x), x
    assert result[0].is_valid()
    return result

@given(data=st.data())
def test_regexes_match(data, match_fixture):
    dfa, pre, pattern = match_fixture
    input_str = data.draw(st.one_of(st.from_regex(pattern, fullmatch=True), *(
        st.from_regex(x) for x in EXAMPLE_REGEXES
    )), label="input_str")

    result = dfa.simulate([chr(x) for x in input_str])
    if pre.fullmatch(input_str):
        assert result in dfa.accepting_states
    else:
        assert result not in dfa.accepting_states

def test_regex_charclass_splits():
    c1, c2 = nmfu.RegexCharClass(frozenset("abc")), nmfu.RegexCharClass(frozenset("bcd"))

    overlap, c1, c2 = c1.split(c2)

    assert isinstance(overlap, nmfu.RegexCharClass)
    assert isinstance(c1, nmfu.RegexCharClass)
    assert isinstance(c2, nmfu.RegexCharClass)
    
    assert overlap.chars == frozenset({"b", "c"})
    assert c1.chars == frozenset({"a"})
    assert c2.chars == frozenset({"d"})

def test_regex_inverted_charclass_splits():
    c1, c2 = nmfu.InvertedRegexCharClass(frozenset("abc")), nmfu.InvertedRegexCharClass(frozenset("bcd"))

    overlap, c1, c2 = c1.split(c2)

    assert isinstance(overlap, nmfu.InvertedRegexCharClass)
    assert isinstance(c1, nmfu.RegexCharClass)
    assert isinstance(c2, nmfu.RegexCharClass)
    
    assert overlap.chars == frozenset({"a", "b", "c", "d"})
    assert c1.chars == frozenset({"d"})
    assert c2.chars == frozenset({"a"})

def test_regex_mixed_charclass_splits():
    c1, c2 = nmfu.InvertedRegexCharClass(frozenset("abc")), nmfu.RegexCharClass(frozenset("bcd"))

    overlap, c1, c2 = c1.split(c2)

    assert isinstance(overlap, nmfu.RegexCharClass)
    assert isinstance(c1, nmfu.InvertedRegexCharClass)
    assert isinstance(c2, nmfu.RegexCharClass)
    
    assert overlap.chars == frozenset({"d"})
    assert c1.chars == frozenset({"a", "b", "c", "d"})
    assert c2.chars == frozenset({"b", "c"})

    c2, c1 = nmfu.InvertedRegexCharClass(frozenset("abc")), nmfu.RegexCharClass(frozenset("bcd"))

    overlap, c2, c1 = c1.split(c2)

    assert isinstance(overlap, nmfu.RegexCharClass)
    assert isinstance(c1, nmfu.InvertedRegexCharClass)
    assert isinstance(c2, nmfu.RegexCharClass)
    
    assert overlap.chars == frozenset({"d"})
    assert c1.chars == frozenset({"a", "b", "c", "d"})
    assert c2.chars == frozenset({"b", "c"})

split_types = [nmfu.InvertedRegexCharClass, nmfu.RegexCharClass]

@given(t1=st.sampled_from(split_types), t2=st.sampled_from(split_types), c1=st.frozensets(st.characters(min_codepoint=ord("a"), max_codepoint=ord("z"))), c2=st.frozensets(st.characters(min_codepoint=ord("a"), max_codepoint=ord("z"))))
def test_arbitrary_regex_splits(t1, t2, c1, c2):
    c1 = t1(c1)
    c2 = t2(c2)

    assume(not c1.isdisjoint(c2) and c1 != c2)

    overlap, c_1, c_2 = c1.split(c2)

    for i, j in itertools.combinations((overlap, c_1, c_2), 2):
        assert i.isdisjoint(j)

    assert not overlap.isdisjoint(c1)
    assert not overlap.isdisjoint(c2)
    assert not c_1.isdisjoint(c1) or c_1.empty()
    assert not c_2.isdisjoint(c2) or c_2.empty()

    assert overlap.union(c_1) == c1
    assert overlap.union(c_2) == c2
