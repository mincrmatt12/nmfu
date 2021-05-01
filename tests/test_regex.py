import re
import nmfu
from hypothesis import given
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
    br"asdf|test+(ab)*"
]

@pytest.fixture(scope="module", params=EXAMPLE_REGEXES)
def match_fixture(request):
    x = request.param
    return nmfu.RegexMatch(nmfu.parser.parse('/' + x.decode("ascii") + '/', start="regex")).convert(defaultdict(lambda: None)), re.compile(x), x

@given(data=st.data())
def test_regexes_match(data, match_fixture):
    dfa, pre, pattern = match_fixture
    input_str = data.draw(st.one_of(st.from_regex(pattern, fullmatch=True), st.binary(min_size=2)))

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
