import pytest
import nmfu
from hypothesis import given, assume
import hypothesis.strategies as st

import ast

@pytest.fixture(scope="module")
def junk_pctx():
    return nmfu.ParseCtx(None)

@given(text=st.from_regex(r'[+\-]?[0-9]+', fullmatch=True))
def test_convert_int_dec(text, junk_pctx):
    assert junk_pctx._convert_int(text) == int(text)

@given(text=st.from_regex(r'[+\-]?0x[0-9a-fA-F]+', fullmatch=True))
def test_convert_int_hex(text, junk_pctx):
    assert junk_pctx._convert_int(text) == int(text, base=16)

@given(text=st.from_regex(r'0b[01]+', fullmatch=True))
def test_convert_int_bin(text, junk_pctx):
    assert junk_pctx._convert_int(text) == int(text, base=2)

def test_string_seq(junk_pctx):
    assert junk_pctx._convert_string(r'"asdf"') == "asdf"
    assert junk_pctx._convert_string(r'"oof\n"') == "oof\n"
    assert junk_pctx._convert_string(r'"\x05\x06"') == "\x05\x06"

def test_binary_seq(junk_pctx):
    assert junk_pctx._convert_binary_string('"00 12 4a56"') == "\x00\x12\x4a\x56"

    with pytest.raises(ValueError):
        junk_pctx._convert_binary_string("00 1")
