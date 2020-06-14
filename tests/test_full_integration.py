"""
Ensure all examples compile without giving errors
"""

import nmfu
import glob 
import os.path
import pytest

# get list of examples

example_files = glob.glob(os.path.join(os.path.dirname(__file__), "../example/*.nmfu"))

@pytest.mark.parametrize("filename", example_files)
def test_full_integration(filename):
    # Effectively just run NMFU

    with open(filename) as f:
        source = f.read()
        
    nmfu.ProgramData.load_commandline_flags(("-O3", filename))
    nmfu.ProgramData.load_source(source)

    pt = nmfu.parser.parse(source)

    pctx = nmfu.ParseCtx(pt)
    pctx.parse()

    dctx = nmfu.DfaCompileCtx(pctx)
    dctx.compile()

    cctx = nmfu.CodegenCtx(dctx, "fake_test")
    cctx.generate_header()
    cctx.generate_source()
    
    # yay
