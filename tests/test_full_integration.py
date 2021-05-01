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

example_files_ok   = glob.glob(os.path.join(os.path.dirname(__file__), "../example/test/*.ok.nmfu"))
example_files_fail = glob.glob(os.path.join(os.path.dirname(__file__), "../example/test/*.fail.nmfu"))

@pytest.mark.parametrize("filename", example_files_ok)
def test_ok(filename):
    # check if the file compiles correctly
    with open(filename) as f:
        source = f.read()
        
    nmfu.ProgramData.load_commandline_flags(("-O3", filename))
    nmfu.ProgramData.load_source(source)

    pt = nmfu.parser.parse(source)

    pctx = nmfu.ParseCtx(pt)
    pctx.parse()

    dctx = nmfu.DfaCompileCtx(pctx)
    dctx.compile()

@pytest.mark.parametrize("filename", example_files_fail)
def test_fail(filename):
    # check if the file doesn't compile correctly
    with open(filename) as f:
        source = f.read()
        
    nmfu.ProgramData.load_commandline_flags(("-O3", filename))
    nmfu.ProgramData.load_source(source)

    # note that the files still should be valid syntax
    pt = nmfu.parser.parse(source)
    
    with pytest.raises(nmfu.NMFUError):
        pctx = nmfu.ParseCtx(pt)
        pctx.parse()

        dctx = nmfu.DfaCompileCtx(pctx)
        dctx.compile()
