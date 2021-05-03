"""
Ensure all examples compile without giving errors
"""

import nmfu
import glob 
import os.path
import pytest
import shlex

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

def common_parse_int_test(filename):
    # check if the file compiles correctly
    with open(filename) as f:
        source = f.read()

    lines = source.splitlines(keepends=False)
    if lines[0].startswith("// args: "):
        args = shlex.split(lines[0][len("// args: "):])
        args.append(filename)
    else:
        args = ("-O3", filename)

    nmfu.ProgramData.load_commandline_flags(args)
    nmfu.ProgramData.load_source(source)

    pt = nmfu.parser.parse(source)
    return pt

def common_run_int_test(pt):
    pctx = nmfu.ParseCtx(pt)
    pctx.parse()

    dctx = nmfu.DfaCompileCtx(pctx)
    dctx.compile()

    cctx = nmfu.CodegenCtx(dctx, "fake_test")
    cctx.generate_header()
    cctx.generate_source()

@pytest.mark.parametrize("filename", example_files_ok)
def test_ok(filename):
    common_run_int_test(common_parse_int_test(filename))

@pytest.mark.parametrize("filename", example_files_fail)
def test_fail(filename):
    # note that the files still should be valid syntax
    pt = common_parse_int_test(filename)
    
    with pytest.raises(nmfu.NMFUError):
        common_run_int_test(pt)

def test_help_doesnt_crash():
    nmfu.ProgramData._print_help()
    nmfu.ProgramData._print_help(show_all=True)
