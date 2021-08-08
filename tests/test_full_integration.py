"""
Ensure all examples compile without giving errors
"""

import nmfu
import glob 
import os.path
import pytest
import shlex
import shutil
import subprocess

# get list of examples

example_files = glob.glob(os.path.join(os.path.dirname(__file__), "../example/*.nmfu"))
example_flag_combos = [
    ("-O3", "-fno-debug-dtree-hide-gc"),
    ("-O0",),
    ("-O2", "-finclude-user-ptr"),
    ("-O2", "-fallocate-str-space-dynamic-on-demand"),
    ("-O2", "-fallocate-str-space-dynamic-on-demand", "-fhook-per-state"),
    ("-O2", "--collapsed-range-length", "6", "-fno-use-cplusplus-guard"),
    ("-O2", "-fallocate-str-space-dynamic-on-demand", "-fstrings-as-u8", "-fdelete-string-free-memory", "-findirect-start-ptr", "-fstrict-done-token-generation")
]

common_flags = ["-fdebug-strict-program-data-errors"]

# check if we can find gcc or clang
compiler = None
for potential in ["gcc", "clang", "cc"]:
    path = shutil.which(potential)
    if path is not None:
        compiler = path
        break

@pytest.mark.parametrize("filename", example_files)
@pytest.mark.parametrize("options", example_flag_combos)
@pytest.mark.skipif(compiler is None, reason="no c compiler found")
def test_full_integration(filename, options, tmpdir):
    # Effectively just run NMFU

    with open(filename) as f:
        source = f.read()

    nmfu.ProgramData.load_commandline_flags((*options, *common_flags, filename))
    nmfu.ProgramData.load_source(source)

    pt = nmfu.parser.parse(source, start="start")

    pctx = nmfu.ParseCtx(pt)
    pctx.parse()

    dctx = nmfu.DfaCompileCtx(pctx)
    dctx.compile()

    os.chdir(tmpdir)

    cctx = nmfu.CodegenCtx(dctx, "undertest")
    with open("undertest.h", "w") as f:
        f.write(cctx.generate_header())
    with open("undertest.c", "w") as f:
        f.write(cctx.generate_source())

    # Check if debug dumper works
    nmfu.debug_dump_datatree(None, target=None)

    # Try to compile
    subprocess.run([compiler, "-c", "undertest.c", "-Wall", "-Werror", "-Wno-unused-label"], check=True)

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

    args = (*args, *common_flags)

    nmfu.ProgramData.load_commandline_flags(args)
    nmfu.ProgramData.load_source(source)

    pt = nmfu.parser.parse(source, start="start")
    return pt

def common_run_int_test(pt):
    pctx = nmfu.ParseCtx(pt)
    pctx.parse()

    dctx = nmfu.DfaCompileCtx(pctx)
    dctx.compile()

    cctx = nmfu.CodegenCtx(dctx, "fake_test")
    cctx.generate_header()
    cctx.generate_source()

    return dctx.dfa

@pytest.mark.parametrize("filename", example_files_ok)
def test_ok(filename):
    final_dfa = common_run_int_test(common_parse_int_test(filename))

    with open(filename) as f:
        source = f.read()

    lines = source.splitlines(keepends=False)
    testcases = []
    i = 0
    while lines[i].startswith("// ok: "):
        testcases.append(lines[i][len("// ok: "):])
        i += 1

    for case in testcases:
        assert final_dfa.simulate_accepts(case)

    testcases = []

    while lines[i].startswith("// bad: "):
        testcases.append(lines[i][len("// bad: "):])
        i += 1

    for case in testcases:
        assert not final_dfa.simulate_accepts(case)

@pytest.mark.parametrize("filename", example_files_fail)
def test_fail(filename):
    # note that the files still should be valid syntax
    pt = common_parse_int_test(filename)
    
    with pytest.raises(nmfu.NMFUError) as e:
        common_run_int_test(pt)

    str(e.value)  # ensure error string generation is tested
