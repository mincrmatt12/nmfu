import pytest
import nmfu

def test_param_conflicts():
    with pytest.raises(RuntimeError, match="Conflict between"):
        nmfu.ProgramData.load_commandline_flags(("-fallocate-str-space-in-struct", "-fallocate-str-space-dynamic", "dummy"))

def test_help_doesnt_crash():
    nmfu.ProgramData._print_help()
    nmfu.ProgramData._print_help(show_all=True)

    with pytest.raises(SystemExit):
        nmfu.ProgramData.load_commandline_flags(("--help",))
    with pytest.raises(SystemExit):
        nmfu.ProgramData.load_commandline_flags(("--help-all",))
    with pytest.raises(SystemExit):
        nmfu.ProgramData.load_commandline_flags(("--version",))

def test_param_load_methods():
    assert nmfu.ProgramData.load_commandline_flags((
        "--flag", "strings-as-u8=yes", "-fno-use-cplusplus-guard", "--flag", "allocate-str-space-dynamic", "dummy.nmfu", "-t", "--dump-prefix", "asdf", "-ddfa", "-odumtest"
    )) == ("dummy.nmfu", "dumtest")

    assert nmfu.ProgramData.do(nmfu.ProgramFlag.STRINGS_AS_U8)
    assert not nmfu.ProgramData.do(nmfu.ProgramFlag.USE_CPLUSPLUS_GUARD)
    assert nmfu.ProgramData.do(nmfu.ProgramFlag.ALLOCATE_STR_SPACE_DYNAMIC)
    assert nmfu.ProgramData.do(nmfu.ProgramFlag.DYNAMIC_MEMORY)
    assert nmfu.ProgramData.dry_run
    assert nmfu.ProgramData.dump_prefix == "asdf"

def test_param_errors():
    with pytest.raises(RuntimeError, match="multiple times"):
        nmfu.ProgramData.load_commandline_flags(("test", "test"))

    with pytest.raises(RuntimeError, match="Missing value for argument"):
        nmfu.ProgramData.load_commandline_flags(("test", "--dump-prefix"))

    with pytest.raises(RuntimeError, match="Unknown option"):
        nmfu.ProgramData.load_commandline_flags(("test", "-GHA"))

    with pytest.raises(RuntimeError, match="No input file"):
        nmfu.ProgramData.load_commandline_flags(())
