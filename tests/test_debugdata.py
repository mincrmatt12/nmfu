import pytest
import nmfu

def test_parent_detection():
    nmfu.ProgramData.load_commandline_flags(["-fdebug-strict-program-data-errors", "dummy"])

    thingy = nmfu.FinishAction()

    with pytest.raises(nmfu.NMFUError):
        nmfu.ProgramData.imbue(thingy, nmfu.DTAG.PARENT, thingy)

    subthingy = nmfu.DFState()

    nmfu.ProgramData.imbue(subthingy, nmfu.DTAG.PARENT, thingy)

    with pytest.raises(nmfu.NMFUError):
        nmfu.ProgramData.imbue(thingy, nmfu.DTAG.PARENT, subthingy)
