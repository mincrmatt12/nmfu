import nmfu
import pytest
from collections import defaultdict

def test_equivalent_state_gen():
    a = nmfu.DFConditionPoint()
    b, c, d, e = (nmfu.DFState() for x in range(4))

    a.transition(nmfu.DFConditionalTransition(nmfu.ConstantCondition(False)).to(b))
    a.transition(nmfu.DFConditionalTransition(nmfu.ConstantCondition(True)).to(d))

    b['c'] = c
    d[nmfu.DFTransition.Else] = e
    d['f'] = None
    d['f'].handles_else()

    assert a.equivalent_on_values() == ({'c', nmfu.DFTransition.Else}, {'f'})

def test_type_checks():
    a = nmfu.DFState()
    b, c, d, e = (nmfu.DFState() for x in range(4))

    with pytest.raises(nmfu.IllegalDFAStateError, match="Invalid condition"):
        a.transition(nmfu.DFConditionalTransition(nmfu.ConstantCondition(False)).to(b))

    a = nmfu.DFConditionPoint()

    with pytest.raises(nmfu.IllegalDFAStateError):
        a.transition(nmfu.DFTransition("a").to(b))
