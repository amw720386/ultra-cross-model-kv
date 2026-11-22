"""CPU: kvx.glm_gate selfcheck (tiny GLM pair): direct cache writer == override forward, whole gate flow runs."""
import pytest

pytest.importorskip("transformers.models.glm_moe_dsa")


def test_glm_gate_selfcheck():
    from kvx import glm_gate
    assert glm_gate.main(["--selfcheck", "1"]) == 0
