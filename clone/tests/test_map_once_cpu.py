"""CPU test of the hybrid connector's map_once scheduler logic (multi-turn: map a conversation's first prompt once,
append later turns natively). vLLM is stubbed when it is not installed."""
import importlib
import logging
import sys
import types


def _connector_class():
    try:
        import vllm  # noqa: F401
    except ImportError:
        names = ["vllm", "vllm.distributed", "vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_connector",
                 "vllm.distributed.kv_transfer.kv_connector.v1", "vllm.distributed.kv_transfer.kv_connector.v1.base",
                 "vllm.logger"]
        for n in names:
            sys.modules.setdefault(n, types.ModuleType(n))
        base = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
        for k in ("KVConnectorBase_V1", "KVConnectorMetadata", "KVConnectorRole", "SupportsHMA"):
            setattr(base, k, type(k, (), {}))
        sys.modules["vllm.logger"].init_logger = logging.getLogger
    return importlib.import_module("kvx.vllm_hybrid_connector").HybridCrossModelConnector


class _Req:
    def __init__(self, rid, ids):
        self.request_id, self.prompt_token_ids, self.kv_transfer_params = rid, ids, None


def _fresh():
    C = _connector_class()
    c = C.__new__(C)
    c.tail, c.align, c.tail_frac, c.min_len, c.map_once = 256, 1, 0.25, 16, True
    c._prefixes = {}
    c._mstats = dict(first_n=0, first_prompt=0, first_mapped=0, later_n=0, later_prompt=0, later_mapped=0)
    return c


def test_map_once_multi_turn():
    c = _fresh()
    m = lambda rid, ids: c.get_num_new_matched_tokens(_Req(rid, ids), 0)[0]
    system = list(range(5000))
    a1 = system + [9001] * 40                 # conversation a: system + first user turn
    a2 = a1 + [7] * 300 + [8] * 50            # + agent reply + tool output
    a3 = a2 + [5] * 10                        # next turn, tiny growth
    b1 = system + [9002] * 60                 # conversation b: same system prompt, different user turn
    assert m("a1", a1) == 4784                # first request: n - 256
    assert m("a2", a2) == 4784                # later turns map only a's first-request prefix
    assert m("a3", a3) == 4784
    # the template re-renders the first prompt's last token once the reply exists ("<think>\n" -> "<think>\n\n...")
    assert m("a4", a1[:-1] + [77, 78] + [6] * 400) == 4784
    assert m("b1", b1) == 4804                # a new conversation gets its own n - 256, not a's prefix
    assert m("b2", b1 + [3] * 500) == 4804
    assert m("s", list(range(100, 140))) == 30  # short first prompt: tail capped at 25%
    assert m("s", list(range(100, 140))) == 30  # the scheduler asking again gets the same decision
    st = c._mstats
    assert (st["first_n"], st["later_n"]) == (3, 4)
    assert st["first_mapped"] == 4784 + 4804 + 30
