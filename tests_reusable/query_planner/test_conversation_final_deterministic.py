from __future__ import annotations

import importlib.util
import importlib.machinery
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent

class DummyLLM:
    def invoke(self, _messages):
        return json.dumps({"resolved_question":"","mode":"standalone","active_topic":"","active_entity":"","confidence":0.0,"reason":""})


def load_conversation():
    runtime = types.ModuleType("ai_platform.runtime.llm")
    runtime.query_understanding_llm = DummyLLM()
    runtime_pkg = types.ModuleType("ai_platform.runtime")
    ai = types.ModuleType("ai_platform")
    ai.runtime = runtime_pkg
    runtime_pkg.llm = runtime
    sys.modules.update({"ai_platform": ai, "ai_platform.runtime": runtime_pkg, "ai_platform.runtime.llm": runtime})
    source = ROOT / "conversation.py.production_v6"
    loader = importlib.machinery.SourceFileLoader("conversation_v6_test", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run():
    c = load_conversation()
    history = [{"role":"user","content":"What are the requirements for M.Tech?"}]
    r = c.resolve_conversation(question="Last date?", chat_history=history)
    assert r["mode"] == "follow_up" and "M.Tech" in r["resolved_question"], r
    r = c.resolve_conversation(question="7.1?", chat_history=history)
    assert r["mode"] == "follow_up" and "M.Tech" in r["resolved_question"], r
    r = c.resolve_conversation(question="Is 7.1 CGPA enough?", chat_history=history)
    assert r["mode"] == "follow_up" and "M.Tech" in r["resolved_question"], r
    r = c.resolve_conversation(question="Is experience compulsory?", chat_history=history)
    assert r["mode"] == "follow_up" and "M.Tech" in r["resolved_question"], r
    history2 = [{"role":"user","content":"Tell me about M.Tech and MBA."}]
    r = c.resolve_conversation(question="What about the application fee for that program?", chat_history=history2)
    assert r["mode"] == "standalone" and r["active_entity"] == "", r
    history3 = [{"role":"user","content":"What are M.Tech eligibility requirements?"}]
    r = c.resolve_conversation(question="Okay, now what about the deadline?", chat_history=history3)
    assert r["mode"] == "follow_up" and r["active_entity"] == "M.Tech" and "M.Tech" in r["resolved_question"], r
    print("PASS conversation single-anchor deadline")
    print("PASS conversation numeric follow-up")
    print("PASS conversation numeric-context follow-up")
    print("PASS conversation short experience follow-up")
    print("PASS conversation ambiguous reference refuses guess")
    print("PASS conversation discourse-prefixed deadline follow-up")
    print("CONVERSATION V6 DETERMINISTIC REGRESSION: PASS")

if __name__ == "__main__":
    run()
