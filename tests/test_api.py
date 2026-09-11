"""Service tests against the real database and models.

Generation is replaced with a fake, so the suite makes no model calls and
costs nothing. Requests log under subjects starting "pytest-", and those rows
are deleted afterwards. Needs the database running: docker compose up -d db.
"""
import time

import anthropic
import httpx2 as httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from pub17 import api, auth, config
from pub17.store import connect

QUESTION = "What is the maximum child tax credit per qualifying child?"


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("PUB17_JWT_SECRET", "test-secret-not-for-production-0123456789")
    with TestClient(api.app) as c:
        yield c
    with connect() as conn:
        conn.execute("DELETE FROM query_log WHERE subject LIKE 'pytest-%'")
    mp.undo()


@pytest.fixture(autouse=True)
def generated(monkeypatch):
    """Record generation calls instead of making them."""
    calls = []

    def fake(question, chunks):
        calls.append(question)
        return {"text": "fake answer [1]", "refused": False, "input_tokens": 10,
                "output_tokens": 5, "cost_usd": 0.0, "generate_ms": 1}

    monkeypatch.setattr(api, "generate_answer", fake)
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", 0.0)
    return calls


def headers(role, sub=None, ttl=300):
    return {"Authorization": f"Bearer {auth.mint(sub or f'pytest-{role}', role, ttl)}"}


def ask(client, role, question=QUESTION, **kw):
    return client.post("/ask", json={"question": question}, headers=headers(role, **kw))


def test_short_secrets_are_refused(monkeypatch):
    monkeypatch.setenv("PUB17_JWT_SECRET", "too-short")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        auth.mint("pytest-x", "viewer")


def test_health_needs_no_token(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ask_requires_a_token(client):
    assert client.post("/ask", json={"question": QUESTION}).status_code in (401, 403)


def test_rejects_forged_and_expired_tokens(client):
    forged = jwt.encode({"sub": "pytest-x", "role": "admin", "exp": int(time.time()) + 60},
                        "a-different-secret-of-sufficient-length", algorithm="HS256")
    r = client.post("/ask", json={"question": QUESTION}, headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401
    r = ask(client, "admin", ttl=-10)
    assert r.status_code == 401 and r.json()["detail"] == "token expired"


def test_viewer_only_sees_publications(client):
    r = ask(client, "viewer")
    assert r.status_code == 200
    files = {c["source_file"] for c in r.json()["citations"]}
    assert files and files <= set(auth.COLLECTIONS["publications"])


def test_preparer_also_sees_form_instructions(client):
    r = ask(client, "preparer", question="Which schedule reports the deduction for qualified tips?")
    files = {c["source_file"] for c in r.json()["citations"]}
    assert files & set(auth.COLLECTIONS["forms"])


def test_low_confidence_refuses_without_calling_the_model(client, generated, monkeypatch):
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", 2.0)  # above any reranker score
    body = ask(client, "viewer", question="What is the capital of France?").json()
    assert body["refused"] and body["refusal_reason"] == "low_confidence"
    assert body["citations"] == [] and generated == []


def test_spend_cap_refuses_before_any_model_call(client, generated, monkeypatch):
    monkeypatch.setattr(config, "DAILY_SPEND_CAP_USD", 0.0)
    assert ask(client, "viewer").status_code == 429
    assert generated == []


def test_model_api_failure_is_logged_and_returns_502(client, monkeypatch):
    def unavailable(question, chunks):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))

    monkeypatch.setattr(api, "generate_answer", unavailable)
    question = f"pytest outage {time.time()}: what is the standard deduction?"
    r = ask(client, "preparer", question=question, sub="pytest-outage")
    assert r.status_code == 502

    rows = client.get("/admin/logs?limit=50", headers=headers("admin")).json()
    row = next(r for r in rows if r["question"] == question)
    assert row["refused"] and row["refusal_reason"] == "upstream_error:APIConnectionError"


def test_admin_routes_are_admin_only_and_every_request_is_logged(client):
    question = f"pytest audit {time.time()}: what is the standard deduction for a single filer?"
    ask(client, "preparer", question=question, sub="pytest-audit")
    assert client.get("/admin/logs", headers=headers("preparer")).status_code == 403

    rows = client.get("/admin/logs?limit=50", headers=headers("admin")).json()
    row = next(r for r in rows if r["question"] == question)
    assert row["subject"] == "pytest-audit" and row["role"] == "preparer"
    assert len(row["chunk_ids"]) == config.TOP_K
    assert row["top_rerank_score"] is not None and row["rerank_ms"] >= 0

    spend = client.get("/admin/spend", headers=headers("admin")).json()
    assert {"spent_today_usd", "daily_cap_usd", "remaining_usd"} <= spend.keys()
