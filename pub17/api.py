"""HTTP service: cited answers behind JWT auth, with every request logged.

    POST /ask            {"question": "..."}   any role; scoped to its collections
    GET  /admin/logs     recent requests       admin only
    GET  /admin/spend    today's spend         admin only
    GET  /health                               no auth

Each /ask passes three gates in order, and every outcome is logged to
query_log with its tokens, cost, per-stage latency, and retrieved chunk ids:

  1. Spend cap. If today's logged spend has reached PUB17_DAILY_SPEND_CAP_USD,
     the request is refused with 429 before any model call.
  2. Confidence. If the reranker's best score is under PUB17_MIN_RERANK_SCORE,
     nothing relevant was retrieved, and the service says so instead of paying
     a model to guess. eval/calibrate_refusal.py picked the threshold.
  3. Generation, with citations. If the model API fails (outage, rate limit,
     exhausted credit), the request is still logged, as upstream_error, and
     the caller gets a 502 rather than a stack trace.

Run:  uvicorn pub17.api:app --port 8000
"""
import time
from contextlib import asynccontextmanager

import anthropic
import jwt
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import auth, config
from .generate import answer as generate_answer
from .rerank import rerank, reranker
from .store import SCHEMA, connect, embedder, search_dense


@asynccontextmanager
async def lifespan(app):
    # Make sure the log columns exist, and load both models at startup so the
    # first request isn't the slow one.
    # Fail at startup, not on the first request, if tokens can't be verified.
    auth._secret()
    with connect() as conn:
        conn.execute(SCHEMA)
    embedder()
    reranker()
    yield


app = FastAPI(title="pub17", lifespan=lifespan)
bearer = HTTPBearer()


def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    try:
        return auth.verify(creds.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "invalid token")


def admin_user(user=Depends(current_user)):
    if not auth.is_admin(user["role"]):
        raise HTTPException(403, "admin role required")
    return user


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    n: int
    publication: str
    source_file: str
    pages: str
    section: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    top_rerank_score: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: dict[str, int]


def _spent_today(conn):
    return float(conn.execute(
        "SELECT coalesce(sum(cost_usd), 0) FROM query_log WHERE asked_at >= date_trunc('day', now())"
    ).fetchone()[0])


def _log(conn, user, question, result, chunks, top_score, timings):
    conn.execute(
        """
        INSERT INTO query_log
            (config_name, subject, role, question, answer, chunk_ids, top_similarity,
             top_rerank_score, input_tokens, output_tokens, cost_usd,
             retrieve_ms, rerank_ms, generate_ms, refused, refusal_reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            config.CONFIG_NAME, user["sub"], user["role"], question, result["text"],
            [c["id"] for c in chunks],
            chunks[0]["similarity"] if chunks and chunks[0]["similarity"] is not None else None,
            top_score, result["input_tokens"], result["output_tokens"], result["cost_usd"],
            timings.get("retrieve", 0), timings.get("rerank", 0), timings.get("generate", 0),
            result["refused"], result.get("refusal_reason"),
        ),
    )


def _ms(t0):
    return int((time.perf_counter() - t0) * 1000)


@app.get("/health")
def health():
    return {"status": "ok", "config": config.CONFIG_NAME}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, user=Depends(current_user)):
    question = req.question.strip()
    timings = {}
    empty = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    with connect() as conn:
        if _spent_today(conn) >= config.DAILY_SPEND_CAP_USD:
            result = {**empty, "text": "", "refused": True, "refusal_reason": "daily_spend_cap"}
            _log(conn, user, question, result, [], None, timings)
            raise HTTPException(429, "daily spend cap reached; try again tomorrow")

        t0 = time.perf_counter()
        candidates = search_dense(
            conn, question, k=config.RERANK_CANDIDATES, sources=auth.sources_for(user["role"])
        )
        timings["retrieve"] = _ms(t0)

        t0 = time.perf_counter()
        chunks = rerank(question, candidates)
        timings["rerank"] = _ms(t0)
        top = chunks[0]["rerank_score"] if chunks else None

        if top is None or top < config.MIN_RERANK_SCORE:
            scope = " and ".join(auth.collections_for(user["role"]))
            result = {
                **empty, "refused": True, "refusal_reason": "low_confidence",
                "text": f"I couldn't find this in the {scope} available to you, so I won't guess.",
            }
        else:
            t0 = time.perf_counter()
            try:
                result = generate_answer(question, chunks)
            except anthropic.APIError as exc:
                timings["generate"] = _ms(t0)
                failed = {**empty, "text": "", "refused": True,
                          "refusal_reason": f"upstream_error:{type(exc).__name__}"}
                _log(conn, user, question, failed, chunks, top, timings)
                raise HTTPException(502, "the answer model is unavailable; the request was logged")
            timings["generate"] = _ms(t0)
            if result["refused"]:
                result["refusal_reason"] = "model_refusal"

        _log(conn, user, question, result, chunks, top, timings)

    return AskResponse(
        answer=result["text"],
        refused=result["refused"],
        refusal_reason=result.get("refusal_reason"),
        citations=[] if result["refused"] else [
            Citation(
                n=i, publication=c["publication"], source_file=c["source_file"],
                pages=str(c["page_start"]) if c["page_start"] == c["page_end"]
                else f"{c['page_start']}-{c['page_end']}",
                section=c.get("section", ""),
            )
            for i, c in enumerate(chunks, start=1)
        ],
        top_rerank_score=top,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        cost_usd=round(result["cost_usd"], 6),
        latency_ms=timings,
    )


@app.get("/admin/logs")
def logs(limit: int = Query(20, ge=1, le=200), user=Depends(admin_user)):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT asked_at, subject, role, question, refused, refusal_reason,
                   top_rerank_score, input_tokens, output_tokens, cost_usd,
                   retrieve_ms, rerank_ms, generate_ms, chunk_ids
            FROM query_log ORDER BY id DESC LIMIT %s
            """,
            (limit,),
        ).fetchall()
    cols = ["asked_at", "subject", "role", "question", "refused", "refusal_reason",
            "top_rerank_score", "input_tokens", "output_tokens", "cost_usd",
            "retrieve_ms", "rerank_ms", "generate_ms", "chunk_ids"]
    return [
        {k: (float(v) if k == "cost_usd" and v is not None else v) for k, v in zip(cols, r)}
        for r in rows
    ]


@app.get("/admin/spend")
def spend(user=Depends(admin_user)):
    with connect() as conn:
        spent = _spent_today(conn)
    return {"spent_today_usd": round(spent, 4), "daily_cap_usd": config.DAILY_SPEND_CAP_USD,
            "remaining_usd": round(max(0.0, config.DAILY_SPEND_CAP_USD - spent), 4)}
