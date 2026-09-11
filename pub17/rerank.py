"""Cross-encoder reranking.

The bi-encoder embeds the question and each chunk separately, which is what
makes it fast enough to search the whole corpus -- and also what makes it
miss a chunk that restates the answer in different words. A cross-encoder
reads the question and one chunk together and scores the pair, which is far
more accurate and far too slow to run over every chunk. So: over-fetch
`config.RERANK_CANDIDATES` dense candidates, rescore those, keep the best
`config.TOP_K`.
"""
import functools

from sentence_transformers import CrossEncoder

from . import config
from .chunking import embed_text


@functools.lru_cache(maxsize=1)
def reranker():
    return CrossEncoder(config.RERANK_MODEL)


def rerank(question, chunks, k=None):
    """Return the top-k of `chunks` by cross-encoder score, best first.

    The question goes in bare: bge's query instruction prefix is for the
    bi-encoder only. The chunk goes in with its section context, if it has one.
    """
    k = k or config.TOP_K
    if not chunks:
        return []
    scores = reranker().predict([(question, embed_text(c)) for c in chunks], batch_size=32)
    ranked = sorted(zip(chunks, scores), key=lambda pair: -pair[1])
    return [dict(c, rerank_score=float(s)) for c, s in ranked[:k]]
