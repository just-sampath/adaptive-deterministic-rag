"""Shared English word lists for lexical text processing."""

from __future__ import annotations

# Function words with no retrieval value. The BM25 sparse encoder drops them, so changing this set
# changes sparse vectors: that needs a new encoder fingerprint and a reindex.
ENGLISH_STOPWORDS = frozenset(
    word
    for line in (
        "a all also although am an and any are as at be because been before being below between",
        "both but by can could did do does doing down during each few for from further had has",
        "have having he her here hers him his how i if in into is it its just me my of off on once",
        "or our out over she should so some such than that the their them then there these they",
        "this those through to too under until up very was we were what when where which while who",
        "why will with would you your",
    )
    for word in line.split()
)
