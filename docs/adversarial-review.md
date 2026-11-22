# Adversarial review record

The repository was reviewed in repeated implementation → verification → adversarial-review loops.
This record captures the failure hypotheses tested and the resulting safeguards.

## Review 1: control flow and concurrency

| Attack or failure hypothesis | Result |
|---|---|
| A provider timeout is mislabeled as the global deadline | Fixed by checking the outer timeout's `expired()` state; regression covered |
| Concurrent ASGI requests share mutable request IDs or identity | Prevented with task-local `ContextVar`; concurrent slow/fast request regression covered |
| Query recovery creates unbounded provider concurrency | Bounded with a configured semaphore |
| Telemetry extends the response after the global RAG deadline | Detailed traces moved to the bounded background batch queue |
| A trace is briefly unreadable before its batch reaches MongoDB | Pending traces are available through the telemetry service until persistence completes |
| Shutdown discards queued telemetry | Worker receives a sentinel only after prior queue entries and is awaited before providers close |

## Review 2: configuration and trust boundaries

| Attack or failure hypothesis | Result |
|---|---|
| A misspelled YAML setting silently falls back to a default | All committed configuration models forbid extra keys |
| A blank optional secret is treated as configured | Blank optional provider values normalize to unset |
| An external endpoint is malformed | HTTP(S) and MongoDB URI schemes are validated before composition |
| Telemetry collections collide with application collections | Settings reject collisions and identical trace/event names |
| An oversized trusted identity causes an internal error | Identity validation is translated to a stable authorization failure |
| Metadata filters replace tenant or ACL filters | Mandatory filters are constructed inside the Qdrant manager and only appended to |
| Malformed MongoDB records escape as `KeyError`/validation internals | Chunk, current-document, and conversation records are validated and translated |

## Review 3: repository scope and provider boundaries

| Attack or failure hypothesis | Result |
|---|---|
| Production code writes to the vector database | An AST scan of `src/` rejects calls to any Qdrant point-writing method and checks that list against the installed client; it also rejects any import of `scripts` |
| Local seeding is reachable in production | There is no HTTP seed route; the seed script is the only writer and refuses to run unless `APP_ENV=local` |
| A stale price table reports false monetary cost | Removed; traces retain only provider-reported usage/request identifiers |
| OTel remains as a hidden runtime or deployment dependency | Removed from code, dependencies, Compose, and configuration |
| A corrupt reranker score poisons ordering | Non-finite and out-of-range Cohere scores are rejected |
| A future mode requires route/controller replacement | Controllers depend on the `QueryHandler` port resolved in the composition root |

## Verification evidence

The final gate runs on the exact declared Python 3.13.9 interpreter:

```bash
uv lock --check
uv run ruff format --check src scripts tests
uv run ruff check src scripts tests
uv run mypy
uv run pytest --cov --cov-report=term-missing
uv pip check
python -m compileall -q src scripts
```

CI (`.github/workflows/ci.yml`) runs the first five checks on every push and pull request, with
ruff over the whole repository. `uv pip check` and `compileall` are local checks that CI does not
run.

Live provider credentials and production datasets are deliberately not part of repository tests.
Deployment should still run a startup/readiness smoke test against its actual MongoDB, Qdrant,
OpenAI, and Cohere endpoints before receiving traffic.
