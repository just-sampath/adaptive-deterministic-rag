# Adaptive Deterministic RAG development

An asynchronous FastAPI backend for bounded, multi-tenant retrieval-augmented generation. MongoDB is
the authoritative store and telemetry sink, Qdrant is the read-only retrieval index, OpenAI supplies
text and embeddings, and Cohere reranks. The main modules are the routes, the RAG handlers and their
four search strategies, the application services, and the manager and provider adapters. Read
`README.md` first for scripts, settings and structure, and `docs/architecture.md` for the dependency
rules. This file holds the rules and conventions that aren't obvious from the code.

## Hard rules

1. **Work on a task branch.** Create it with `git switch -c <name>`.
2. **Don't commit, push, merge or open a PR unless the user asks.** Every push and pull request runs
   CI (`.github/workflows/ci.yml`).
3. **Never push `main` wholesale.** Local `main` can hold commits that must stay local. When asked to
   push, push only the commit the user names (`git push origin <sha>:main`) and say what was left
   behind.
4. **Other people and agents work in this folder.** Run `git status` before you start. Changes you
   didn't make aren't yours: don't revert, reformat, stage or commit them, and don't switch branches
   over them without asking.
5. **Never run commands that discard work:** `git clean`, `git stash -u`, `git reset --hard`,
   `git checkout -- .`, force-push. Untracked files here can be someone's only copy.
6. **Never invent facts:** no model names, prices, token costs, benchmark or evaluation results,
   latency figures, provider limits or provider API behavior you haven't checked. There is no
   model-price table on purpose, and traces record only what provider responses report. Ask, or
   leave the claim out.
7. **Commit and PR text is plain and written as the owner would write it.** One imperative sentence
   as the subject, then a short body that says why, matching the existing history. Never mention AI
   tools or assistants, and never add `Co-Authored-By` or "generated with" lines.
8. **If your environment can't write inside `.git`,** say so and carry on with the file changes.
   Don't work around it; the user creates the branch or commit.
9. **Stay in scope.** Fix what you were asked to fix. Report other problems you notice; don't change
   them.

## Commands

```bash
uv python install 3.13.9
uv sync --frozen --all-groups
uv run --frozen python -m scripts.prefetch_tokenizer
cp .env.example .env
docker compose up -d mongo qdrant        # MongoDB :27017, Qdrant :6333
uv run --frozen uvicorn server:app --app-dir src --host 0.0.0.0 --port 8000   # http://localhost:8000/docs
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov --cov-report=term-missing   # coverage must stay at or above 80%
uv pip check                             # CI doesn't run this one
```

There is no separate build step. `docker compose up --build` builds and runs the API image.

## Facts and content

- **One source per fact.** Stable values live in `config/default.yaml` and are read through
  `utils/settings.py`; prompts live in `utils/prompts.py`. Never retype a default in another module.
  When the README or `docs/architecture.md` quotes a default, it must match the YAML.
- **Facts about providers** (model names, API parameters, Structured Outputs schema rules, Qdrant
  and Cohere behavior) come from the provider's official documentation or the installed SDK, and
  are re-read when you change them. Say in your report which source you read.
- **Writing reads as a person wrote it,** not an AI.
- **List every user-facing text change in your report** as before → after: prompts, API error
  messages, answer disclosures and other text a caller or model sees.

## Security and data boundaries

- Every Qdrant query carries tenant, role and `active` filters, added inside the vector manager.
  MongoDB then rechecks tenant, ACL, deletion state and the current document version before content
  reaches a prompt. Never bypass or weaken either check. Recheck request metadata on hydrated
  seeds and neighbors. Before presenting or storing an answer, revalidate every packed passage
  from the winning attempt, including its model-visible content. Derive public sources and
  citations from that final read.
- Authorize conversation pairs before either history-consuming stage. Recheck inherited evidence
  with the winning attempt before delivery. Preserve the partial disclosure when saving a turn.
  Deploy compatible readers everywhere before enabling enriched conversation writes.
- `scripts/seed_dummy_data.py` is the only vector writer, and it refuses to run unless
  `APP_ENV=local`. No file under `src/` writes Qdrant points or imports `scripts`
  (`tests/test_seed_boundary.py` enforces both). Don't add an ingestion or write route.
- Production fails closed: an unset `APP_ENV` means production, and startup rejects placeholder or
  weak secrets and plain-HTTP credentialed endpoints. Don't relax these checks.
- Every new free-text trace field goes through `TelemetryService` redaction. Logs never contain
  request text, answers, secrets or rejected input. The JSON formatter writes exceptions as types
  and frame locations only (`utils/logging.py`), and settings errors never echo their input.
  SDK loggers that can print payloads stay at `WARNING`.
- Personal data and secrets are never put in URLs or query strings.
- Model output selects among fixed branches. It never names a strategy, adds a step or changes a
  budget, and budgets stay finite (at most four strategies and one generation repair).

## Code conventions

- **Routes are thin:** they validate transport data and call a handler port through `RouteContext`.
  Routes, middleware and `routes/exceptions.py` never import `container`, managers, providers, `db`
  or `scripts`.
- **Follow the dependency rule in `docs/architecture.md`.** `models/` imports no other application
  package. Services depend only on models, service protocols, other services and utils. Managers
  don't import providers. `container.py` is the only composition root and the only place that picks
  a concrete provider; business modules contain no provider-selection conditionals.
- **Deterministic policy stays apart from I/O.** Decisions live in handlers and services; MongoDB
  and Qdrant calls live in managers; SDK clients live in providers.
- **Option sets are `StrEnum` or `Literal` types** in `utils/settings.py`, so an invalid value is a
  type error and a startup error.
- **Settings** are read and validated once in `utils/settings.py`. Unknown YAML keys fail startup,
  and invalid values fail closed rather than silently taking a default. A new setting is documented
  everywhere settings are listed: `config/default.yaml` (and `config/local.yaml` if it needs a local
  override), `.env.example`, the README configuration tables and environment block, and
  `docker-compose.yml` when the container needs it.
- **Stable values go in YAML, not environment variables.** The environment holds only secrets,
  external endpoints and deployment selection. Propose a YAML value before a new environment
  variable.
- **Concurrency must stay safe.** No mutable shared per-request state: request data lives in locals
  or `ContextVar`s. Every external call has a timeout from the `timeouts` settings. Run concurrent
  work through `run_pair` and `run_all` in `utils/concurrency.py`. The container owns every client
  and closes it on shutdown.
- **Keep ports narrow.** Add the smallest `Protocol` only when an existing one can't do the job, and
  follow the safe extension checklist in `docs/architecture.md`.
- **Encoder or vector-layout changes** get a new fingerprint and a new physical collection. Reindex,
  then switch the alias; never change the alias's target collection in place.
- Python modules use snake_case.

## Tests

- **Keep automated tests proportional to the risk.** Add or update a focused test when a change
  affects tenant or role isolation, the retry and abstention policy, citations, redaction, settings
  validation, startup, timeouts or concurrency. Reuse existing coverage; don't add tests simply
  because a file changed.
- **Routine edits stay lightweight.** Doc, comment and wording changes don't automatically need new
  tests.
- **Assert outcomes, not incidental implementation.** Keep exact text assertions only where the
  wording itself matters (error contracts, disclosures, redacted output).
- **Run the checks affected by the change.** Broaden or repeat them only when new edits, failures or
  open concerns justify it. Don't re-run a check that already passed on unchanged code.
- **A check that loops over a registry** (search strategies, collections, indexes, muted loggers)
  still asserts the required entries by name; a list compared with itself proves little.
- **Settings need valid and rejected coverage** when their behavior changes, in
  `tests/test_config.py`.
- **A new adapter needs boundary and concurrency tests.**
- Tests must never reach a real service. Use the fakes in `tests/fakes.py` or an in-memory client.
- Coverage of `handlers`, `services` and `utils` must stay at or above 80%.

## Environment

- The owner develops on Windows 11 with PowerShell and Git Bash; CI runs on Ubuntu. README commands
  are POSIX. In PowerShell, set `$env:PYTHONPATH = "src"` instead of the `PYTHONPATH=src` prefix.
- The repository uses LF line endings and has no `.gitattributes`. On Windows, Python's
  `Path.write_text` writes CRLF, so pass `newline="\n"` when a script writes repository files.
- Python is pinned to exactly 3.13.9 and CI uses uv 0.11.33. Use `--frozen`, and don't change
  `uv.lock` unless the task is a dependency change.
- The local stack uses ports 8000 (API), 27017 (MongoDB), 6333 and 6334 (Qdrant). Don't start a
  second server on one of them.
- An unset `APP_ENV` means production, and the seed script refuses to run outside `local`.
  `.env.example` sets `APP_ENV=local`. Don't remove these guards.
- Keep measurements, evaluation runs and other scratch output out of the repository.

## Plans: FeatureScopes and ImplementedFeatures

Larger work arrives as a plan: a Markdown file the owner points you to. Plans are not kept in the
repository. They live in two folders in the repository root, and neither is ever staged or
committed.

| Folder | Holds | Git |
| --- | --- | --- |
| `FeatureScopes/` | Work not yet done: feature scopes, implementation plans, briefs shared by several plans, mock-ups, reviews waiting for action | Untracked; never stage it (or add it to `.gitignore`) |
| `ImplementedFeatures/` | Finished work: the scopes, plans and briefs of features that have shipped, kept for reference | In `.gitignore` |

- **These files can be the only copy.** They are not on the remote. Never delete, overwrite or clean
  them up without asking.
- **One folder per piece of work.** A single plan can be one file. Work with several phases, a
  shared brief or mock-ups gets its own folder (`<yyyy-mm-dd>-<topic>/` or `<topic>/`), so it moves
  as one piece.
- **Reading a plan.** Read the whole plan, and any brief it points to, before changing anything.
  Plans say what to change, what done looks like and which rules apply; working out how is your job.
  Where a plan and this file disagree, follow the plan and say so.
- **Writing a plan.** Write it so another agent with no context from your session can build from it
  directly: the goal and what done looks like, the files to change and in what order, the rules
  that apply, the traps you found while reading the code, the tests each step affects, and what to
  do when blocked on a decision only the owner can make. Several phases share one brief rather than
  repeating it.
- **Finishing.** Once the work has shipped, or when the owner says so, move its file or folder from
  `FeatureScopes/` to `ImplementedFeatures/` and note at the top what shipped (with the commit) and
  what was left undone.
- **Nothing in the repository links to either folder.** Code, README and tests must not depend on
  them.

## Documentation upkeep

- **Keep the README true.** Update its scope list, layout table, configuration tables, environment
  block, routes table and quality gates when your change affects them.
- **Keep `docs/architecture.md` true.** Update its dependency rules, state machine, startup steps and
  seam table when your change affects them.
- A change that alters one of these conventions updates `AGENTS.md` in the same change; a one-off
  exception is recorded with its scope.

## Before you report

1. For code or configuration changes, run `ruff format --check`, `ruff check`, `mypy` and `pytest`
   with coverage, plus `uv lock --check` and `uv pip check` when dependencies change.
   Documentation-only changes need a diff and consistency check, not a test run.
2. Run the relevant behavior checks and report their results. If the change touches startup,
   providers or the stores, say whether you ran the API against the local Docker stack.

Then report: what changed and where, every user-facing text change as before → after, new or changed
settings, test results with counts and coverage, and plainly what you could not verify. Passing tests
with fakes is not a live provider check; say which you actually ran.
