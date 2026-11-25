# Evaluation fixtures

`golden.example.jsonl` demonstrates the reviewed question format for an offline evaluation set.
Replace placeholder document and chunk IDs with stable IDs from a non-production evaluation tenant.
Keep answerable, deliberately unanswerable, ACL-negative, single-part, and multi-part questions in
the suite.

`ranking.example.json` demonstrates inputs for the pure ranking metrics in
`src/services/evaluation.py`: recall@k, precision@k, reciprocal rank, and binary NDCG@k. Evaluation
is intentionally an offline concern; this RAG backend exposes no evaluation or ingestion route.

For end-to-end evaluation, send each golden question to `/v1/rag/query`, retain its MongoDB trace,
and score these dimensions independently:

- semantic result-status accuracy, especially safe abstention;
- required and acceptable document coverage;
- verified-source precision;
- reviewed answer-fact coverage;
- strategy activation and repair rates;
- provider-reported token usage and latency.

Do not tune and report on the same set. Keep development and held-out suites separate and rerun them
whenever prompts, models, retrieval budgets, fusion, context packing, or provider configuration
changes.

## Counting handler and HTTP outcomes

Join each trace to the `http.request.completed` event with the same `request_id`. The trace's
`final_status` is the selected internal verdict. `final_source_ids` contains its verified chunk
IDs. A later evidence check can withdraw that answer, and a store failure can prevent a response.

- Count returned answers by `returned_status` when `handler_outcome` is `returned`. Compare
  `returned_document_ids` with the question's expected documents. A returned abstention has an
  empty list. Failed and cancelled handlers have no returned status or document list.
- For observed HTTP completions, require both `response_started` and `response_completed`.
  Group by the event's `status_code`. Start and completion each record a successful ASGI
  send. They establish server handoff. A started 200 with an interrupted body is an interrupted
  exchange. A completed 503 is a completed error response.
- Keep separate counts for interrupted exchanges, missing transport events, and historical
  records whose outcome fields are absent. Missing traces and events are possible because
  persistence is best effort. Count each side of an unmatched join independently. A client-side
  observation is needed to establish receipt.
