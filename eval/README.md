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
