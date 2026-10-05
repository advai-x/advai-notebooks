# Optional model workflows

[01_retrieval_comparison.ipynb](01_retrieval_comparison.ipynb) independently trains
both models, then loads their exports for six-way retrieval evaluation.
The reusable orchestration lives in `src/advai_notebooks/workflows/`.
Neither embedding nor ranker imports this optional workflow.

Use the task notebooks for independent adjustments. Workflow settings live in
`configs/workflows/`; they do not alter training settings or sample preparation.
