# advai-notebooks

Notebooks and reusable Python code for model fine-tuning and evaluation in local and Google Colab environments.

## Layout

```text
notebooks/
  _shared/                  # Shared Colab setup and utilities
  embedding/                # Embedding fine-tuning
  ranker/                   # Reranker fine-tuning
  decision/                 # Decision-model fine-tuning and evaluation
  llm/
    base/                   # Base-model continued pre-training
    instruct/               # Instruction tuning

configs/                    # Versioned experiment configuration
src/advai_notebooks/        # Reusable implementation code
tests/                      # Unit and regression tests
data/                       # Local/sample data; large data is not committed
outputs/                    # Generated checkpoints and reports; not committed
requirements/               # Environment-specific dependencies
```

Notebook files orchestrate experiments and present results. Data processing, trainers, metrics and provider clients belong in `src/advai_notebooks` so they can be tested and reused across Colab sessions.
