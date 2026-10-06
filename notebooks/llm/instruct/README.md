# LLM instruct fine-tuning

[01_finetune.ipynb](01_finetune.ipynb) runs validation, training, test evaluation,
export/reload, inference and an offline publication preview. Each execution creates
a new output directory. The default uses an offline random tiny Llama decoder and
synthetic data; outputs do not establish domain quality.

```bash
python -m pip install -e '.[llm-instruct,notebooks,dev]'
python -m advai_notebooks.llm.instruct validate --config configs/llm/instruct/smoke.yaml
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/smoke.yaml
```

Select `lora.yaml` for an adapter experiment. See [the LLM guide](../../../docs/llm.md)
for data schemas, resume, full/adapter/merged formats and Hugging Face publication.
