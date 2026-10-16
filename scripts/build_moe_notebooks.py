"""Scaffold bilingual native MoE CPT/SFT tutorials; replacement clears outputs."""

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def make(task, chinese):
    cells = []
    cls = "BaseModel" if task == "base" else "InstructModel"
    objective = "continued pre-training (CPT)" if task == "base" else "assistant-only SFT"

    def prose(en, zh):
        cells.append(nbf.v4.new_markdown_cell(zh if chinese else en))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    prose(
        f"# Native MoE: {objective}\n\nEnglish | [简体中文](02_moe_finetune.zh-CN.ipynb)\n\n"
        "## Goal\n\nTrain a real, randomly initialized tiny OLMoE, inspect router/expert gradients, "
        "resume an epoch checkpoint and reload native artifacts. Then adapt attention **and native "
        "3D expert parameters** with LoRA. These synthetic examples validate mechanics, not model "
        "quality or large-model GPU fit. MoE is an architecture within Base/Instruct, not another task.",
        f"# 原生 MoE：{objective}\n\n[English](02_moe_finetune.ipynb) | 简体中文\n\n"
        "## Goal / 目标\n\n训练真正的随机 tiny OLMoE，检查 router/expert 梯度、恢复 epoch checkpoint "
        "并重载原生产物。再用 LoRA 微调 attention 与原生三维 expert 参数。合成样例只验证工程机制，"
        "不证明模型质量或大模型 GPU 适配。MoE 是 Base/Instruct 内的架构，不是新增任务。",
    )
    prose(
        "## Setup\n\nUse a Python kernel with `.[llm-" + task + ",notebooks,publish]`. "
        "Only Colab installs below. SDK pins: Transformers 5.18.0, PEFT 0.18.1, Accelerate 1.15.0. "
        "Default runs are CPU/fp32 and download no model weights. No uploads occur. "
        "Keep real datasets, credentials and checkpoints outside Git.",
        "## Setup / 环境\n\nKernel 需安装 `.[llm-" + task + ",notebooks,publish]`。"
        "下格仅在 Colab 安装。固定 SDK：Transformers 5.18.0、PEFT 0.18.1、Accelerate 1.15.0。"
        "默认 CPU/fp32，不下载模型权重，也不上传产物。真实数据、凭据与 checkpoint 不进入 Git。",
    )
    code(f"""
        import importlib.util
        import subprocess
        import sys
        from pathlib import Path

        try:
            IN_COLAB = importlib.util.find_spec("google.colab") is not None
        except ModuleNotFoundError:
            IN_COLAB = False
        if IN_COLAB:
            ROOT = Path("/content/advai-notebooks")
            if not ROOT.exists():
                subprocess.run(["git", "clone", "--depth", "1",
                                "https://github.com/advai-x/advai-notebooks.git", str(ROOT)], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                            f"{{ROOT}}[llm-{task},notebooks,publish]"], check=True)
        else:
            ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                        if (p / "pyproject.toml").is_file())
        print("Repository ready; no model download requested.")
    """)
    code(f"""
        import uuid
        import torch
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display
        from transformers import TrainerCallback
        from transformers.utils import logging as hf_logging
        from advai_notebooks.common.runtime import setup
        from advai_notebooks.llm.{task}.config import load_config
        from advai_notebooks.llm.{task}.data import load_data, prepare_data
        from advai_notebooks.llm.{task}.model import {cls}
        from advai_notebooks.llm.{task}.pipeline import run
        from advai_notebooks.llm.{task}.publish import publish
        from advai_notebooks.llm.moe import describe

        hf_logging.disable_progress_bar()
        plt.rcParams.update({{"font.family": "DejaVu Sans", "font.size": 11}})
        run_id = uuid.uuid4().hex[:8]
        cfg = load_config(ROOT / "configs/llm/{task}/moe/smoke.yaml", ROOT)
        cfg.output_dir = f"outputs/llm/{task}/moe-tutorial-{{run_id}}"
        print({{"model": cfg.model.id, "architecture": cfg.model.architecture,
               "seed": cfg.training.seed, "dtype": cfg.model.dtype}})
    """)
    prose(
        "## Steps\n\n### 1. Validate data and inventory actual parameters\n\nThe task owns "
        "document/conversation splitting and supervision. The tiny native MoE has four experts "
        "per layer and selects two per token. Active experts do **not** reduce resident weights.",
        "## Steps / 步骤\n\n### 1. 校验数据并统计真实参数\n\n任务负责文档/对话划分与监督。"
        "原生 tiny MoE 每层四个 expert，每 token 选择两个；激活 expert 较少不代表常驻权重减少。",
    )
    code(f"""
        rows, summary = load_data(cfg)
        initial = {cls}(cfg.model, setup(cfg), cfg.root, cfg.training, cfg.generation)
        prepared = prepare_data(rows, initial, cfg)
        print({{"splits": summary["splits"],
               "encoded_rows": {{k: len(v) for k, v in prepared.items()}}}})
        print(describe(initial.model))
        del initial, prepared
    """)
    prose(
        "### 2. Train full weights, interrupt after a complete epoch, then resume\n\n"
        "`latest` resolves to a native Trainer checkpoint with optimizer/scheduler/RNG state. "
        "Full MoE restoration uses native weight-layout conversion before Trainer restores the "
        "remaining state. Inference exports cannot resume. Keep config/data/runtime unchanged.",
        "### 2. 全参数训练，在完整 epoch 后中断并恢复\n\n`latest` 解析到包含 optimizer、scheduler、"
        "RNG 的原生 Trainer checkpoint。Full MoE 先用原生加载器转换权重布局，再由 Trainer 恢复其余状态。"
        "推理导出不能续训；配置、数据与运行环境保持不变。",
    )
    code("""
        class StopAfterEpochOne(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                if state.epoch >= 1:
                    control.should_training_stop = True

        first = run(cfg, callbacks=[StopAfterEpochOne()])
        assert not first["training"]["completed"]
        report = run(cfg, resume=cfg.path(cfg.output_dir) / "latest")
        assert report["training"]["completed"]
        audit = report["training"]["moe_gradients"]
        assert audit["groups"]["router"]["nonzero_gradient_steps"] > 0
        assert audit["groups"]["experts"]["nonzero_gradient_steps"] > 0
        print({"selected_epoch": report["training"]["best_epoch"], "gradient_audit": audit})
    """)
    prose(
        "### 3. Compare language loss separately from routing diagnostics\n\nValidation NLL "
        "selects the checkpoint, including epoch zero; test is report-only. NLL uses shifted "
        "supervision tokens, PPL = exp(NLL). The native auxiliary loss is **not** added to NLL/PPL. "
        "During accumulation the objective is a target-token-weighted mean of microbatch-local "
        "routing losses, not the nonlinear router loss of one concatenated batch.",
        "### 3. 语言损失与路由诊断分开报告\n\n验证集 NLL 选模，包含 epoch zero；test 仅报告。"
        "NLL 按移位后的监督 token 加权，PPL = exp(NLL)。原生辅助损失不计入 NLL/PPL。梯度累积采用"
        "按监督 token 加权的 microbatch 局部路由损失均值，不等同于拼接大 batch 的非线性路由损失。",
    )
    code("""
        table = "| Test diagnostic | Baseline | Selected |\\n| --- | ---: | ---: |\\n"
        for name in ("nll", "perplexity"):
            a, b = [report[key]["metrics"][name]
                    for key in ("baseline_test", "finetuned_test")]
            table += f"| {name} | {a:.4f} | {b:.4f} |\\n"
        a, b = [report[key]["moe"]["aux_loss"]
                for key in ("baseline_test", "finetuned_test")]
        table += f"| Router auxiliary loss (not NLL) | {a:.4f} | {b:.4f} |\\n"
        display(Markdown(table))
        print("Synthetic smoke only — this is not a quality benchmark.")
    """)
    prose(
        "### 4. Inspect expert utilization\n\nThe chart uses the selected model's test report, "
        "all nonpadding input tokens (including SFT prompts), and native MoE-layer order. "
        "Shares divide **top-k assignments**, sum to 100%, and start from zero. The dotted "
        "line is equal allocation, not a quality target. Small synthetic counts have no benchmark meaning.",
        "### 4. 检查专家使用分布\n\n图来自选中模型的 test 报告，统计所有非 padding 输入 token"
        "（含 SFT prompt），按原生 MoE 层顺序编号。分母是 top-k 分配次数，各柱合计 100%，纵轴从零开始。"
        "虚线表示平均分配，不是质量目标；少量合成统计不构成 benchmark。",
    )
    code("""
        layer = report["finetuned_test"]["moe"]["layers"][0]
        counts, shares = layer["assignment_counts"], layer["assignment_share"]
        assert sum(counts) == layer["input_tokens"] * 2
        assert abs(sum(shares) - 1) < 1e-8
        fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
        bars = ax.bar(range(len(shares)), [100 * s for s in shares], color="#176EAA", width=0.6)
        ax.axhline(100 / len(shares), color="#63758A", linestyle="--", label="Equal allocation")
        ax.bar_label(bars, labels=[f"{100*s:.1f}% (n={n})" for s, n in zip(shares, counts)],
                     padding=4, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.set(ylim=(0, max(100 / len(shares), max(shares) * 100) * 1.3),
               xticks=range(len(shares)), xlabel="Expert index",
               ylabel="Share of top-k assignments (%)",
               title=f"Synthetic test · MoE layer 0 · {layer['input_tokens']} input tokens · top-2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        fig.savefig(cfg.path(cfg.output_dir) / "expert-utilization.png", dpi=140)
        plt.show()
    """)
    prose(
        "### 5. Adapt attention and 3D expert parameters with LoRA\n\nMoE LoRA freezes the "
        "router in v1. `target_modules` covers attention; `target_parameters` covers native "
        "expert tensors. PEFT 0.18.1 emits a generic unsupported-layer warning for these parameter "
        "wrappers; the checks below establish gradients and merge/reload parity only for this "
        "tested tiny backend. Revalidate on real checkpoints before a long run.",
        "### 5. 用 LoRA 微调 attention 与三维 expert 参数\n\n首版 LoRA 冻结 router。"
        "`target_modules` 指定 attention，`target_parameters` 指定原生 expert 张量。PEFT 0.18.1 "
        "会为这些参数 wrapper 发出通用 unsupported-layer 警告；以下检查仅证明当前 tiny 后端的梯度与"
        "合并重载一致性。真实模型长训前需重新验证。",
    )
    code(f"""
        adapter_cfg = load_config(ROOT / "configs/llm/{task}/moe/smoke_experts_lora.yaml", ROOT)
        adapter_cfg.output_dir = f"outputs/llm/{task}/moe-experts-tutorial-{{run_id}}"
        adapter_report = run(adapter_cfg)
        adapter_audit = adapter_report["training"]["moe_gradients"]["groups"]
        assert adapter_audit["router"]["trainable_tensors"] == 0
        assert adapter_audit["experts"]["nonzero_gradient_steps"] > 0
        assert adapter_audit["other"]["nonzero_gradient_steps"] > 0
        print(adapter_audit)
    """)
    prose(
        "## Checks\n\nReload the adapter, merge a copy, compare logits and greedy generation, "
        "and preview publication offline. This verifies artifact contracts, not response quality. "
        "Only trained adapters change; original base weights remain frozen.",
        "## Checks / 检查\n\n重载 adapter、合并副本、对比 logits 与贪心生成，并离线预览发布。"
        "验证产物契约，不评价回答质量；只有 adapter 更新，原始 base 权重保持冻结。",
    )
    code(f"""
        adapter = {cls}.load(adapter_report["export"])
        merged_path = adapter.save(adapter_cfg.path(adapter_cfg.output_dir) / "merged", merged=True)
        merged = {cls}.load(merged_path)
        batch = adapter.tokenizer("the sky is blue", return_tensors="pt")
        adapter.model.eval()
        merged.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(adapter.model(**batch).logits,
                                       merged.model(**batch).logits, rtol=1e-5, atol=1e-5)
        plan = publish(adapter_report["export"], "example/moe-{task}-smoke", dry_run=True)
        names = {{item["path"] for item in plan["files"]}}
        assert "adapter_model.safetensors" in names
        assert not names & {{"optimizer.pt", "rng_state.pth", "raw-data.jsonl", ".env"}}
        print({{"native_merge_equal": True, "publication": plan["status"], "files": len(names)}})
    """)
    if task == "base":
        code("""
            result = adapter.generate(["the sky is"], max_new_tokens=3)
            assert result == merged.generate(["the sky is"], max_new_tokens=3)
            print({"synthetic_continuation": result})
        """)
    else:
        code("""
            messages = [{"role": "user", "content": "what color is the sky ?"}]
            result = adapter.chat(messages, max_new_tokens=3)
            assert result == merged.chat(messages, max_new_tokens=3)
            print({"synthetic_reply": result})
        """)
    prose(
        "## Next Steps\n\nRead the [MoE guide](../../../docs/llm-moe.md). Try `smoke_qwen3.yaml` "
        "for the second native backend or `smoke_lora.yaml` for attention-only adapters. Copy "
        "`olmoe_lora.yaml` or `qwen3_lora.yaml` into ignored `configs/local/`; replace data/output "
        "paths and audit the pinned model/tokenizer/license. The Qwen Instruct preset deliberately "
        "uses an explicit text-only non-thinking template. Real weights need all experts resident "
        "(roughly 14 GB OLMoE / 61 GB Qwen3 in bf16 before training overhead). Weight budgets are "
        "lower-bound guards, not fit promises. Real GPU training, AMP recovery and domain quality "
        "are unverified here; no QLoRA, expert parallelism or MoE PPO/GRPO is provided. "
        "Use persistent storage for Colab checkpoints; free Colab fit is not guaranteed.",
        "## Next Steps / 下一步\n\n阅读 [MoE 指南](../../../docs/llm-moe.zh-CN.md)。"
        "用 `smoke_qwen3.yaml` 检查另一原生后端，或用 `smoke_lora.yaml` 仅适配 attention。"
        "真实训练将 `olmoe_lora.yaml` / `qwen3_lora.yaml` 复制到被忽略的 `configs/local/`，"
        "替换数据与输出路径，并核查固定版本的模型、tokenizer 和许可证。Qwen Instruct 配置有意使用"
        "显式纯文本 non-thinking 模板。所有专家权重仍常驻（bf16 仅权重约 OLMoE 14 GB / Qwen3 61 GB，"
        "另需训练内存）。预算检查只是下限防护，不保证能跑。这里未验证真实 GPU 训练、AMP 恢复或领域质量，"
        "不含 QLoRA、expert parallel 或 MoE PPO/GRPO；Colab checkpoint 应持久化，免费 Colab 不保证适配。",
    )
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Explicitly clear saved outputs")
    args = parser.parse_args()
    paths = [
        (
            task,
            chinese,
            ROOT / f"notebooks/llm/{task}/02_moe_finetune{'.zh-CN' if chinese else ''}.ipynb",
        )
        for task in ("base", "instruct")
        for chinese in (False, True)
    ]
    if not args.replace and any(path.exists() for _, _, path in paths):
        raise ValueError("Notebooks exist; --replace explicitly clears saved outputs")
    for task, chinese, path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(make(task, chinese), path)


if __name__ == "__main__":
    main()
