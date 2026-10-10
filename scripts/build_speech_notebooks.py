"""Regenerate paired speech tutorial scaffolds with nbformat (clears saved outputs)."""

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def make(kind, chinese):
    cells = []

    def prose(en, zh):
        cells.append(nbf.v4.new_markdown_cell(zh if chinese else en))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    title = "Whisper speech-to-text" if kind == "stt" else "Single-speaker text-to-speech"
    title_zh = "Whisper 语音转文字" if kind == "stt" else "单说话人文字转语音"
    prose(
        f"# {title}\n\nEnglish | [简体中文](01_finetune.zh-CN.ipynb)\n\n## Goal\n\n"
        "Walk through validated audio, epoch-boundary recovery, held-out evaluation, "
        "export/reload and inference using the same implementation as the CLI. "
        "The default is an offline CPU smoke run, not evidence of speech quality.",
        f"# {title_zh}\n\n[English](01_finetune.ipynb) | 简体中文\n\n## Goal / 目标\n\n"
        "使用与 CLI 相同的实现，完成音频校验、epoch 边界恢复、held-out 评估、导出重载与推理。"
        "默认执行离线 CPU smoke，不代表真实语音质量。",
    )
    caveat = (
        "The smoke backend is a randomly initialized native Whisper with a synthetic tokenizer. "
        "Sine waves do not contain their text labels."
        if kind == "stt"
        else "The smoke backend is a text-conditioned GRU over quantized PCM, **not Qwen**. "
        "Sine waves and its generated waveform are not meaningful speech."
    )
    caveat_zh = (
        "Smoke 使用随机初始化原生 Whisper 与合成 tokenizer。正弦音不包含对应文字。"
        if kind == "stt"
        else "Smoke 是粗量化 PCM 的文本条件 GRU，**不是 Qwen**。正弦音与生成波形不是有意义的语音。"
    )
    prose(
        f"> {caveat}\n\n## Setup\n\nUse a fresh speech environment: Transformers 4.57.3 "
        "cannot be mixed with tracks pinning 5.18.0. The next cell installs dependencies only "
        "in Colab. Locally install the matching extra before opening this notebook. Restart a "
        "Colab runtime if installing replaces packages already loaded. Keep long experiments on Drive.",
        f"> {caveat_zh}\n\n## Setup / 环境准备\n\n语音使用独立环境：Transformers 4.57.3 "
        "不能与固定 5.18.0 的任务混装。下一格仅在 Colab 安装依赖，本地需预先安装对应 extra。"
        "若替换已加载依赖请重启 runtime；长期实验使用 Drive 持久化。",
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
                            f"{{ROOT}}[{kind},notebooks]"], check=True)
        else:
            ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
                        if (path / "pyproject.toml").exists())
        print("Repository and speech environment ready.")
    """)
    prose(
        "### Imports and parameters\n\nThe seed, data, budgets and optimizer come from the checked-in YAML. "
        "A fresh output directory makes repeated notebook execution safe.",
        "### 导入与参数\n\nSeed、数据、预算与 optimizer 来自仓库 YAML；每次新建输出目录，支持完整重跑。",
    )
    code(f"""
        import json
        import uuid
        import warnings
        import torch
        from IPython.display import Audio, Markdown, display
        from advai_notebooks.{kind}.config import load_config
        from advai_notebooks.{kind}.data import load_data, prepare_data
        from advai_notebooks.{kind}.model import {kind.upper()}Model
        from advai_notebooks.{kind}.pipeline import run
        from advai_notebooks.{kind}.publish import publish
        from advai_notebooks.common.runtime import setup
        from transformers.utils import logging

        logging.set_verbosity_error()
        warnings.filterwarnings("ignore", message="Moving the following attributes.*")
        cfg = load_config(ROOT / "configs/{kind}/smoke.yaml", ROOT)
        cfg.output_dir = f"outputs/{kind}/notebook-{{uuid.uuid4().hex[:8]}}"
        print(json.dumps({{"model": cfg.model.id, "seed": cfg.training.seed,
                          "epochs": cfg.training.epochs, "device": cfg.runtime.device}}, indent=2))
    """)
    prose(
        "## Steps\n\n### 1. Validate and prepare audio\n\nGroups and identical decoded audio "
        "cannot cross splits. Clips must be finite mono audio within duration limits. "
        "Inspect a bounded manifest preview and listen to the synthetic input.",
        "## Steps / 步骤\n\n### 1. 校验与准备音频\n\nGroup 和相同解码音频不能跨 split；"
        "片段须为有限单声道且满足时长。检查有限 manifest 样例并试听合成输入。",
    )
    code(
        """
        rows, summary = load_data(cfg)
        validator = MODEL_CLASS(cfg.model, setup(cfg), cfg.root, cfg.training)
        prepared = prepare_data(rows, validator, cfg, fit=True)
        print(json.dumps({"splits": summary["splits"], "sample_rate": summary["sample_rate"]}, indent=2))
        for row in rows[:2]:
            print({key: row[key] for key in ("id", "group_id", "split", "text")})
        display(Audio(rows[0]["waveform"], rate=summary["sample_rate"]))
        del validator, prepared
    """.replace("MODEL_CLASS", kind.upper() + "Model")
    )
    prose(
        "### 2. Train and deliberately interrupt after epoch one\n\nThe pipeline saves a baseline "
        "and atomic checkpoints before invoking the callback. This simulates a disconnected "
        "runtime at an epoch boundary without losing the optimizer/RNG state.",
        "### 2. 训练并在 epoch one 后模拟中断\n\nPipeline 在调用 callback 前保存基线与原子 "
        "checkpoint，模拟 epoch 边界断连，同时保留 optimizer/RNG 状态。",
    )
    code("""
        class TutorialInterruption(Exception):
            pass

        def interrupt_after_first_epoch(epoch, state):
            if epoch == 1:
                raise TutorialInterruption("Saved epoch-one checkpoint.")

        try:
            run(cfg, callbacks=(interrupt_after_first_epoch,))
        except TutorialInterruption as error:
            print(error)
        latest = cfg.path(cfg.output_dir) / "latest"
        assert (latest / "state.json").is_file()
    """)
    prose(
        "### 3. Resume and inspect held-out metrics\n\nResume requires unchanged config, data, source "
        "and environment. Validation chooses the lowest metric, including the baseline; test is report-only.",
        "### 3. 恢复并检查 held-out 指标\n\n恢复须保持配置、数据、源码和环境不变。"
        "Validation 选择最小指标，包含基线；test 仅报告。",
    )
    code("""
        report = run(cfg, resume=latest)
        assert report["training"]["epoch"] == cfg.training.epochs
        metrics = {"baseline_test": report["baseline_test"]["metrics"],
                   "selected_test": report["finetuned_test"]["metrics"],
                   "selected_epoch": report["training"]["best_epoch"]}
        print(json.dumps(metrics, indent=2))
    """)
    prose(
        "The values above are a smoke diagnostic, not a benchmark. STT WER/CER can exceed 1 "
        "because insertions count; unsegmented Chinese is better assessed with CER. TTS loss "
        "is teacher forcing, not a naturalness score. A baseline can remain the selected model.",
        "上述数值只诊断 smoke，不是 benchmark。STT 因插入错误 WER/CER 可大于 1；未分词中文优先 CER。"
        "TTS loss 是 teacher forcing 而非自然度评分；基线也可能被选为最佳模型。",
    )
    prose(
        "### 4. Reload and infer\n\nLoad the selected export without the original training dataset. "
        "No weights or audio are uploaded.",
        "### 4. 重载与推理\n\n加载最佳导出包，不依赖原训练数据。不会上传权重或音频。",
    )
    code(f"selected = {kind.upper()}Model.load(report['export'], 'cpu')")
    if kind == "stt":
        code("""
            transcription = selected.transcribe(rows[-1]["audio"])
            print({"reference": rows[-1]["text"], "transcription": transcription})
        """)
    else:
        code("""
            listening_text = "red"
            generated = cfg.path(cfg.output_dir) / "listening/selected.wav"
            baseline_audio = cfg.path(cfg.output_dir) / "listening/baseline.wav"
            baseline = TTSModel.load(cfg.path(cfg.output_dir) / "baseline", "cpu")
            baseline.synthesize(listening_text, baseline_audio)
            sample = selected.synthesize(listening_text, generated)
            print({"sample_rate": sample["sample_rate"], "duration_seconds": sample["duration"]})
            display(Markdown("**Baseline / 基线 — synthetic waveform**"))
            display(Audio(filename=str(baseline_audio)))
            display(Markdown("**Selected / 最佳 — synthetic waveform, not real speech**"))
            display(Audio(filename=str(generated)))
        """)
    prose(
        "## Checks\n\nConfirm metadata/native assets, reload parity and an offline whitelist publication "
        "preview. Only publish a voice embedding with explicit speaker permission; it is derived biometric data.",
        "## Checks / 检查\n\n检查原生导出与元数据、重载一致性及离线白名单发布预览。"
        "Speaker embedding 是衍生生物特征，只有获得明确授权后才发布。",
    )
    code(f"""
        export = Path(report["export"])
        assert (export / "advai_metadata.json").is_file()
        reloaded = {kind.upper()}Model.load(export, "cpu")
        for name, value in selected.model.state_dict().items():
            assert torch.equal(value, reloaded.model.state_dict()[name])
        plan = publish(export, "example/{kind}-smoke", dry_run=True)
        names = [item["path"] for item in plan["files"]]
        assert not any(name.endswith((".wav", ".jsonl", ".pt")) for name in names)
        print({{"reload_equal": True, "publish": plan["status"], "files": len(names)}})
    """)
    preset = "hf.yaml" if kind == "stt" else "qwen3.yaml"
    prose(
        f"## Next Steps\n\nSee the [full guide](../../docs/{kind}.md). Copy "
        f"`configs/{kind}/{preset}` to `configs/local/`, pin the model and use your own "
        "consented speech corpus with independent sessions and held-out text. "
        "The real preset requires target-hardware validation; do not train a pretrained speech "
        "model on these tones. This executed tutorial does not establish GPU fine-tuning, "
        "free-Colab fit or real speech quality. Use fixed unseen texts for baseline/selected TTS listening.",
        f"## Next Steps / 下一步\n\n阅读[完整指南](../../docs/{kind}.zh-CN.md)，复制 "
        f"`configs/{kind}/{preset}` 到 `configs/local/`，固定模型版本并使用自己的授权语音数据，"
        "按独立会话划分与保留未见文本。真实配置需目标硬件验证，不要在这些正弦音上训练预训练语音模型。"
        "已执行教程不证明 GPU 微调、免费 Colab 显存适配或语音质量。TTS 使用相同未见文本对比基线与最佳模型听感。",
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
    parser.add_argument(
        "--replace", action="store_true", help="Explicitly replace existing notebooks"
    )
    args = parser.parse_args()
    paths = [
        (
            kind,
            chinese,
            ROOT
            / "notebooks"
            / kind
            / ("01_finetune.zh-CN.ipynb" if chinese else "01_finetune.ipynb"),
        )
        for kind in ("stt", "tts")
        for chinese in (False, True)
    ]
    if not args.replace and any(path.exists() for _, _, path in paths):
        raise ValueError("Notebooks exist; --replace explicitly clears saved outputs")
    for kind, chinese, path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(make(kind, chinese), path)


if __name__ == "__main__":
    main()
