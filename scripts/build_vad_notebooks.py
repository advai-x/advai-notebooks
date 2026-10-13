"""Create bilingual VAD tutorial scaffolds; explicit replacement clears outputs."""

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def make(tutorial, chinese):
    cells = []

    def prose(en, zh):
        cells.append(nbf.v4.new_markdown_cell(zh if chinese else en))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    stem = "01_finetune" if tutorial == "training" else "02_detect_and_segment"
    title = (
        "VAD: decoder fine-tuning"
        if tutorial == "training"
        else "VAD: offline detection and segmentation"
    )
    title_zh = "VAD：解码器微调" if tutorial == "training" else "VAD：离线检测与切分"
    prose(
        f"# {title}\n\nEnglish | [简体中文]({stem}.zh-CN.ipynb)\n\n## Goal\n\n"
        "Use the same implementation as the CLI. The default synthetic CPU tutorial verifies "
        "mechanics, not speech quality. Real Silero adapts only its selected decoder; this "
        "tutorial's random small feature projection is **not pretrained Silero**.",
        f"# {title_zh}\n\n[English]({stem}.ipynb) | 简体中文\n\n## Goal / 目标\n\n"
        "使用与 CLI 相同的实现。默认合成 CPU 教程验证工程流程，不代表语音质量。真实 Silero "
        "仅微调选中 decoder；本教程的小型随机特征映射**不是预训练 Silero**。",
    )
    prose(
        "## Setup\n\nLocally install `.[vad,notebooks,publish]` in a matching kernel. The next cell "
        "installs only in Colab. VAD requires no Transformers dependency. Keep private recordings "
        "and long-running outputs on persistent storage. No uploads occur in this notebook.",
        "## Setup / 环境\n\n本地在对应 kernel 安装 `.[vad,notebooks,publish]`。下一格只在 Colab 安装。"
        "VAD 不依赖 Transformers。私有录音和长时间任务的输出放在持久存储；Notebook 不上传数据。",
    )
    code("""
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
                            f"{ROOT}[vad,notebooks,publish]"], check=True)
        else:
            ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
                        if (path / "pyproject.toml").is_file())
        print("Repository and VAD environment ready.")
    """)
    prose(
        "### Parameters and imports\n\nA fresh output path makes the tutorial safe to rerun. "
        "The YAML records the seed, thresholds, sample rate and optimizer.",
        "### 参数与导入\n\n全新输出路径支持完整重跑。YAML 记录 seed、阈值、采样率和 optimizer。",
    )
    code(f"""
        import json
        import uuid
        import torch
        import matplotlib.pyplot as plt
        from IPython.display import Audio, Markdown, display
        from advai_notebooks.vad.config import load_config
        from advai_notebooks.vad.data import load_data, prepare_data
        from advai_notebooks.vad.model import VADModel
        from advai_notebooks.vad.pipeline import run
        from advai_notebooks.vad.publish import publish
        from advai_notebooks.vad.visualize import plot_detection
        from advai_notebooks.common.runtime import setup

        plt.rcParams.update({{"font.family": "DejaVu Sans", "font.size": 11}})
        cfg = load_config(ROOT / "configs/vad/smoke.yaml", ROOT)
        cfg.output_dir = f"outputs/vad/{stem}-{{uuid.uuid4().hex[:8]}}"
        print({{"backend": cfg.model.id, "sample_rate": cfg.model.sample_rate,
               "seed": cfg.training.seed, "frame_seconds": 0.032}})
    """)
    if tutorial == "training":
        prose(
            "## Steps\n\n### 1. Validate audio and full interval annotations\n\nEvery unannotated "
            "region is negative, not unknown. `[]` explicitly means no speech. Split original "
            "recordings before clipping. These fixtures contain tones, not human speech.",
            "## Steps / 步骤\n\n### 1. 校验音频与完整区间标注\n\n所有未标注区域视为负例而非未知。"
            "`[]` 明确表示无语音。先按原录音划分再切片；这些样例是正弦音，不是真人语音。",
        )
        code("""
            rows, summary = load_data(cfg)
            validator = VADModel(cfg.model, setup(cfg))
            prepared = prepare_data(rows, validator, cfg)
            print({"splits": summary["splits"],
                   "prepared": {key: len(value) for key, value in prepared.items()}})
            for row in rows[:3]:
                print({"id": row["id"], "split": row["split"], "speech_segments": row["speech_segments"]})
            display(Audio(rows[0]["waveform"], rate=cfg.model.sample_rate))
            del validator, prepared
        """)
        prose(
            "### 2. Train and simulate an epoch-boundary interruption\n\nThe saved checkpoint "
            "includes optimizer and RNG state. Frozen features receive no gradients.",
            "### 2. 训练并模拟 epoch 边界中断\n\nCheckpoint 包含 optimizer 与 RNG；冻结特征不接收梯度。",
        )
        code("""
            class TutorialInterruption(Exception):
                pass

            def stop_after_first_epoch(epoch, state):
                if epoch == 1:
                    raise TutorialInterruption("Epoch-one checkpoint saved.")

            try:
                run(cfg, callbacks=(stop_after_first_epoch,))
            except TutorialInterruption as error:
                print(error)
            latest = cfg.path(cfg.output_dir) / "latest"
            assert (latest / "state.json").is_file()
        """)
        prose(
            "### 3. Resume, select and calibrate using validation only\n\nBCE selects the weights; "
            "validation interval F1 selects the threshold grid candidate. Test is report-only. "
            "The table compares weights under the same selected policy, not a speech benchmark.",
            "### 3. 恢复、选模与仅验证集校准\n\nBCE 选权重，验证集区间 F1 选阈值网格；test 仅报告。"
            "表格在同一选中政策下对比权重，不是语音 benchmark。",
        )
        code("""
            report = run(cfg, resume=latest)
            assert report["training"]["epoch"] == cfg.training.epochs
            assert report["calibration"]["split"] == "validation"
            baseline = report["baseline_test"]["metrics"]
            selected_metrics = report["finetuned_test"]["metrics"]
            table = "| Metric | Baseline | Selected |\\n| --- | ---: | ---: |\\n"
            for metric in ("bce", "precision", "recall", "f1", "miss_rate", "false_alarm_rate", "segment_f1"):
                table += f"| {metric} | {baseline[metric]:.4f} | {selected_metrics[metric]:.4f} |\\n"
            display(Markdown(table))
            print({"selected_epoch": report["training"]["best_epoch"],
                   "calibration": report["calibration"]["selected"]})
        """)
        prose(
            "### 4. Reload and inspect the time axis\n\nThe upper row shows waveform amplitude, "
            "the middle shows probability and thresholds, and the bottom compares detected "
            "intervals with synthetic labels. Differences demonstrate why real held-out evaluation matters.",
            "### 4. 重载并检查时间轴\n\n上行为波形，中行为概率与阈值，下行为检测区间与合成标签。"
            "差异说明必须在真实 held-out 数据上评估。",
        )
        code("""
            model = VADModel.load(report["export"], "cpu")
            detection = model.detect(rows[-1]["audio"])
            figure = plot_detection(detection, model.settings, rows[-1]["speech_segments"])
            figure.suptitle("Synthetic VAD smoke — not a speech-quality benchmark")
            figure.savefig(cfg.path(cfg.output_dir) / "timeline.png", dpi=140)
            plt.show()
        """)
        prose(
            "## Checks\n\nVerify reload parity and an offline publication preview. Neither "
            "recordings nor training state belong in an inference package.",
            "## Checks / 检查\n\n检查重载一致性与离线发布预览；推理包不含录音或训练状态。",
        )
        code("""
            reloaded = VADModel.load(report["export"], "cpu")
            assert model.detect(rows[-1]["audio"]) == reloaded.detect(rows[-1]["audio"])
            for name, value in model.model.state_dict().items():
                assert torch.equal(value, reloaded.model.state_dict()[name])
            plan = publish(report["export"], "example/vad-smoke", dry_run=True)
            names = [item["path"] for item in plan["files"]]
            assert not any(name.endswith((".wav", ".jsonl", ".pt")) for name in names)
            print({"reload_equal": True, "publish": plan["status"], "files": len(names)})
        """)
    else:
        prose(
            "## Steps\n\n### 1. Choose an inference export\n\nThe default creates a tiny "
            "synthetic demonstration export. Set `CHECKPOINT` to your own trusted VAD export to "
            "skip training. Real native inference can also initialize `VADModel` directly from "
            "the pinned `silero.yaml` model settings; training data is not needed for detection.",
            "## Steps / 步骤\n\n### 1. 选择推理导出\n\n默认创建 tiny 合成示例。设置 `CHECKPOINT` "
            "为自己的可信 VAD 导出可跳过训练。真实原生推理也可直接用固定 `silero.yaml` 模型参数"
            "初始化 `VADModel`；检测不需要训练数据。",
        )
        code("""
            CHECKPOINT = None  # Set to a trusted native export for real recordings.
            if CHECKPOINT is None:
                report = run(cfg)
                CHECKPOINT = report["export"]
            model = VADModel.load(CHECKPOINT, "cpu")
            AUDIO_PATH = ROOT / "data/sample/vad/audio/one.wav"
            REFERENCE = [{"start": 0.32, "end": 0.8}, {"start": 1.12, "end": 1.6}]
            # Change AUDIO_PATH and REFERENCE together; use None if annotations are unavailable.
            print({"backend": model.settings.id, "threshold": model.settings.threshold,
                   "exit_threshold": model.settings.exit_threshold})
        """)
        prose(
            "### 2. Detect and visualize\n\nProbabilities are not transcripts. Hysteresis, "
            "short-silence merging, duration filtering and padding all affect boundaries. "
            "This plot uses synthetic tones and cannot demonstrate real speech quality.",
            "### 2. 检测与可视化\n\n概率不是转录。滞回、短静音合并、时长过滤与 padding 共同影响边界。"
            "图中的正弦音不能证明真实语音质量。",
        )
        code("""
            detection = model.detect(AUDIO_PATH)
            print({"duration_seconds": detection["duration"],
                   "speech_segments": detection["speech_segments"]})
            figure = plot_detection(detection, model.settings, REFERENCE)
            figure.suptitle("Offline VAD diagnostic — validate on your own recordings")
            cfg.path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            figure.savefig(cfg.path(cfg.output_dir) / "timeline.png", dpi=140)
            plt.show()
        """)
        prose(
            "### 3. Write bounded chunks with original offsets\n\nA new destination protects "
            "existing audio. This demonstration uses 0.4-second chunks to expose offsets; "
            "real Whisper inference can use a limit up to 30 seconds. No speech yields no chunks.",
            "### 3. 切分并保留源时间偏移\n\n新目录保护已有音频。示例用 0.4 秒以展示偏移；"
            "真实 Whisper 推理可设最多 30 秒。无语音时不生成切片。",
        )
        code("""
            result = model.segment(AUDIO_PATH, cfg.path(cfg.output_dir) / "chunks",
                                   max_chunk_duration=0.4)
            print({"chunks": len(result["chunks"]), "contains_transcript_labels": False})
            for chunk in result["chunks"][:5]:
                print({key: round(chunk[key], 4) for key in ("source_start", "source_end", "duration")})
            if result["chunks"]:
                display(Audio(filename=result["chunks"][0]["audio"]))
        """)
        prose(
            "## Checks\n\nCheck source offsets, chunk duration and safe reruns. The chunk manifest "
            "contains no transcript. Align labels before using it for supervised STT training.",
            "## Checks / 检查\n\n检查偏移、时长与重跑保护。Manifest 不生成文字标签；监督 STT 训练前须对齐。",
        )
        code("""
            for chunk in result["chunks"]:
                assert 0 <= chunk["source_start"] < chunk["source_end"] <= result["duration"]
                assert chunk["duration"] <= 0.4 + 1 / model.settings.sample_rate
                assert "text" not in chunk
            try:
                model.segment(AUDIO_PATH, cfg.path(cfg.output_dir) / "chunks")
            except ValueError as error:
                print("Existing destination correctly refused.")
            else:
                raise AssertionError("Existing chunks must never be overwritten")
        """)
    prose(
        "## Next Steps\n\nRead the [full guide](../../docs/vad.md). For real training, copy "
        "`configs/vad/silero.yaml` into ignored `configs/local/`, use consented fully annotated "
        "audio and independent recording groups. Do not train the real model on these tones. "
        "For STT inference, transcribe chunks and merge using source offsets. For TTS, inspect "
        "leading/trailing silence without automatically removing sentence-internal pauses. "
        "GPU/Colab fit, real-time performance and target-domain speech quality are not established here.",
        "## Next Steps / 下一步\n\n阅读[完整指南](../../docs/vad.zh-CN.md)。真实训练复制 "
        "`configs/vad/silero.yaml` 到被忽略的 `configs/local/`，使用授权、完整标注音频，按原录音分组。"
        "不要用正弦音训练真实模型。STT 推理按源偏移合并转写；TTS 可检查首尾静音，但不自动删除句中停顿。"
        "这里不证明 GPU/Colab 适配、实时性能或目标场景语音质量。",
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
            tutorial,
            chinese,
            ROOT / "notebooks/vad" / (stem + (".zh-CN" if chinese else "") + ".ipynb"),
        )
        for tutorial, stem in (("training", "01_finetune"), ("inference", "02_detect_and_segment"))
        for chinese in (False, True)
    ]
    if not args.replace and any(path.exists() for _, _, path in paths):
        raise ValueError("Notebooks exist; --replace explicitly clears saved outputs")
    for tutorial, chinese, path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(make(tutorial, chinese), path)


if __name__ == "__main__":
    main()
