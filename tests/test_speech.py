"""Offline speech lifecycle, audio contracts and native Qwen teacher forcing."""

import copy
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def task(kind):
    return {
        name: importlib.import_module(f"advai_notebooks.{kind}.{name}")
        for name in ("config", "model", "pipeline", "data", "cli", "publish")
    }


def configuration(kind, output):
    cfg = task(kind)["config"].load_config(ROOT / f"configs/{kind}/smoke.yaml")
    cfg.output_dir = str(output)
    return cfg


@pytest.mark.parametrize("kind", ["stt", "tts"])
def test_exact_resume_reload_and_publication(tmp_path, monkeypatch, kind):
    modules = task(kind)
    cls = getattr(modules["model"], kind.upper() + "Model")
    cfg = configuration(kind, tmp_path / "full")
    reference = modules["pipeline"].run(cfg)
    interrupted = configuration(kind, tmp_path / "interrupted")

    def stop(epoch, state):
        if epoch == 1:
            raise RuntimeError("interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        modules["pipeline"].run(interrupted, callbacks=(stop,))
    resumed = modules["pipeline"].run(interrupted, resume=Path(interrupted.output_dir) / "latest")
    assert reference["training"]["history"] == resumed["training"]["history"]
    for export in ("export", "latest_export"):
        a, b = cls.load(reference[export]), cls.load(resumed[export])
        for name, value in a.model.state_dict().items():
            assert torch.equal(value, b.model.state_dict()[name]), name
    model = cls.load(resumed["export"])
    if kind == "stt":
        audio = ROOT / "data/sample/stt/audio/one.wav"
        assert model.transcribe(audio) == cls.load(reference["export"]).transcribe(audio)
    else:
        a, b = tmp_path / "a.wav", tmp_path / "b.wav"
        model.synthesize("red", a)
        cls.load(reference["export"]).synthesize("red", b)
        assert a.read_bytes() == b.read_bytes()
        with pytest.raises(ValueError, match="already exists"):
            model.synthesize("red", a)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: pytest.fail("offline only"))
    (Path(resumed["export"]) / "private.wav").write_bytes(b"private")
    plan = modules["publish"].publish(resumed["export"], f"test/{kind}", dry_run=True)
    names = {item["path"] for item in plan["files"]}
    assert "advai_metadata.json" in names
    assert "private.wav" not in names
    if kind == "tts":
        assert "voice.safetensors" in names
    interrupted.training.learning_rate *= 2
    with pytest.raises(ValueError, match="mismatch"):
        modules["pipeline"].run(interrupted, resume=Path(interrupted.output_dir) / "latest")


@pytest.mark.parametrize("kind", ["stt", "tts"])
def test_cli_lifecycle(tmp_path, kind):
    import yaml

    modules = task(kind)
    cfg = configuration(kind, tmp_path / "run")
    file = tmp_path / "config.yaml"
    values = cfg.to_dict()
    for key in ("kind", "root"):
        values.pop(key, None)
    file.write_text(yaml.safe_dump(values))
    common = ["--config", str(file), "--root", str(ROOT)]
    modules["cli"].main(["validate", *common])
    modules["cli"].main(["run", *common])
    checkpoint = str(tmp_path / "run/export")
    modules["cli"].main(
        [
            "evaluate",
            *common,
            "--checkpoint",
            checkpoint,
            "--split",
            "test",
            "--output",
            str(tmp_path / "evaluation.json"),
        ]
    )
    modules["cli"].main(
        ["export", "--checkpoint", checkpoint, "--output", str(tmp_path / "export")]
    )
    inputs = (
        [{"audio": str(ROOT / "data/sample/stt/audio/one.wav")}]
        if kind == "stt"
        else [{"text": "red", "output_audio": str(tmp_path / "speech.wav")}]
    )
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(inputs))
    modules["cli"].main(
        [
            "transcribe" if kind == "stt" else "synthesize",
            "--checkpoint",
            checkpoint,
            "--input",
            str(input_file),
            "--output",
            str(tmp_path / "predictions.json"),
        ]
    )
    assert len(json.loads((tmp_path / "predictions.json").read_text())) == 1


@pytest.mark.parametrize("kind", ["stt", "tts"])
def test_audio_leakage_and_config_contract(tmp_path, kind):
    modules = task(kind)
    cfg = configuration(kind, tmp_path / "run")
    shutil.copytree(ROOT / "data/sample" / kind, tmp_path / "data")
    file = tmp_path / "data/samples.jsonl"
    cfg.data.samples = str(file)
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    original = copy.deepcopy(rows)
    rows[-1]["audio"] = rows[0]["audio"]
    file.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="crosses splits"):
        modules["data"].load_data(cfg)
    rows = original
    rows[0]["audio"] = "../escape.wav"
    file.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="relative"):
        modules["data"].load_data(cfg)
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  surprise: true\n")
    with pytest.raises(ValueError, match="Unknown"):
        modules["config"].load_config(bad, ROOT)
    cfg.runtime.precision = "bf16"
    with pytest.raises(ValueError, match="fp32"):
        modules["pipeline"].run(cfg)


def test_audio_validation_and_resampling(tmp_path):
    from advai_notebooks.common.audio import local_audio, read_audio, write_audio

    mono = tmp_path / "mono.wav"
    write_audio(mono, np.sin(np.arange(4000) * 0.1) * 0.1, 8000)
    samples, hashes = read_audio(mono, 16000)
    assert len(samples) == 8000 and len(hashes) == 2
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((1000, 2)), 8000)
    with pytest.raises(ValueError, match="mono"):
        read_audio(stereo, 16000)
    with pytest.raises(ValueError, match="duration"):
        read_audio(mono, 16000, max_duration=0.1)
    invalid = tmp_path / "invalid.wav"
    sf.write(invalid, np.full(1000, np.nan), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="finite"):
        read_audio(invalid, 16000)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "escape.wav").symlink_to(mono)
    with pytest.raises(ValueError, match="escaping"):
        local_audio(nested, "escape.wav", 16000)


def test_tts_reference_is_training_only_and_single_speaker(tmp_path):
    cfg = configuration("tts", tmp_path / "run")
    cfg.data.reference_audio = "audio/five.wav"
    with pytest.raises(ValueError, match="training"):
        task("tts")["data"].load_data(cfg)
    shutil.copytree(ROOT / "data/sample/tts", tmp_path / "data")
    cfg.data.samples = str(tmp_path / "data/samples.jsonl")
    rows = [json.loads(line) for line in Path(cfg.data.samples).read_text().splitlines()]
    rows[-1]["speaker_id"] = "another"
    Path(cfg.data.samples).write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="single|one"):
        task("tts")["data"].load_data(cfg)


def test_corpus_error_rates_are_not_mean_of_utterance_rates():
    from advai_notebooks.stt.metrics import error_rates as error_counts

    values = error_counts(["a", "b c d"], ["x", "b c d"], "none")
    assert values["wer"] == 0.25 and values["cer"] == 0.25
    assert error_counts(["你好"], ["您好"], "none")["cer"] == 0.5
    assert error_counts(["a"], ["a b c"], "none")["wer"] == 2
    assert error_counts([" Ａ "], ["a"], "nfkc_lower_whitespace")["wer"] == 0
    with pytest.raises(ValueError, match="empty"):
        error_counts(["   "], ["a"], "nfkc_lower_whitespace")


def test_whisper_duration_language_and_label_padding(tmp_path):
    modules = task("stt")
    cfg = configuration("stt", tmp_path / "run")
    cfg.data.max_duration = 2
    with pytest.raises(ValueError, match="1 second"):
        modules["config"].validate_config(cfg)
    cfg.model.id = "openai/whisper-tiny"
    with pytest.raises(ValueError, match="explicit.*language"):
        modules["config"].validate_config(cfg)
    cfg = configuration("stt", tmp_path / "other")
    model = modules["model"].STTModel(cfg.model)
    features = torch.zeros(16, 100)
    rows = [
        {"labels": torch.tensor([4, 2]), "input_features": features},
        {"labels": torch.tensor([5, 6, 2]), "input_features": features},
    ]
    assert model.batch(rows)["labels"].tolist() == [[4, 2, -100], [5, 6, 2]]
    assert model.loss_weight(rows) == 5


@pytest.fixture
def native_qwen():
    pytest.importorskip("qwen_tts")
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = Qwen3TTSConfig(
        tokenizer_type="qwen3_tts_tokenizer_12hz",
        tts_model_type="base",
        tts_model_size="0b6",
        tts_pad_token_id=54,
        tts_bos_token_id=55,
        tts_eos_token_id=56,
        talker_config={
            "vocab_size": 64,
            "text_vocab_size": 64,
            "hidden_size": 16,
            "text_hidden_size": 24,
            "intermediate_size": 32,
            "head_dim": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_code_groups": 16,
            "spk_id": {},
            "codec_language_id": {},
            "codec_pad_id": 32,
            "codec_bos_id": 33,
            "codec_eos_token_id": 34,
            "codec_nothink_id": 35,
            "codec_think_bos_id": 36,
            "codec_think_eos_id": 37,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "interleaved": True,
            },
            "code_predictor_config": {
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "num_code_groups": 16,
            },
        },
        speaker_encoder_config={
            "enc_dim": 16,
            "enc_channels": [16, 16, 16, 16, 48],
            "enc_res2net_scale": 2,
            "enc_se_channels": 8,
            "enc_attention_channels": 8,
        },
    )
    return Qwen3TTSForConditionalGeneration(config)


def test_native_qwen_gradients_and_next_token_alignment(native_qwen):
    from advai_notebooks.tts.qwen import native_loss, pack

    model = native_qwen
    rows = [
        {
            "text_ids": torch.tensor([1, 2, 3, 4, 5][:length]),
            "codes": torch.randint(0, 32, (frames, 16)),
            "speaker_embedding": torch.randn(16),
        }
        for length, frames in ((4, 2), (5, 3))
    ]
    embedding, labels, attention, mask, codes = pack(model, rows)
    assert attention.sum(1).tolist() == [13, 15]
    assert mask.sum(1).tolist() == [2, 3]
    # Each first code is supervised by the preceding codec BOS, not its own input.
    for i, row in enumerate(rows):
        start = len(row["text_ids"]) + 7
        assert labels[i, start - 1] == row["codes"][0, 0]
        assert labels[i, start + len(row["codes"]) - 1] == 34
    model.eval()
    output = model.talker(
        inputs_embeds=embedding,
        attention_mask=attention,
        output_hidden_states=True,
        use_cache=False,
    )
    residual_logits, _ = model.talker.forward_sub_talker_finetune(
        codes, output.hidden_states[0][-1][mask]
    )
    expected = torch.nn.functional.cross_entropy(
        output.logits.flatten(0, 1), labels.flatten()
    ) + 0.3 * torch.nn.functional.cross_entropy(
        residual_logits.flatten(0, 1), codes[:, 1:].flatten()
    )
    actual = native_loss(model, rows)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for name in (
        "codec_head.weight",
        "text_projection.linear_fc1.weight",
        "code_predictor.lm_head.0.weight",
    ):
        parameter = dict(model.talker.named_parameters())[name]
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    before = model.talker.codec_head.weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, model.talker.codec_head.weight)


def test_qwen_sdk_boundary_and_native_export_assets(tmp_path, monkeypatch, native_qwen):
    """Native weights are real; codec/SDK I/O are spies, not pretrained quality tests."""
    from qwen_tts import Qwen3TTSModel
    from safetensors.torch import load_file, save_file

    from advai_notebooks.tts.config import ModelConfig
    from advai_notebooks.tts.model import TTSModel
    from advai_notebooks.tts.publish import publish

    class Processor:
        def __call__(self, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])}

        def save_pretrained(self, path):
            (Path(path) / "tokenizer_config.json").write_text("{}")

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return [np.zeros(1000, dtype=np.float32)], 24000

    native_qwen.speech_tokenizer = SimpleNamespace(model=torch.nn.Linear(1, 1))
    native_qwen.generate_config = {"max_new_tokens": 2048, "temperature": 0.9}
    wrapped = TTSModel.__new__(TTSModel)
    wrapped.settings = ModelConfig(id="native-test", max_text_tokens=16, max_new_tokens=4)
    wrapped.device, wrapped.model = torch.device("cpu"), native_qwen
    wrapped.voice = torch.randn(16)
    wrapped.base_path = tmp_path / "base"
    codec = wrapped.base_path / "speech_tokenizer"
    codec.mkdir(parents=True)
    (codec / "config.json").write_text("{}")
    (codec / "preprocessor_config.json").write_text("{}")
    save_file({"fixture": torch.zeros(1)}, str(codec / "model.safetensors"))
    (codec / "private.wav").write_bytes(b"not for upload")
    wrapped.sdk = SimpleNamespace(
        model=native_qwen, processor=Processor(), generate_voice_clone=generate
    )
    wrapped._configure_qwen()
    assert not any(p.requires_grad for p in native_qwen.speaker_encoder.parameters())
    assert not any(p.requires_grad for p in native_qwen.speech_tokenizer.model.parameters())
    wrapped.fit_voice(np.sin(np.arange(6000, dtype=np.float32) * .1) * .1)
    assert wrapped.voice.shape == (16,) and torch.isfinite(wrapped.voice).all()
    wrapped.synthesize("hello", tmp_path / "one.wav")
    assert calls[-1]["non_streaming_mode"] is True
    assert calls[-1]["do_sample"] is False and calls[-1]["subtalker_dosample"] is False
    assert calls[-1]["voice_clone_prompt"]["ref_code"] == [None]
    assert calls[-1]["voice_clone_prompt"]["icl_mode"] == [False]
    assert "ref_audio" not in calls[-1]
    destination = wrapped.save(tmp_path / "export")
    assert (
        json.loads((destination / "generation_config.json").read_text())
        == native_qwen.generate_config
    )

    def load_sdk(path, **kwargs):
        assert kwargs["local_files_only"] is True
        assert (Path(path) / "speech_tokenizer/model.safetensors").is_file()
        native_qwen.load_state_dict(load_file(str(Path(path) / "model.safetensors")))
        return SimpleNamespace(
            model=native_qwen, processor=Processor(), generate_voice_clone=generate
        )

    monkeypatch.setattr(Qwen3TTSModel, "from_pretrained", load_sdk)
    loaded = TTSModel.load(destination)
    assert torch.equal(loaded.voice, wrapped.voice)
    loaded.synthesize("hello", tmp_path / "two.wav")
    assert (tmp_path / "one.wav").read_bytes() == (tmp_path / "two.wav").read_bytes()
    plan = publish(destination, "test/native-tts", dry_run=True)
    names = {item["path"] for item in plan["files"]}
    assert {
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/config.json",
        "speech_tokenizer/preprocessor_config.json",
        "voice.safetensors",
        "generation_config.json",
    } <= names
    assert "speech_tokenizer/private.wav" not in names


@pytest.mark.parametrize("kind", ["stt", "tts"])
def test_speech_has_no_other_task_imports(tmp_path, kind):
    source = """
import importlib, importlib.abc, sys
from pathlib import Path
kind, root, output = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
other = {"llm", "embedding", "ranker", "decision", "diffusion", "vlm", "vla", "vm", "stt", "tts"} - {kind}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == "advai_notebooks." + item or fullname.startswith("advai_notebooks." + item + ".") for item in other):
            raise AssertionError("Cross-task import: " + fullname)
        if kind == "stt" and fullname.startswith("qwen_tts"):
            raise AssertionError("STT must not depend on Qwen TTS")
sys.meta_path.insert(0, Guard())
prefix = "advai_notebooks." + kind
cfg = importlib.import_module(prefix + ".config").load_config(root / "configs" / kind / "smoke.yaml")
cfg.output_dir, cfg.training.epochs = output, 1
importlib.import_module(prefix + ".pipeline").run(cfg)
print("INDEPENDENT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", source, kind, str(ROOT), str(tmp_path / kind)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INDEPENDENT_OK" in result.stdout
