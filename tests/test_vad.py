"""Offline VAD contracts, actual pinned Silero gradients and native export parity."""

import copy
import importlib.metadata
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from advai_notebooks.common.runtime import setup
from advai_notebooks.vad.cli import main
from advai_notebooks.vad.config import ModelConfig, load_config, validate_config
from advai_notebooks.vad.data import load_data, prepare_data
from advai_notebooks.vad.losses import loss
from advai_notebooks.vad.metrics import frame_counts, interval_counts, rates
from advai_notebooks.vad.model import VADModel
from advai_notebooks.vad.pipeline import calibrate, run
from advai_notebooks.vad.publish import publish
from advai_notebooks.vad.segments import labels_for_frames, speech_intervals, validate_segments

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "data/sample/vad/audio/one.wav"


def config(output):
    cfg = load_config(ROOT / "configs/vad/smoke.yaml")
    cfg.output_dir = str(output)
    return cfg


def test_exact_resume_and_offline_publication(tmp_path, monkeypatch):
    reference = run(config(tmp_path / "full"))
    cfg = config(tmp_path / "interrupted")

    def interrupt(epoch, state):
        if epoch == 1:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run(cfg, callbacks=(interrupt,))
    resumed = run(cfg, resume=tmp_path / "interrupted/latest")
    assert reference["training"] == resumed["training"]
    assert reference["calibration"] == resumed["calibration"]
    for key in ("export", "latest_export"):
        a, b = VADModel.load(reference[key]), VADModel.load(resumed[key])
        assert a.settings == b.settings
        for name, value in a.model.state_dict().items():
            assert torch.equal(value, b.model.state_dict()[name])
        assert a.detect(AUDIO) == b.detect(AUDIO)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: pytest.fail("No network"))
    export = Path(resumed["export"])
    (export / "private.wav").write_bytes(b"private")
    (export / "training_state.pt").write_bytes(b"private")
    names = {item["path"] for item in publish(export, "example/vad", dry_run=True)["files"]}
    assert {"advai_metadata.json", "model.safetensors"} <= names
    assert not {"private.wav", "training_state.pt"} & names
    cfg.training.learning_rate *= 2
    with pytest.raises(ValueError, match="mismatch"):
        run(cfg, resume=tmp_path / "interrupted/latest")


def test_cli_lifecycle_and_safe_segmentation(tmp_path):
    import yaml

    cfg = config(tmp_path / "run")
    values = cfg.to_dict()
    values.pop("kind")
    values.pop("root")
    settings = tmp_path / "config.yaml"
    settings.write_text(yaml.safe_dump(values))
    args = ["--config", str(settings), "--root", str(ROOT)]
    main(["validate", *args])
    main(["run", *args])
    export = str(tmp_path / "run/export")
    main(
        [
            "evaluate",
            *args,
            "--checkpoint",
            export,
            "--split",
            "test",
            "--output",
            str(tmp_path / "evaluation.json"),
        ]
    )
    main(["export", "--checkpoint", export, "--output", str(tmp_path / "copy")])
    main(
        [
            "detect",
            "--checkpoint",
            export,
            "--audio",
            str(AUDIO),
            "--output",
            str(tmp_path / "detection.json"),
        ]
    )
    main(
        [
            "segment",
            "--checkpoint",
            export,
            "--audio",
            str(AUDIO),
            "--output-dir",
            str(tmp_path / "chunks"),
            "--max-chunk-duration",
            "0.4",
        ]
    )
    result = json.loads((tmp_path / "chunks/segments.json").read_text())
    assert result["chunks"]
    for chunk in result["chunks"]:
        waveform, rate = sf.read(chunk["audio"])
        assert rate == cfg.model.sample_rate
        assert len(waveform) <= 6400
        assert chunk["source_end"] - chunk["source_start"] == pytest.approx(len(waveform) / rate)
        assert "text" not in chunk
    with pytest.raises(ValueError, match="must be new"):
        VADModel.load(export).segment(AUDIO, tmp_path / "chunks")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        [{"start": -1, "end": 1}],
        [{"start": 0, "end": 3}],
        [{"start": 0.5, "end": 0.5}],
        [{"start": 0, "end": float("nan")}],
        [{"start": True, "end": 1}],
        [{"start": 0, "end": 1}, {"start": 0.5, "end": 1.5}],
    ],
)
def test_interval_contract_rejects_invalid_annotations(bad):
    with pytest.raises(ValueError):
        validate_segments(bad, 2)
    assert validate_segments([], 2) == []


def test_audio_group_content_and_path_leakage(tmp_path):
    cfg = config(tmp_path / "run")
    shutil.copytree(ROOT / "data/sample/vad", tmp_path / "data")
    cfg.data.samples = str(tmp_path / "data/samples.jsonl")
    source = Path(cfg.data.samples)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for field, value, message in (
        ("audio", rows[0]["audio"], "Duplicate"),
        ("group_id", rows[0]["group_id"], "group"),
        ("audio", "../escape.wav", "relative"),
        ("speech_segments", None, "must be a list"),
    ):
        changed = copy.deepcopy(rows)
        changed[-1][field] = value
        source.write_text("\n".join(json.dumps(row) for row in changed))
        with pytest.raises(ValueError, match=message):
            load_data(cfg)


def test_no_speech_only_split_is_explicitly_rejected(tmp_path):
    cfg = config(tmp_path / "run")
    shutil.copytree(ROOT / "data/sample/vad", tmp_path / "data")
    cfg.data.samples = str(tmp_path / "data/samples.jsonl")
    source = Path(cfg.data.samples)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        if row["split"] == "validation":
            row["speech_segments"] = []
    source.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="both speech and nonspeech"):
        load_data(cfg)


def test_frame_majority_partial_padding_and_metric_denominators():
    labels, weights = labels_for_frames([{"start": 0.016, "end": 0.048}], 800, 16000)
    assert labels.tolist() == [0, 1]
    assert weights.tolist() == [1, 0.5625]
    counts = frame_counts([1, 0, 1], [0.8, 0.8, 0.1], [1, 1, 0.5], 0.5)
    assert counts == pytest.approx([0.032, 0.032, 0.016, 0])
    assert rates(*counts)["recall"] == pytest.approx(2 / 3)
    assert rates(*counts)["precision"] == 0.5
    assert rates(*counts)["f1"] == pytest.approx(4 / 7)
    segments = interval_counts([{"start": 1, "end": 3}], [{"start": 2, "end": 4}], 5)
    assert segments.tolist() == [1, 1, 1, 2]


def test_hysteresis_silence_padding_and_eof():
    settings = ModelConfig(min_speech_duration=0.032, min_silence_duration=0.064, speech_pad=0)
    assert speech_intervals([0.1] * 8, 0.256, settings) == []
    assert speech_intervals([0.8, 0.4, 0.1, 0.1, 0.8, 0.8], 0.192, settings) == [
        {"start": 0, "end": 0.064},
        {"start": 0.128, "end": 0.192},
    ]
    assert speech_intervals([0.8, 0.1, 0.8], 0.096, settings) == [{"start": 0, "end": 0.096}]
    assert speech_intervals([0.8, 0.8, 0.1], 0.096, settings) == [{"start": 0, "end": 0.064}]
    settings = replace(settings, speech_pad=0.032)
    assert speech_intervals([0.8, 0.1, 0.1, 0.8], 0.128, settings) == [{"start": 0, "end": 0.128}]
    with pytest.raises(ValueError, match="count"):
        speech_intervals([0.8], 0.096, settings)


def test_validation_only_calibration(tmp_path):
    cfg = config(tmp_path / "run")
    model = VADModel(cfg.model, setup(cfg))
    rows, _ = load_data(cfg)
    prepared = prepare_data(rows, model, cfg)
    reference = calibrate(model, prepared, cfg)
    prepared["test"] = []
    assert reference == calibrate(model, prepared, cfg)
    assert reference["split"] == "validation"


@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_actual_silero_gradients_export_and_native_parity(tmp_path, monkeypatch, sample_rate):
    pytest.importorskip("silero_vad")
    cfg = load_config(ROOT / "configs/vad/silero.yaml")
    cfg.model.sample_rate = sample_rate
    cfg.data.samples = "data/sample/vad/samples.jsonl"
    device = setup(cfg)
    model = VADModel(cfg.model, device)
    rows, _ = load_data(cfg)
    prepared = prepare_data(rows, model, cfg)
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    model.model.train()
    optimizer = torch.optim.Adam([p for p in model.model.parameters() if p.requires_grad], lr=0.001)
    value = loss(model, prepared["train"][:2])
    assert torch.isfinite(value)
    value.backward()
    assert all(p.grad is None for p in model.model.encoder.parameters())
    assert all(p.grad is None for p in model.model.stft.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.decoder.parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(value, model.model.state_dict()[key])
        for key, value in before.items()
        if key.startswith("decoder.")
    )
    assert all(
        torch.equal(value, model.model.state_dict()[key])
        for key, value in before.items()
        if not key.startswith("decoder.")
    )
    model.model.eval()
    export = model.save(tmp_path / "export")
    assert "Silero Team" in (export / "LICENSE").read_text()
    native = torch.jit.load(str(export / "silero_vad.jit")).eval()
    waveform, _ = sf.read(AUDIO, dtype="float32")
    if sample_rate == 8000:
        from scipy.signal import resample_poly

        waveform = resample_poly(waveform, 1, 2).astype(np.float32)
    native_probs = native.audio_forward(torch.tensor(waveform).unsqueeze(0), sample_rate).flatten()
    eager = torch.tensor(model.detect(AUDIO)["probabilities"])
    assert torch.allclose(native_probs, eager, atol=1e-6, rtol=1e-5)
    monkeypatch.setattr(
        importlib.metadata, "distribution", lambda *a: pytest.fail("No package lookup")
    )
    reloaded = VADModel.load(export)
    assert reloaded.detect(AUDIO) == model.detect(AUDIO)
    plan = publish(export, "example/vad-native", dry_run=True)
    assert {"silero_vad.jit", "LICENSE"} <= {f["path"] for f in plan["files"]}


def test_config_identity_and_export_integrity(tmp_path):
    cfg = config(tmp_path / "run")
    cfg.model.id = "silero"
    with pytest.raises(ValueError, match="Pin"):
        validate_config(cfg)
    cfg = config(tmp_path / "run")
    cfg.model.exit_threshold = 0.8
    with pytest.raises(ValueError, match="lower"):
        validate_config(cfg)
    cfg = config(tmp_path / "run")
    cfg.evaluation.thresholds = [float("nan")]
    with pytest.raises(ValueError):
        validate_config(cfg)
    export = VADModel(ModelConfig()).save(tmp_path / "export")
    weights = export / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        VADModel.load(export)


def test_no_speech_never_falls_back_to_whole_recording(tmp_path):
    model = VADModel(ModelConfig())
    with torch.no_grad():
        model.model.decoder.decoder[2].weight.zero_()
        model.model.decoder.decoder[2].bias.fill_(-20)
    assert model.detect(AUDIO)["speech_segments"] == []
    result = model.segment(AUDIO, tmp_path / "empty")
    assert result["chunks"] == []
    assert list((tmp_path / "empty").iterdir()) == [tmp_path / "empty/segments.json"]
    with pytest.raises(ValueError, match="one audio sample"):
        model.segment(AUDIO, tmp_path / "invalid", max_chunk_duration=1e-10)
    assert not (tmp_path / "invalid").exists()


def test_help_lists_the_offline_commands(capsys):
    main(["--help"])
    text = capsys.readouterr().out
    assert "segment" in text and "detect-batch" in text


def test_fixture_regeneration_is_exact_and_never_overwrites_differing_audio(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vad_fixtures", ROOT / "scripts/prepare_vad_fixtures.py"
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    fixtures.prepare(tmp_path)
    files = sorted((tmp_path / "data/sample/vad/audio").glob("*.wav"))
    original = [path.read_bytes() for path in files]
    fixtures.prepare(tmp_path)
    assert original == [path.read_bytes() for path in files]
    assert all(
        path.read_bytes() == (ROOT / "data/sample/vad/audio" / path.name).read_bytes()
        for path in files
    )
    sf.write(files[0], np.zeros(32000), 16000, subtype="PCM_16")
    changed = files[0].read_bytes()
    with pytest.raises(ValueError, match="overwrite differing"):
        fixtures.prepare(tmp_path)
    assert files[0].read_bytes() == changed


def test_native_silero_epoch_resume(tmp_path):
    pytest.importorskip("silero_vad")
    cfg = load_config(ROOT / "configs/vad/silero.yaml")
    cfg.data.samples = "data/sample/vad/samples.jsonl"
    cfg.training.epochs = 2
    cfg.output_dir = str(tmp_path / "full")
    full = run(cfg)
    cfg.output_dir = str(tmp_path / "resumed")

    def interrupt(epoch, state):
        if epoch == 1:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run(cfg, callbacks=(interrupt,))
    resumed = run(cfg, resume=tmp_path / "resumed/latest")
    assert full["training"] == resumed["training"]
    assert full["calibration"] == resumed["calibration"]
    assert VADModel.load(full["export"]).detect(AUDIO) == VADModel.load(resumed["export"]).detect(
        AUDIO
    )
    native = torch.jit.load(str(tmp_path / "resumed/export/silero_vad.jit"))
    original = torch.jit.load(
        str(
            importlib.metadata.distribution("silero-vad").locate_file(
                "silero_vad/data/silero_vad.jit"
            )
        )
    )
    assert all(
        torch.equal(value, original._model_8k.state_dict()[key])
        for key, value in native._model_8k.state_dict().items()
    )
    (tmp_path / "resumed/export/LICENSE").unlink()
    with pytest.raises(ValueError, match="retain"):
        VADModel.load(resumed["export"])
