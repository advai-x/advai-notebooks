"""Notebook-native diagnostic timeline, not a speech-quality benchmark."""

import numpy as np

from advai_notebooks.common.audio import read_audio


def plot_detection(result, settings, reference=None):
    import matplotlib.pyplot as plt

    waveform, _ = read_audio(
        result["audio"],
        result["sample_rate"],
        min_duration=0.001,
        max_duration=settings.max_inference_duration,
    )
    figure, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, layout="constrained")
    step = max(1, len(waveform) // 10000)
    axes[0].plot(
        np.arange(0, len(waveform), step) / result["sample_rate"],
        waveform[::step],
        color="#24689c",
        linewidth=0.8,
    )
    axes[0].set(title="Input audio", ylabel="PCM amplitude", ylim=(-1, 1))
    times = np.minimum((np.arange(len(result["probabilities"])) + 0.5) * 0.032, result["duration"])
    axes[1].plot(times, result["probabilities"], color="#24689c", label="Speech probability")
    axes[1].axhline(settings.threshold, color="#444444", linestyle="--", label="Enter threshold")
    axes[1].axhline(settings.exit_threshold, color="#888888", linestyle=":", label="Exit threshold")
    axes[1].set(ylabel="Probability", ylim=(0, 1))
    axes[1].legend(loc="upper right", fontsize=9)
    for segment in result["speech_segments"]:
        axes[2].broken_barh(
            [(segment["start"], segment["end"] - segment["start"])],
            (0.65, 0.25),
            facecolors="#24689c",
            edgecolors="#163b58",
        )
    if reference is not None:
        for segment in reference:
            axes[2].broken_barh(
                [(segment["start"], segment["end"] - segment["start"])],
                (0.15, 0.25),
                facecolors="none",
                edgecolors="#8b642b",
                hatch="///",
            )
    axes[2].set(
        yticks=[0.275, 0.775],
        yticklabels=["Reference" if reference is not None else "Reference unavailable", "Detected"],
        ylim=(0, 1),
        xlabel="Time (seconds)",
        xlim=(0, result["duration"]),
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    return figure
