from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_NAME = "Edit With My Voice"
DEFAULT_REF_TEXT = "Paste the exact words spoken in the reference clip here."
DEFAULT_WORKSPACE = Path(os.environ.get("EWMW_WORKSPACE", Path.cwd() / "workspace")).resolve()

ITALIAN_MODEL_REPO = "alien79/F5-TTS-italian"
ITALIAN_MODEL_CKPT = "model_159600.safetensors"
ITALIAN_MODEL_VOCAB = "vocab.txt"

MODEL_PRESETS: dict[str, dict[str, str | None]] = {
    "F5-TTS base (multilingual/general)": {
        "model": "F5TTS_v1_Base",
        "repo": None,
        "ckpt_file": None,
        "vocab_file": None,
        "note": "Uses the default F5-TTS model downloaded by the f5-tts package.",
    },
    "Italian F5-TTS — alien79/F5-TTS-italian": {
        "model": "F5TTS_Base",
        "repo": ITALIAN_MODEL_REPO,
        "ckpt_file": ITALIAN_MODEL_CKPT,
        "vocab_file": ITALIAN_MODEL_VOCAB,
        "note": "Italian-only fine-tune. First use downloads checkpoint and vocab from Hugging Face.",
    },
}

DEFAULT_MODEL_PRESET = "F5-TTS base (multilingual/general)"


@dataclass(frozen=True)
class Workspace:
    root: Path = DEFAULT_WORKSPACE

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    def ensure(self) -> None:
        for folder in (self.input_dir, self.work_dir, self.output_dir):
            folder.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], timeout: int | None = None) -> str:
    process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if process.returncode != 0:
        tail = process.stdout[-5000:] if process.stdout else ""
        raise RuntimeError("Command failed:\n" + " ".join(map(str, cmd)) + "\n\n" + tail)
    return process.stdout


def safe_name(value: str | Path, default: str = "file") -> str:
    name = Path(str(value)).name.replace(" ", "_")
    cleaned = "".join(c for c in name if c.isalnum() or c in "._-").strip("._-")
    return cleaned or default


def ts_to_seconds(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        raise ValueError("Timestamp is empty")
    if ":" not in text:
        return float(text)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Invalid timestamp: {value}")


def seconds_to_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes_int = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes_int:02d}:{sec:06.3f}"
    return f"{minutes_int:02d}:{sec:06.3f}"


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and restart the terminal. "
            "On Windows you can try: winget install --id Gyan.FFmpeg -e"
        )
    return path


def ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe was not found. It is installed together with FFmpeg.")
    return path


def f5_cli_path() -> str:
    env_path = os.environ.get("F5_TTS_BIN")
    if env_path and Path(env_path).exists():
        return env_path
    for name in ("f5-tts_infer-cli", "f5-tts_infer-cli.exe"):
        found = shutil.which(name)
        if found:
            return found
    # Helpful fallback for users launching without activating the venv.
    local_candidates = [
        Path.cwd() / ".venv" / "Scripts" / "f5-tts_infer-cli.exe",
        Path.cwd() / ".venv" / "bin" / "f5-tts_infer-cli",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "F5-TTS CLI was not found. Run the setup script first, or set F5_TTS_BIN to f5-tts_infer-cli."
    )


def detect_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def ffprobe_duration(path: str | Path) -> float:
    out = run([
        ffprobe_path(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ])
    return float(out.strip())


def load_mono(path: str | Path, sr: int | None = None):
    import numpy as np
    import soundfile as sf

    y, native_sr = sf.read(str(path), always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr and native_sr != sr:
        import librosa

        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr).astype(np.float32)
        native_sr = sr
    return y, native_sr


def write_wav(path: str | Path, y, sr: int) -> str:
    import numpy as np
    import soundfile as sf

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
    return str(out)


def rms(x) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def trim_audio(path: str | Path, top_db: float = 35.0) -> str:
    import librosa

    y, sr = load_mono(path)
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    out = Path(path).with_name(Path(path).stem + "_trim.wav")
    return write_wav(out, trimmed, sr)


def prepare_source(src_file: str | None, workspace: Workspace | None = None) -> tuple[str, str, str]:
    if not src_file:
        raise ValueError("Upload an audio or video file first.")
    workspace = workspace or Workspace()
    workspace.ensure()
    src = Path(src_file)
    copied = workspace.input_dir / f"{int(time.time())}_{safe_name(src)}"
    shutil.copy2(src, copied)
    wav = workspace.work_dir / "current_source.wav"
    run([
        ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(copied),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "1",
        str(wav),
    ], timeout=240)
    duration = ffprobe_duration(wav)
    msg = f"Prepared 48 kHz mono WAV. Duration: {seconds_to_ts(duration)}. Workspace: {workspace.root}"
    return str(wav), str(wav), msg


def extract_clip(source_wav: str | None, start: str, end: str, name: str, workspace: Workspace | None = None) -> tuple[str, str]:
    if not source_wav:
        raise ValueError("Prepare a source file first.")
    workspace = workspace or Workspace()
    workspace.ensure()
    s = ts_to_seconds(start)
    e = ts_to_seconds(end)
    if e <= s:
        raise ValueError("End must be greater than start.")
    out_name = safe_name(name or f"clip_{seconds_to_ts(s)}_{seconds_to_ts(e)}", default="clip")
    if not out_name.lower().endswith(".wav"):
        out_name += ".wav"
    out = workspace.work_dir / out_name
    run([
        ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{s:.3f}",
        "-to",
        f"{e:.3f}",
        "-i",
        source_wav,
        "-ar",
        "48000",
        "-ac",
        "1",
        str(out),
    ], timeout=90)
    return str(out), f"Extracted {seconds_to_ts(e - s)} to {out}"


def normalize_text_for_model(text: str, model_preset: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    if model_preset.startswith("Italian") and cleaned and cleaned[-1] not in ".!?…":
        cleaned += "."
    return cleaned


def resolve_model_preset(model_preset: str) -> tuple[list[str], str]:
    from huggingface_hub import hf_hub_download

    preset = MODEL_PRESETS.get(model_preset) or MODEL_PRESETS[DEFAULT_MODEL_PRESET]
    args = ["--model", str(preset["model"])]
    note = str(preset["note"])
    if preset.get("repo"):
        repo = str(preset["repo"])
        ckpt_path = hf_hub_download(repo_id=repo, filename=str(preset["ckpt_file"]))
        vocab_path = hf_hub_download(repo_id=repo, filename=str(preset["vocab_file"]))
        args.extend(["--ckpt_file", ckpt_path, "--vocab_file", vocab_path])
        note += f"\nCheckpoint: {ckpt_path}\nVocab: {vocab_path}"
    return args, note


def synthesize_f5(
    ref_audio: str | None,
    ref_text: str,
    gen_text: str,
    model_preset: str,
    speed: float,
    nfe_step: int,
    device_choice: str,
    remove_silence: bool,
    consent_confirmed: bool,
    workspace: Workspace | None = None,
) -> tuple[str, str, str]:
    if not consent_confirmed:
        raise ValueError("Confirm that you are authorized to use this voice before generating speech.")
    if not ref_audio:
        raise ValueError("A reference audio clip is required.")
    workspace = workspace or Workspace()
    workspace.ensure()
    gen_text_norm = normalize_text_for_model(gen_text, model_preset)
    ref_text_norm = normalize_text_for_model(ref_text or DEFAULT_REF_TEXT, model_preset)
    if not gen_text_norm:
        raise ValueError("Write the text to generate.")

    model_args, model_note = resolve_model_preset(model_preset)
    out_dir = workspace.work_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = safe_name(gen_text_norm[:44] or "generated", default="generated") + f"_{uuid.uuid4().hex[:6]}.wav"
    device = detect_device(device_choice)

    cmd = [
        f5_cli_path(),
        *model_args,
        "--ref_audio",
        ref_audio,
        "--ref_text",
        ref_text_norm,
        "--gen_text",
        gen_text_norm,
        "--output_dir",
        str(out_dir),
        "--output_file",
        out_name,
        "--device",
        device,
        "--nfe_step",
        str(int(nfe_step)),
        "--speed",
        str(float(speed)),
        "--no_legacy_text",
    ]
    if remove_silence:
        cmd.append("--remove_silence")

    log = run(cmd, timeout=1200)
    out = out_dir / out_name
    if not out.exists():
        wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not wavs:
            raise RuntimeError("F5-TTS did not produce a WAV file. Log:\n" + log[-4000:])
        out = wavs[0]
    trimmed = trim_audio(out)
    duration = ffprobe_duration(trimmed)
    message = (
        f"Generated: {trimmed}\n"
        f"Duration after trim: {seconds_to_ts(duration)}\n"
        f"Device: {device}\n"
        f"Model preset: {model_preset}\n{model_note}\n\n"
        f"Reference text: {ref_text_norm}\n"
        f"Generated text: {gen_text_norm}\n\n"
        f"Log tail:\n{log[-1500:]}"
    )
    return trimmed, trimmed, message


def parse_edits(edits_json: str) -> list[dict[str, Any]]:
    try:
        edits = json.loads(edits_json or "[]")
    except Exception as exc:
        raise ValueError(f"Invalid edit JSON: {exc}") from exc
    if not isinstance(edits, list):
        raise ValueError("The edit JSON must be a list.")
    for index, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            raise ValueError(f"Edit #{index} must be an object.")
        for key in ("start", "end", "replacement"):
            if key not in edit:
                raise ValueError(f"Edit #{index} is missing `{key}`.")
        start = ts_to_seconds(edit["start"])
        end = ts_to_seconds(edit["end"])
        if end <= start:
            raise ValueError(f"Edit #{index}: end must be greater than start.")
        if not Path(str(edit["replacement"])).exists():
            raise ValueError(f"Edit #{index}: replacement does not exist: {edit['replacement']}")
        edit["start"] = start
        edit["end"] = end
    return edits


def append_edit(
    edits_json: str,
    start: str,
    end: str,
    replacement_audio: str | None,
    generated_audio: str | None,
    label: str,
) -> str:
    edits = parse_edits(edits_json) if (edits_json or "").strip() else []
    replacement = replacement_audio or generated_audio
    if not replacement:
        raise ValueError("Select or generate replacement audio first.")
    s = ts_to_seconds(start)
    e = ts_to_seconds(end)
    if e <= s:
        raise ValueError("End must be greater than start.")
    edits.append({
        "label": label or f"edit {len(edits) + 1}",
        "start": round(s, 3),
        "end": round(e, 3),
        "replacement": replacement,
    })
    return json.dumps(edits, indent=2, ensure_ascii=False)


def fit_to_len(y, n: int, preserve_pitch: bool = True):
    import numpy as np

    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(y) == n:
        return y.copy()
    if len(y) < 2:
        return np.zeros(n, dtype=np.float32)
    if preserve_pitch:
        rate = len(y) / n
        try:
            import librosa

            out = librosa.effects.time_stretch(y.astype(np.float32), rate=rate).astype(np.float32)
        except Exception:
            out = np.interp(
                np.linspace(0, 1, n, endpoint=False),
                np.linspace(0, 1, len(y), endpoint=False),
                y,
            ).astype(np.float32)
    else:
        out = np.interp(
            np.linspace(0, 1, n, endpoint=False),
            np.linspace(0, 1, len(y), endpoint=False),
            y,
        ).astype(np.float32)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n].astype(np.float32)


def splice(
    base,
    replacement,
    start: int,
    end: int,
    sr: int,
    crossfade_ms: float,
    gain_mode: str,
    preserve_pitch: bool,
) -> None:
    import numpy as np

    n = max(1, end - start)
    replacement = fit_to_len(replacement, n, preserve_pitch=preserve_pitch)
    original = base[start:end].copy()
    if gain_mode == "RMS match":
        gain = rms(original) / max(rms(replacement), 1e-6)
        replacement = replacement * max(0.35, min(2.0, gain))
    crossfade = min(int(sr * crossfade_ms / 1000.0), max(1, n // 3))
    out = replacement.copy()
    if crossfade > 0 and len(original) >= crossfade:
        fade_in = np.linspace(0.0, 1.0, crossfade, endpoint=False, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, crossfade, endpoint=False, dtype=np.float32)
        out[:crossfade] = original[:crossfade] * (1.0 - fade_in) + replacement[:crossfade] * fade_in
        out[-crossfade:] = original[-crossfade:] * (1.0 - fade_out) + replacement[-crossfade:] * fade_out
    base[start:end] = out


def concat_file_line(path: Path) -> str:
    # ffmpeg concat demuxer uses single quotes; apostrophes are escaped as '\''.
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'"


def make_preview(original_wav: str, fixed_wav: str, edits: list[dict[str, Any]], stem: str, workspace: Workspace) -> str:
    tmp = workspace.work_dir / f"preview_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    for index, edit in enumerate(edits, 1):
        start = max(0.0, float(edit["start"]) - 1.4)
        end = float(edit["end"]) + 1.6
        before = tmp / f"{index:02d}_before.wav"
        after = tmp / f"{index:02d}_after.wav"
        silence = tmp / f"{index:02d}_silence.wav"
        for src, out in ((original_wav, before), (fixed_wav, after)):
            run([
                ffmpeg_path(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                "-i",
                src,
                "-ar",
                "48000",
                "-ac",
                "1",
                str(out),
            ], timeout=90)
        run([
            ffmpeg_path(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono",
            "-t",
            "0.35",
            str(silence),
        ], timeout=30)
        parts.extend([before, silence, after, silence])
    listfile = tmp / "concat.txt"
    listfile.write_text("\n".join(concat_file_line(p) for p in parts) + "\n", encoding="utf-8")
    preview_wav = workspace.output_dir / f"{stem}_preview_before_after.wav"
    preview_m4a = workspace.output_dir / f"{stem}_preview_before_after.m4a"
    run([ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(preview_wav)], timeout=180)
    run([ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(preview_wav), "-c:a", "aac", "-b:a", "128k", str(preview_m4a)], timeout=180)
    return str(preview_m4a)


def apply_edits(
    source_wav: str | None,
    edits_json: str,
    output_stem: str,
    crossfade_ms: float,
    gain_mode: str,
    preserve_pitch: bool,
    workspace: Workspace | None = None,
) -> tuple[str, str, str, str]:
    if not source_wav:
        raise ValueError("Prepare a source file first.")
    workspace = workspace or Workspace()
    workspace.ensure()
    edits = parse_edits(edits_json)
    if not edits:
        raise ValueError("No edits to apply.")
    base, sr = load_mono(source_wav)
    fixed = base.copy()
    report: list[str] = []
    for edit in edits:
        import librosa

        replacement, _ = load_mono(edit["replacement"], sr)
        replacement, _ = librosa.effects.trim(replacement, top_db=38)
        start = int(round(float(edit["start"]) * sr))
        end = int(round(float(edit["end"]) * sr))
        start = max(0, min(start, len(fixed) - 1))
        end = max(start + 1, min(end, len(fixed)))
        splice(fixed, replacement, start, end, sr, crossfade_ms, gain_mode, preserve_pitch)
        report.append(
            f"- {edit.get('label', 'edit')}: {seconds_to_ts(start / sr)} → {seconds_to_ts(end / sr)} "
            f"using {Path(str(edit['replacement'])).name}"
        )
    stem = safe_name(output_stem or "audio_fixed", default="audio_fixed").removesuffix(".wav").removesuffix(".m4a")
    out_wav = workspace.output_dir / f"{stem}.wav"
    out_m4a = workspace.output_dir / f"{stem}.m4a"
    write_wav(out_wav, fixed, sr)
    run([ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(out_wav), "-c:a", "aac", "-b:a", "128k", str(out_m4a)], timeout=240)
    preview = make_preview(source_wav, str(out_wav), edits, stem, workspace)
    log = "Applied edits:\n" + "\n".join(report) + f"\n\nOutput WAV: {out_wav}\nOutput M4A: {out_m4a}\nPreview: {preview}"
    return str(out_m4a), str(out_wav), preview, log
