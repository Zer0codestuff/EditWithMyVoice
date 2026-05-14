from __future__ import annotations

import gradio as gr

from .core import (
    APP_NAME,
    DEFAULT_MODEL_PRESET,
    DEFAULT_REF_TEXT,
    MODEL_PRESETS,
    Workspace,
    append_edit,
    apply_edits,
    command_exists,
    extract_clip,
    f5_cli_path,
    ffmpeg_path,
    prepare_source,
    synthesize_f5,
)

CSS = """
.gradio-container {max-width: 1180px !important}
#title {text-align:center; margin-bottom: 0.5rem}
.small-note {font-size: 0.92em; opacity: 0.8}
"""


def _wrap(fn):
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # Gradio displays this cleanly.
            raise gr.Error(str(exc)) from exc

    return inner


def system_check() -> str:
    lines = [f"Workspace: `{Workspace().root}`"]
    try:
        lines.append(f"FFmpeg: `{ffmpeg_path()}`")
    except Exception as exc:
        lines.append(f"FFmpeg: ❌ {exc}")
    try:
        lines.append(f"F5-TTS CLI: `{f5_cli_path()}`")
    except Exception as exc:
        lines.append(f"F5-TTS CLI: ❌ {exc}")
    lines.append(f"Git: {'available' if command_exists('git') else 'not required'}")
    return "\n".join(f"- {line}" for line in lines)


def clear_edits() -> str:
    return "[]"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_NAME, theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown(
            "# Edit With My Voice\n"
            "Local tool for authorized voice-cloning assisted audio repair and precise phrase replacement.",
            elem_id="title",
        )

        source_wav_state = gr.State("")
        generated_audio_state = gr.State("")

        with gr.Tab("1. Project"):
            gr.Markdown(
                "Upload an audio/video file. The app converts it into a local 48 kHz mono WAV workspace copy."
            )
            gr.Markdown(system_check())
            src_file = gr.File(
                label="Original audio/video",
                file_types=["audio", "video", ".m4a", ".mp3", ".wav", ".mp4", ".mov", ".ogg", ".flac"],
            )
            prepare_btn = gr.Button("Prepare source", variant="primary")
            prepared_audio = gr.Audio(label="Prepared source", type="filepath")
            prepare_log = gr.Textbox(label="Log", lines=3)
            prepare_btn.click(
                _wrap(prepare_source),
                inputs=[src_file],
                outputs=[source_wav_state, prepared_audio, prepare_log],
            )

        with gr.Tab("2. Reference / donor clip"):
            gr.Markdown(
                "Extract clean speech from the same voice. Use it either as an F5-TTS reference or as a direct donor clip."
            )
            with gr.Row():
                clip_start = gr.Textbox(label="Start", value="00:00.000")
                clip_end = gr.Textbox(label="End", value="00:12.000")
                clip_name = gr.Textbox(label="Clip name", value="reference.wav")
            extract_btn = gr.Button("Extract clip", variant="primary")
            clip_audio = gr.Audio(label="Extracted clip", type="filepath")
            clip_log = gr.Textbox(label="Log", lines=2)
            extract_btn.click(
                _wrap(extract_clip),
                inputs=[source_wav_state, clip_start, clip_end, clip_name],
                outputs=[clip_audio, clip_log],
            )

        with gr.Tab("3. Generate replacement"):
            gr.Markdown(
                "Generate the corrected word or phrase with F5-TTS. For natural edits, generate a short phrase rather than a single isolated word."
            )
            consent = gr.Checkbox(
                label="I confirm I am authorized to use this voice and create the replacement audio.",
                value=False,
            )
            ref_audio = gr.Audio(label="Reference audio", type="filepath")
            ref_text = gr.Textbox(label="Exact/approximate transcript of the reference", value=DEFAULT_REF_TEXT, lines=3)
            gen_text = gr.Textbox(label="Text to generate", value="the corrected phrase", lines=2)
            model_preset = gr.Dropdown(
                list(MODEL_PRESETS.keys()),
                value=DEFAULT_MODEL_PRESET,
                label="F5-TTS model preset",
            )
            with gr.Row():
                speed = gr.Slider(0.75, 1.35, value=1.0, step=0.05, label="Speed")
                nfe = gr.Slider(8, 48, value=24, step=4, label="Quality steps")
                device = gr.Dropdown(["auto", "cuda", "cpu", "mps"], value="auto", label="Device")
                remove_silence = gr.Checkbox(value=False, label="Ask F5-TTS to remove silence")
            synth_btn = gr.Button("Generate phrase", variant="primary")
            generated_audio = gr.Audio(label="Generated audio", type="filepath")
            synth_log = gr.Textbox(label="Generation log", lines=10)
            synth_btn.click(
                _wrap(synthesize_f5),
                inputs=[ref_audio, ref_text, gen_text, model_preset, speed, nfe, device, remove_silence, consent],
                outputs=[generated_audio, generated_audio_state, synth_log],
            )

        with gr.Tab("4. Edit and export"):
            gr.Markdown(
                "Add timestamped replacements. If no manual replacement file is selected, the app uses the last generated phrase."
            )
            with gr.Row():
                edit_start = gr.Textbox(label="Edit start", value="00:00.000")
                edit_end = gr.Textbox(label="Edit end", value="00:01.000")
                edit_label = gr.Textbox(label="Label", value="correction")
            replacement_audio = gr.Audio(label="Manual replacement audio (optional)", type="filepath")
            with gr.Row():
                add_edit_btn = gr.Button("Add edit")
                clear_btn = gr.Button("Clear edits")
            edits_json = gr.Code(label="Edit JSON", language="json", value="[]", lines=12)
            add_edit_btn.click(
                _wrap(append_edit),
                inputs=[edits_json, edit_start, edit_end, replacement_audio, generated_audio_state, edit_label],
                outputs=[edits_json],
            )
            clear_btn.click(clear_edits, outputs=[edits_json])

            with gr.Row():
                out_stem = gr.Textbox(label="Output name", value="audio_fixed")
                crossfade_ms = gr.Slider(0, 120, value=30, step=5, label="Crossfade ms")
                gain_mode = gr.Dropdown(["RMS match", "keep replacement gain"], value="RMS match", label="Gain")
                preserve_pitch = gr.Checkbox(value=True, label="Preserve pitch while time-fitting")
            apply_btn = gr.Button("Apply edits and create preview", variant="primary")
            out_m4a = gr.Audio(label="Corrected M4A", type="filepath")
            out_wav_file = gr.File(label="Corrected WAV")
            preview_audio = gr.Audio(label="Before/after preview", type="filepath")
            apply_log = gr.Textbox(label="Export log", lines=10)
            apply_btn.click(
                _wrap(apply_edits),
                inputs=[source_wav_state, edits_json, out_stem, crossfade_ms, gain_mode, preserve_pitch],
                outputs=[out_m4a, out_wav_file, preview_audio, apply_log],
            )

        with gr.Tab("Guide"):
            gr.Markdown(
                """
## Recommended workflow

1. **Prepare source**: upload audio/video and create the working WAV.
2. **Extract reference**: choose 10–30 seconds of clean speech from the same speaker.
3. **Generate phrase**: enter the reference transcript and the corrected phrase.
4. **Add edit**: set start/end timestamps and add the replacement.
5. **Apply**: listen to the before/after preview before using the final export.

## Quality tips

- Replace a phrase, not just a naked word, when possible.
- Cut on silence, breaths, or consonants; avoid cutting inside long vowels.
- Keep crossfades around 20–40 ms for short speech edits.
- If AI sounds unnatural, first search for a real donor phrase in the same recording.
- Write numbers as words for TTS.

## Responsible use

Only use voices you own or have permission to edit. Do not use this tool to impersonate or mislead people.
                """
            )
    return demo


def main() -> None:
    Workspace().ensure()
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
