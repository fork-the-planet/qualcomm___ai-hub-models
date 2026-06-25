# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.deepspeech2.app import DeepSpeech2App
from qai_hub_models.models.deepspeech2.model import DEFAULT_AUDIO, MODEL_ID, DeepSpeech2
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)


def main(is_test: bool = False) -> None:
    """Run DeepSpeech2 demo for speech recognition."""
    parser = get_model_cli_parser(DeepSpeech2)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file for transcription (.wav format recommended)",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load model (handles both local weights and hub-model-id)
    print("Loading DeepSpeech2 model...")
    model = demo_model_from_cli_args(DeepSpeech2, MODEL_ID, args)

    # For on-device or Hub models, we need to match the compiled fixed shape (3500 frames)
    context_len = None
    is_on_device = (
        args.eval_mode == "on-device" if hasattr(args, "eval_mode") else False
    )
    if args.hub_model_id or is_on_device:
        input_spec = DeepSpeech2.get_input_spec()
        context_len = input_spec["input"][0][1]

    app = DeepSpeech2App(model, context_len=context_len)
    print("Model loaded successfully!")

    # Load default audio if not provided
    audio_file = args.audio_file
    if not audio_file:
        audio_file = str(DEFAULT_AUDIO.fetch())

    print(f"\nTranscribing: {audio_file}")
    transcription = app.predict(audio_file)
    print(f"\nTranscription: {transcription}")


if __name__ == "__main__":
    main()
