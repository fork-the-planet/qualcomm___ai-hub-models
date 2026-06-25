# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from math import gcd

import numpy as np
import scipy.signal as sig
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms

from qai_hub_models.models.deepspeech2.model import BLANK_IDX, LABELS
from qai_hub_models.models.protocols import ExecutableModelProtocol


def ctc_greedy_decode(indices: torch.Tensor) -> str:
    """
    CTC greedy decoding for a single utterance.

    Parameters
    ----------
    indices
        Argmax token indices of shape ``(1, time)`` or ``(time,)``, as
        returned directly by the model's ``forward()`` pass.

    Returns
    -------
    str
        Decoded text string.
    """
    if indices.dim() == 2:
        indices = indices.squeeze(0)

    decoded = []
    prev_idx = None
    for idx in indices.tolist():
        if idx not in (prev_idx, BLANK_IDX) and 0 <= idx < len(LABELS):
            decoded.append(LABELS[idx])
        prev_idx = idx

    return "".join(decoded)


def preprocess_waveform_to_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    window_size: float = 0.02,
    window_stride: float = 0.01,
) -> torch.Tensor:
    """
    Convert a waveform tensor to a normalized spectrogram.

    Parameters
    ----------
    waveform
        1-D tensor of audio samples (already at target sample_rate).
    sample_rate
        Sample rate in Hz.
    window_size
        STFT window size in seconds.
    window_stride
        STFT hop length in seconds.

    Returns
    -------
    torch.Tensor
        Spectrogram tensor of shape (1, time, 161).
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    n_fft = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        power=None,
        window_fn=torch.hamming_window,
    )
    complex_spec = spec_transform(waveform)
    if complex_spec.is_complex():
        spect = complex_spec.abs()
    elif complex_spec.shape[-1] == 2:
        spect = torch.sqrt(complex_spec[..., 0] ** 2 + complex_spec[..., 1] ** 2)
    else:
        spect = complex_spec

    spect = torch.log1p(spect)
    mean = spect.mean()
    std = spect.std()
    spect = (spect - mean) / (std + 1e-9)

    spect = spect.squeeze(0).transpose(0, 1)
    return spect.unsqueeze(0)


class DeepSpeech2App:
    """
    DeepSpeech2 application for speech-to-text conversion.

    Handles audio preprocessing (mel spectrogram extraction) and
    CTC decoding (greedy decoder) for speech recognition.

    Parameters
    ----------
    model
        DeepSpeech2 model instance.
    context_len
        Optional fixed sequence length for input (time dimension).
        If set, inputs will be padded/cropped to this length.
        Required for on-device inference with fixed-shape models.
    """

    def __init__(
        self, model: ExecutableModelProtocol, context_len: int | None = None
    ) -> None:
        self.model = model
        self.context_len = context_len
        n_fft = int(16000 * 0.02)
        hop_length = int(16000 * 0.01)
        self._spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            power=None,
            window_fn=torch.hamming_window,
        )

    def preprocess_audio(
        self,
        audio_path: str,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Load and preprocess audio file to spectrogram.

        Parameters
        ----------
        audio_path
            Path to audio file.
        sample_rate
            Target sample rate (Hz).

        Returns
        -------
        torch.Tensor
            Spectrogram tensor of shape (1, time, n_feature).
        """
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        # soundfile returns (L, C); convert to (C, L)
        waveform = torch.from_numpy(np.transpose(audio, (1, 0)))

        if sr != sample_rate:
            g = gcd(sr, sample_rate)
            waveform_np = waveform.numpy()
            waveform_np = sig.resample_poly(
                waveform_np, sample_rate // g, sr // g, axis=-1
            )
            waveform = torch.from_numpy(waveform_np.astype(np.float32))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return self._waveform_to_spectrogram(waveform.squeeze(0))

    def _waveform_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        complex_spec = self._spec_transform(waveform)
        if complex_spec.is_complex():
            spect = complex_spec.abs()
        elif complex_spec.shape[-1] == 2:
            spect = torch.sqrt(complex_spec[..., 0] ** 2 + complex_spec[..., 1] ** 2)
        else:
            spect = complex_spec
        spect = torch.log1p(spect)
        mean = spect.mean()
        std = spect.std()
        spect = (spect - mean) / (std + 1e-9)
        spect = spect.squeeze(0).transpose(0, 1)
        return spect.unsqueeze(0)

    def predict(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.

        Parameters
        ----------
        audio_path
            Path to audio file.

        Returns
        -------
        str
            Transcribed text.
        """
        mel_spec = self.preprocess_audio(audio_path)

        if self.context_len is not None:
            current_len = mel_spec.shape[1]
            target_len = self.context_len

            if current_len > target_len:
                mel_spec = mel_spec[:, :target_len, :]
            elif current_len < target_len:
                pad_len = target_len - current_len
                mel_spec = F.pad(mel_spec, (0, 0, 0, pad_len))

        indices = self.model(mel_spec)
        if isinstance(indices, tuple):
            indices = indices[0]

        return ctc_greedy_decode(indices)

    def predict_batch(self, audio_paths: list[str]) -> list[str]:
        """
        Transcribe multiple audio files.

        Parameters
        ----------
        audio_paths
            List of paths to audio files.

        Returns
        -------
        list[str]
            List of transcribed texts.
        """
        return [self.predict(path) for path in audio_paths]
