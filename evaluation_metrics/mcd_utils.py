#!/usr/bin/env python3

# Copyright 2020 Wen-Chin Huang and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# ported from https://github.com/espnet/espnet/blob/master/utils/mcd_calculate.py

"""Evaluate MCD between generated and groundtruth audios with SPTK-based mcep."""

from typing import Tuple

import numpy as np
import pysptk
from fastdtw import fastdtw
from scipy import spatial


def sptk_extract(
    x: np.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
    is_padding: bool = False,
) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Sampling rate
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).
        is_padding (bool): Whether to pad the end of signal (default=False).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).

    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")

    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1

    # get window function
    win = pysptk.sptk.hamming(n_fft)

    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # calculate spectrogram
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return np.stack(mcep)


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    # https://sp-nitech.github.io/sptk/latest/main/mgcep.html#_CPPv4N4sptk19MelCepstralAnalysisE
    if fs == 8000:
        return 13, 0.31
    elif fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 32000:
        return 36, 0.50
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


def calculate(
    inf_audio,
    ref_audio,
    fs,
    n_fft=1024,
    n_shift=256,
    mcep_dim=None,
    mcep_alpha=None,
):
    """Calculate MCD."""

    # extract ground truth and converted features
    gen_mcep = sptk_extract(
        x=inf_audio,
        fs=fs,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )
    gt_mcep = sptk_extract(
        x=ref_audio,
        fs=fs,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )

    # DTW
    _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    gen_mcep_dtw = gen_mcep[twf[0]]
    gt_mcep_dtw = gt_mcep[twf[1]]

    # MCD
    diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    return mcd
