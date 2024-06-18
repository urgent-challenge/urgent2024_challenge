import logging
from pathlib import Path

import fast_bss_eval
import librosa
import numpy as np
import soundfile as sf
import soxr
from pesq import PesqError, pesq
from pystoi import stoi
from tqdm.contrib.concurrent import process_map

from mcd_utils import calculate as calculate_mcd


METRICS = ("PESQ", "ESTOI", "SDR", "LSD", "MCD")

if "VISQOL" in METRICS:
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2

    visqol_config = visqol_config_pb2.VisqolConfig()
    # 16kHz for speech mode
    visqol_config.audio.sample_rate = 16000
    visqol_config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    visqol_config.options.svr_model_path = str(
        Path(visqol_lib_py.__file__).parent / "model" / svr_model_path
    )
    visqol_api = visqol_lib_py.VisqolApi()
    visqol_api.Create(visqol_config)


################################################################
# Definition of metrics
################################################################
def get_2fmodel_metric(ref, inf, fs=48000):
    """Calculate 2f-model.

    References: https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS/

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        ret (float): 2f-model value between [0, 100]
    """
    raise NotImplementedError


def estoi_metric(ref, inf, fs=16000):
    """Calculate Extended Short-Time Objective Intelligibility (ESTOI).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        estoi (float): ESTOI value between [0, 1]
    """
    return stoi(ref, inf, fs_sig=fs, extended=True)


def lsd_metric(ref, inf, fs, nfft=0.032, hop=0.016, p=2, eps=1.0e-08):
    """Calculate Log-Spectral Distance (LSD).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
        nfft (float): FFT length in seconds
        hop (float): hop length in seconds
        p (float): the order of norm
        eps (float): epsilon value for numerical stability
    Returns:
        mcd (float): LSD value between [0, +inf)
    """
    scaling_factor = np.sum(ref * inf) / (np.sum(inf**2) + eps)
    inf = inf * scaling_factor

    nfft = int(fs * nfft)
    hop = int(fs * hop)
    # T x F
    ref_spec = np.abs(librosa.stft(ref, hop_length=hop, n_fft=nfft)).T
    inf_spec = np.abs(librosa.stft(inf, hop_length=hop, n_fft=nfft)).T
    lsd = np.log(ref_spec**2 / ((inf_spec + eps) ** 2) + eps) ** p
    lsd = np.mean(np.mean(lsd, axis=1) ** (1 / p), axis=0)
    return lsd


def mcd_metric(ref, inf, fs, eps=1.0e-08):
    """Calculate Mel Cepstral Distortion (MCD).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
        eps (float): epsilon value for numerical stability
    Returns:
        mcd (float): MCD value between [0, +inf)
    """
    scaling_factor = np.sum(ref * inf) / (np.sum(inf**2) + eps)
    return calculate_mcd(ref, inf * scaling_factor, fs)


def pesq_metric(ref, inf, fs=8000):
    """Calculate Perceptual Evaluation of Speech Quality (PESQ).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        pesq (float): PESQ value between [-0.5, 4.5]
    """
    assert ref.shape == inf.shape
    if fs == 8000:
        mode = "nb"
    elif fs == 16000:
        mode = "wb"
    elif fs > 16000:
        mode = "wb"
        ref = soxr.resample(ref, fs, 16000)
        inf = soxr.resample(inf, fs, 16000)
        fs = 16000
    else:
        raise ValueError(
            "sample rate must be 8000 or 16000+ for PESQ evaluation, " f"but got {fs}"
        )
    pesq_score = pesq(
        fs,
        ref,
        inf,
        mode=mode,
        on_error=PesqError.RETURN_VALUES,
    )
    if pesq_score == PesqError.NO_UTTERANCES_DETECTED:
        logging.warning(
            f"[PESQ] Error: No utterances detected. " "Skipping this sample."
        )
    else:
        return pesq_score


def sdr_metric(ref, inf):
    """Calculate signal-to-distortion ratio (SDR).

    Args:
        ref (np.ndarray): reference signal (num_src, time)
        inf (np.ndarray): enhanced signal (num_src, time)
    Returns:
        sdr (float): SDR values (unbounded)
    """
    assert ref.shape == inf.shape
    if ref.ndim == 1:
        ref = ref[None, :]
        inf = inf[None, :]
    else:
        assert ref.ndim == 2, ref.shape
    num_src, _ = ref.shape
    sdr, sir, sar = fast_bss_eval.bss_eval_sources(
        ref, inf, compute_permutation=False, clamp_db=50.0
    )
    return float(np.mean(sdr))


def visqol_metric(ref, inf, fs=16000):
    """Calculate Virtual Speech Quality Objective Listener (VISQOL).

    Reference: https://github.com/google/visqol

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        visqol (float): VISQOL value between [1, 5]
    """
    ref = soxr.resample(ref, fs, 16000)
    inf = soxr.resample(inf, fs, 16000)
    fs = 16000

    similarity_result = visqol_api.Measure(ref, inf)

    return similarity_result


################################################################
# Main entry
################################################################
def main(args):
    refs = {}
    with open(args.ref_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            refs[uid] = audio_path

    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, refs[uid], audio_path))

    ret = process_map(
        process_one_pair,
        data_pairs,
        max_workers=args.nj,
        chunksize=args.chunksize,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {metric: (outdir / f"{metric}.scp").open("w") for metric in METRICS}

    for uid, score in ret:
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    with (outdir / "RESULTS.txt").open("w") as f:
        for metric in METRICS:
            mean_score = np.nanmean([score[metric] for uid, score in ret])
            f.write(f"{metric}: {mean_score:.4f}\n")
    print(f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True)


def process_one_pair(data_pair):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path, dtype="float32")
    inf, fs2 = sf.read(inf_path, dtype="float32")
    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)

    scores = {}
    for metric in METRICS:
        if metric == "PESQ":
            pesq_score = pesq_metric(ref, inf, fs=fs)
            scores[metric] = pesq_score if pesq_score is not None else np.nan
        elif metric == "ESTOI":
            scores[metric] = estoi_metric(ref, inf, fs=fs)
        elif metric == "SDR":
            scores[metric] = sdr_metric(ref, inf)
        elif metric == "LSD":
            scores[metric] = lsd_metric(ref, inf, fs=fs)
        elif metric == "MCD":
            scores[metric] = mcd_metric(ref, inf, fs=fs)
        elif metric == "VISQOL":
            scores[metric] = visqol_metric(ref, inf, fs=fs)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_scp",
        type=str,
        required=True,
        help="Path to the scp file containing reference signals",
    )
    parser.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        help="Path to the scp file containing enhanced signals",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for writing metrics",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=8,
        help="Number of parallel workers to speed up evaluation",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="Chunk size used in process_map",
    )
    args = parser.parse_args()

    main(args)
