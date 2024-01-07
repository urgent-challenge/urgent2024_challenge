import logging
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from mir_eval.separation import bss_eval_sources
from pesq import PesqError, pesq
from pystoi import stoi
from tqdm.contrib.concurrent import process_map


METRICS = (
    "PESQ", "ESTOI", "SDR", "MCD", "VISQOL"
)


################################################################
# Definition of metrics
################################################################
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


def mcd_metric(ref, inf):
    """Calculate Mel Cepstral Distortion (MCD).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
    Returns:
        mcd (float): MCD value (unbounded)
    """
    return


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
        ref = librosa.resample(ref, orig_sr=fs, target_sr=16000)
        inf = librosa.resample(inf, orig_sr=fs, target_sr=16000)
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
        sdr (np.ndarray): SDR values (unbounded)
    """
    assert ref.shape == inf.shape
    num_src, _ = ref.shape
    sdr, sir, sar, perm = bss_eval_sources(ref, inf, compute_permutation=True)
    return sdr


def visqol_metric(ref, inf, fs=48000):
    """Calculate Virtual Speech Quality Objective Listener (VISQOL).

    Reference: https://github.com/google/visqol

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        visqol (float): VISQOL value between [1, 5]
    """
    return


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
    print(f"Overall results have been written in {outdir / 'RESULTS.md'}", flush=True)


def process_one_pair(data_pair):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path)
    inf, fs2 = sf.read(inf_path)
    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)

    scores = {}
    for metric in METRICS:
        if metric == "PESQ":
            pesq_score = pesq_metric(ref, inf, fs=fs)
            if pesq_score is not None:
                scores[metric] = pesq_score
        elif metric == "ESTOI":
            scores[metric] = estoi_metric(ref, inf, fs=fs)
        elif metric == "SDR":
            scores[metric] = sdr_metric(ref, inf)
        elif metric == "MCD":
            scores[metric] = mcd_metric(ref, inf)
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
