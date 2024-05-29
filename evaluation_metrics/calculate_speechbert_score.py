from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from tqdm import tqdm

from discrete_speech_metrics import SpeechBERTScore as SBS


METRICS = ("SpeechBERTScore",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
class SpeechBERTScore:
    """SpeechBERTScore.

    Reference:
        SpeechBERTScore: Reference-Aware Automatic Evaluation of Speech
        Generation Leveraging NLP Evaluation Metrics
        https://arxiv.org/abs/2401.16812
    """

    def __init__(self, device="cpu"):
        self.speech_bert_score = SBS(
            sr=TARGET_FS, model_type="hubert-base", layer=8, use_gpu="cuda" in device
        )

    def __call__(self, reference: np.ndarray, sample: np.ndarray) -> float:
        precision, recall, f1_score = self.speech_bert_score.score(reference, sample)
        return precision, recall, f1_score


def speech_bert_score_metric(model, ref, inf, fs=16000):
    """Calculate the SpeechBERTScore between ref and inf.

    Args:
        model (torch.nn.Module): SpeechBERTScore model
            Please use the model with model_type="hubert-base" and layer=8
            to get comparable results.
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        score (float): SpeechBERTScore precision value between [0, 1]
    """
    if fs != TARGET_FS:
        ref = soxr.resample(ref, fs, TARGET_FS)
        inf = soxr.resample(inf, fs, TARGET_FS)
    with torch.no_grad():
        score = model(ref, inf)
    return score[0]


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

    size = len(data_pairs)
    assert 1 <= args.job <= args.nsplits <= size
    interval = size // args.nsplits
    start = (args.job - 1) * interval
    end = size if args.job == args.nsplits else start + interval
    data_pairs = data_pairs[start:end]
    print(
        f"[Job {args.job}/{args.nsplits}] Processing ({len(data_pairs)}/{size}) samples",
        flush=True,
    )
    suffix = "" if args.nsplits == args.job == 1 else f".{args.job}"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {
        metric: (outdir / f"{metric}{suffix}.scp").open("w") for metric in METRICS
    }

    model = SpeechBERTScore(device=args.device)
    model.speech_bert_score.model.eval()
    ret = []
    for uid, ref_audio, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair((uid, ref_audio, inf_audio), model=model)
        ret.append((uid, score))
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    if args.nsplits == args.job == 1:
        with (outdir / "RESULTS.txt").open("w") as f:
            for metric in METRICS:
                mean_score = np.nanmean([score[metric] for uid, score in ret])
                f.write(f"{metric}: {mean_score:.4f}\n")
        print(
            f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True
        )


def process_one_pair(data_pair, model=None):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path, dtype="float32")
    inf, fs2 = sf.read(inf_path, dtype="float32")
    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)
    assert ref.ndim == 1, ref.shape

    scores = {}
    for metric in METRICS:
        if metric == "SpeechBERTScore":
            scores[metric] = speech_bert_score_metric(model, ref, inf, fs=fs)
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
        "--device",
        type=str,
        default="cpu",
        help="Device for running speaker embedding extraction",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=1,
        help="Total number of computing nodes to speed up evaluation",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=1,
        help="Index of the current node (starting from 1)",
    )
    args = parser.parse_args()

    main(args)
