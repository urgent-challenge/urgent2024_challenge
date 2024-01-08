from functools import partial
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm.contrib.concurrent import process_map

from espnet2.bin.spk_inference import Speech2Embedding


METRICS = ("SpeakerSimilarity",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
def speaker_similarity_metric(model, ref, inf, fs=16000):
    """Calculate the cosine similarity between ref and inf speaker embeddings.

    Args:
        model (torch.nn.Module): speaker model
            Please use the model in https://huggingface.co/espnet/voxcelebs12_rawnet3
            to get comparable results.
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        similarity (float): cosine similarity value between [0, 1]
    """
    if fs != TARGET_FS:
        ref = librosa.resample(ref, orig_sr=fs, target_sr=TARGET_FS)
        inf = librosa.resample(inf, orig_sr=fs, target_sr=TARGET_FS)
    with torch.no_grad():
        ref_emb = model(ref)
        inf_emb = model(inf)
        similarity = torch.cosine_similarity(ref_emb, inf_emb, dim=-1).item()
    return similarity


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

    model = Speech2SpkEmbedding.from_pretrained(model_tag="espnet/voxcelebs12_rawnet3")
    model.to(device=args.device).eval()
    ret = process_map(
        partial(process_one_pair, model=model, device=args.device),
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


def process_one_pair(data_pair, model=None, dtype="float32", device="cpu"):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path)
    inf, fs2 = sf.read(inf_path)
    assert isinstance(model, torch.nn.Module), model
    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)
    assert ref.ndim == 1, ref.shape

    scores = {}
    for metric in METRICS:
        if metric == "SpeakerSimilarity":
            scores[metric] = speaker_similarity_metric(
                model, ref, inf, fs=fs, device=device
            )
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
