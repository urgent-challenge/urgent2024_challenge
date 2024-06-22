import json
import pickle
from functools import partial
from pathlib import Path

import soundfile as sf
import torch
from tqdm.contrib.concurrent import process_map


def estimate_bandwidth(audios, threshold=-50.0, nfft=512, hop=256, sample_rate=16000):
    uid, audio_path = audios
    if isinstance(audio_path, dict):
        st = int(audio_path["start"] * sample_rate)
        et = int(audio_path["end"] * sample_rate)
        idx = slice(st, et)
        audio_path = audio_path["audio_path"]
    else:
        idx = slice(None)

    try:
        audio, fs = sf.read(audio_path)
    except:
        # Some of the downloaded DNS5 speech audio files may be broken (extracted from
        # dns5_fullband/Track1_Headset/read_speech.tgz.part*) according to our tests.
        print(f"Error: cannot open audio file '{audio_path}'. Skipping it", flush=True)
        return
    if audio.ndim > 1:
        audio = audio[idx].T
    else:
        audio = audio[None, idx]
    spec = torch.stft(
        torch.from_numpy(audio),
        n_fft=int(nfft / sample_rate * fs),
        hop_length=int(hop / sample_rate * fs),
        window=torch.hann_window(int(nfft / sample_rate * fs)),
        onesided=True,
        return_complex=True,
    )
    freq = torch.fft.rfftfreq(int(nfft / sample_rate * fs), d=1 / fs)
    assert len(freq) == spec.size(1), (freq.shape, spec.shape)
    power = spec.real.pow(2) + spec.imag.pow(2)
    # (C, F, T) -> (C, F)
    mean_power = power.mean(2)
    peak = mean_power.max(1).values
    min_energy = peak.min() * 10 ** (threshold / 10)
    for i in range(len(freq) - 1, -1, -1):
        if mean_power[:, i].min() > min_energy:
            return uid, [str(audio_path), freq[i].item()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        nargs="+",
        help="Path to the directory containing audios or "
        "path to the wav.scp file containing paths to audios",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to the output file for writing bandwidth information",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-50,
        help="Minimum energy level in dB relative to the peak value of the spectrum\n"
        "The highest frequencies satisfying the above condition is considered as the "
        "bandwidth",
    )
    parser.add_argument(
        "--audio_format", type=str, default="wav", help="Suffix of the audio files"
    )
    parser.add_argument("--nj", type=int, default=8, help="Number of parallel workers")
    parser.add_argument(
        "--chunksize", type=int, default=1000, help="Chunk size for each worker"
    )
    args = parser.parse_args()

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)

    all_audios = []
    for audio_dir in args.audio_dir:
        if Path(audio_dir).is_dir():
            audios = list(Path(audio_dir).rglob("*." + args.audio_format))
            audios = list(zip([p.stem for p in audios], audios))
        elif Path(audio_dir).is_file() and Path(audio_dir).suffix == ".scp":
            audios = []
            with open(audio_dir, "r") as f:
                for line in f:
                    uid, path = line.strip().split(maxsplit=1)
                    audios.append((uid, path))
        elif Path(audio_dir).is_file() and Path(audio_dir).suffix == ".json":
            audios = []
            with open(audio_dir, "r") as f:
                for uid, dic in json.load(f).items():
                    audios.append((uid, dic))
        else:
            raise ValueError(f"Invalid format: {audio_dir}")
        all_audios.extend(audios)
    pkl_file = Path(args.outfile).with_suffix(".pkl")
    if pkl_file.exists():
        print(f"Loading existing pkl file: {pkl_file}")
        with pkl_file.open("r") as f:
            ret0 = pickle.load(f)
    else:
        ret0 = process_map(
            partial(estimate_bandwidth, threshold=args.threshold),
            audios,
            chunksize=args.chunksize,
            max_workers=args.nj,
        )
        with pkl_file.open("wb") as f:
            pickle.dump(ret0, f)

    ret = {}
    for uid_val in ret0:
        if uid_val is None:
            continue
        uid, val = uid_val
        i = 1; uid2 = uid
        while uid2 in ret:
            i += 1
            uid2 = f"{uid}({i})"
        ret[uid2] = val

    if args.outfile.endswith(".json"):
        with open(args.outfile, "w") as f:
            json.dump(ret, f, indent=2)
    else:
        with open(args.outfile, "w") as f:
            for uid, (bandwidth, audio_path) in ret.items():
                f.write(f"{uid} {bandwidth} {audio_path}\n")
