import json
import pickle
from functools import partial
from pathlib import Path

import soundfile as sf
import torch
from tqdm.contrib.concurrent import process_map


def estimate_bandwidth(audios, threshold=-50.0, nfft=512, hop=256, sample_rate=16000):
        uid, audio_path = audios
        try:
            audio, fs = sf.read(audio_path)
        except:
            print(f"Error: cannot open audio file '{audio_path}'. Skipping it", flush=True)
            return
        if audio.ndim > 1:
            audio = audio.T
        else:
            audio = audio[None, :]
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

    all_audios = []
    for audio_dir in args.audio_dir:
        if Path(audio_dir).is_dir():
            audios = list(Path(audio_dir).rglob("*." + args.audio_format))
            audios = list(zip(list(range(len(audios))), audios))
        elif Path(audio_dir).is_file() and Path(audio_dir).suffix == ".scp":
            audios = []
            with open(audio_dir, "r") as f:
                for line in f:
                    uid, path = line.strip().split(maxsplit=1)
                    audios.append((uid, path))
        else:
            raise ValueError(f"Invalid format: {audio_dir}")
        all_audios.extend(audios)
    ret0 = process_map(
        partial(estimate_bandwidth, threshold=args.threshold),
        audios,
        chunksize=args.chunksize,
        max_workers=args.nj,
    )

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "bandwidth.pkl", "wb") as f:
        pickle.dump(ret0, f)
    ret = {uid_val[0]: uid_val[1] for uid_val in ret0 if uid_val is not None}
    if args.outfile.endswith(".json"):
        with open(args.outfile, "w") as f:
            json.dump(ret, f, indent=2)
    else:
        with open(args.outfile, "w") as f:
            for uid, (bandwidth, audio_path) in ret.items():
                f.write(f"{uid} {bandwidth} {audio_path}\n")
