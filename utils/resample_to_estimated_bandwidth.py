import json
from functools import partial
from pathlib import Path

import librosa
import soundfile as sf
from tqdm.contrib.concurrent import process_map


sampling_rates = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


def resample_to_estimated_bandwidth(path_bw, outdir, resample_type="kaiser_best"):
    audio_path, est_bandwidth = path_bw
    try:
        audio, fs = sf.read(audio_path)
    except:
        print(f"Error: cannot open audio file '{audio_path}'. Skipping it", flush=True)
    for sr in sampling_rates:
        if est_bandwidth * 2 <= sr:
            est_fs = sr
            break
    else:
        est_fs = sampling_rates[-1]
    if est_fs == fs:
        return audio_path, fs

    if audio.ndim > 1:
        audio = audio.T
    audio = librosa.resample(
        audio, orig_sr=fs, target_sr=est_fs, res_type=resample_type
    )
    if audio.ndim > 1:
        audio = audio.T
    outfile = str(Path(outdir) / (Path(audio_path).stem + ".wav"))
    sf.write(outfile, audio, est_fs)
    return outfile, est_fs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bandwidth_data",
        type=str,
        required=True,
        help="Path to the json/scp file containing audio paths and "
        "the corresponding bandwidth inforamtion",
    )
    parser.add_argument(
        "--out_scpfile", type=str, required=True, help="Path to the output scp file"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for storing resampled audios",
    )
    parser.add_argument(
        "--resample_type", type=str, default="kaiser_best", help="Resampling type"
    )
    parser.add_argument("--nj", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--chunksize", type=int, default=1, help="Chunksize for parallel jobs"
    )

    args = parser.parse_args()

    if Path(args.bandwidth_data).suffix == ".json":
        with open(args.bandwidth_data, "r") as f:
            audios = list(json.load(f).values())
    elif Path(args.audio_dir).suffix == ".scp":
        audios = []
        with open(args.audio_dir, "r") as f:
            for line in f:
                uid, bandwidth, path = line.strip().split(maxsplit=1)
                audios.append((path, bandwidth))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ret = process_map(
        partial(
            resample_to_estimated_bandwidth,
            outdir=args.outdir,
            resample_type=args.resample_type,
        ),
        audios,
        max_workers=args.nj,
        chunksize=args.chunksize,
    )

    Path(args.out_scpfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_scpfile, "w") as f:
        for audio_path, fs in ret:
            uid = Path(audio_path).stem
            f.write(f"{uid} {fs} {audio_path}\n")
