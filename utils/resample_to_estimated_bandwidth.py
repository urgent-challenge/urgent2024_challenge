import json
from functools import partial
import math
from pathlib import Path

# import librosa
import soxr
import soundfile as sf
from tqdm.contrib.concurrent import process_map


sampling_rates = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


def resample_to_estimated_bandwidth(
    uid_path_bw, idx, max_files_per_dir, num_digits, outdir
):
    uid, audio_path, est_bandwidth = uid_path_bw
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
        return uid, audio_path, fs

    audio = soxr.resample(audio, fs, est_fs)

    subdir = f"{idx // max_files_per_dir:0{num_digits}x}"
    outfile = Path(outdir) / subdir / (uid + ".wav")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(outfile), audio, est_fs)
    return uid, outfile, est_fs


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
    parser.add_argument("--nj", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument(
        "--chunksize", type=int, default=1, help="Chunksize for parallel jobs"
    )
    parser.add_argument(
        "-m",
        "--max_files",
        type=int,
        default=10000,
        help="The maximum number of files per sub-directory. "
        "This is useful for systems that limit the max number of files per directory",
    )

    args = parser.parse_args()

    if Path(args.bandwidth_data).suffix == ".json":
        audios = []
        with open(args.bandwidth_data, "r") as f:
            for uid, (path, bandwidth) in json.load(f).items():
                audios.append((uid, path, bandwidth))
    elif Path(args.audio_dir).suffix == ".scp":
        audios = []
        with open(args.audio_dir, "r") as f:
            for line in f:
                uid, bandwidth, path = line.strip().split(maxsplit=1)
                audios.append((uid, path, bandwidth))

    indices = list(range(len(audios)))
    num_digits = math.ceil(math.log(len(indices) / args.max_files, 16))

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    ret = process_map(
        partial(
            resample_to_estimated_bandwidth,
            max_files_per_dir=args.max_files,
            num_digits=num_digits,
            outdir=args.outdir,
        ),
        audios,
        indices,
        max_workers=args.nj,
        chunksize=args.chunksize,
    )

    Path(args.out_scpfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_scpfile, "w") as f:
        for uid, audio_path, fs in ret:
            f.write(f"{uid} {fs} {audio_path}\n")
