from pathlib import Path

from tqdm.contrib.concurrent import process_map


def get_transcript(txt):
    uid = txt.name[:-15]
    with txt.open("r") as f:
        transcript = f.read().strip()
    return uid, transcript


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_scp",
        type=str,
        required=True,
        help="Path to the scp file containing LibriTTS audio IDs in the first column",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        nargs="+",
        help="Path to the directory containing LibriTTS audios",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to the output text file for writing transcripts for all samples",
    )
    parser.add_argument("--nj", type=int, default=8, help="Number of parallel workers")
    parser.add_argument(
        "--chunksize", type=int, default=1000, help="Chunk size for each worker"
    )
    args = parser.parse_args()

    all_txt = []
    for audio_dir in args.audio_dir:
        all_txt.extend(list(Path(audio_dir).rglob("*.normalized.txt")))
    ret = process_map(
        get_transcript,
        all_txt,
        chunksize=args.chunksize,
        max_workers=args.nj,
    )
    ret = dict(ret)

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w") as out:
        with open(args.audio_scp, "r") as f:
            for line in f:
                uid, path = line.strip().split(maxsplit=1)
                transcript = ret[uid]
                out.write(f"{uid} {transcript}\n")
