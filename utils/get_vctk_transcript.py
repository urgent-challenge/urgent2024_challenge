from pathlib import Path

from tqdm.contrib.concurrent import process_map


def get_transcript(txt):
    uid = txt.stem
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
        help="Path to the scp file containing VCTK audio IDs in the first column",
    )
    parser.add_argument(
        "--vctk_dir",
        type=str,
        required=True,
        help="Path to the root directory of the VCTK corpus",
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

    txt_dir = Path(args.vctk_dir) / "txt"
    assert txt_dir.exists()
    all_txt = list(txt_dir.rglob("*.txt"))
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
                if uid.startswith("p315"):
                    print(
                        f"[uid={uid}] No text is available for speaker 'p315'. "
                        "Use empty text instead"
                    )
                    out.write(f"{uid} <not-available>\n")
                    continue
                if uid.endswith("_mic1") or uid.endswith("_mic2"):
                    transcript = ret[uid[:-5]]
                else:
                    transcript = ret[uid]
                out.write(f"{uid} {transcript}\n")
