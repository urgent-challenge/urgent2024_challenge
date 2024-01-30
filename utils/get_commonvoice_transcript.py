from pathlib import Path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_scp",
        type=str,
        required=True,
        help="Path to the scp file containing CommonVoice audio IDs in the 1st column",
    )
    parser.add_argument(
        "--tsv_path",
        type=str,
        required=True,
        help="Path to the tsv file containing information about the CommonVoice subset",
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

    all_txt = {}
    with open(args.tsv_path, "r") as f:
        headers = f.readline().strip().split("\t")
        fname_idx = headers.index("path")
        txt_idx = headers.index("sentence")
        for line in f:
            tup = line.strip().split("\t")
            uid = Path(tup[fname_idx]).stem
            all_txt[uid] = tup[txt_idx]

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w") as out:
        with open(args.audio_scp, "r") as f:
            for line in f:
                uid, path = line.strip().split(maxsplit=1)
                transcript = all_txt[uid]
                out.write(f"{uid} {transcript}\n")
