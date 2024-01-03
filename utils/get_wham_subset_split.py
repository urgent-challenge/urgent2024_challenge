from pathlib import Path


def get_subset_split(uids, data):
    return [data[uid] for uid in uids]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scp_path",
        type=str,
        required=True,
        help="Path to the scp file containing audios",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the csv file containing meta information about WHAM! noises",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to the output scp file for storing subset samples",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="Train",
        choices=("Train", "Valid", "Test"),
        help="Subset name",
    )
    args = parser.parse_args()

    uids = []
    with open(args.csv_path, "r") as f:
        headers = f.readline().strip().split(",")
        fname_idx = headers.index("Filename")
        subset_idx = headers.index("WHAM! Split")
        for line in f:
            tup = line.strip().split(",")
            if tup[subset_idx] == args.subset:
                uids.append(Path(tup[fname_idx]).stem)

    data = {}
    with open(args.scp_path, "r") as f:
        for line in f:
            uid = line.strip().split(maxsplit=1)[0]
            data[uid] = line

    ret = get_subset_split(uids, data)
    print(f"{args.subset} split: {len(ret)} samples")

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with Path(args.outfile).open("w") as f:
        for line in ret:
            f.write(line)
