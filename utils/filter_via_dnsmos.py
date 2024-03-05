import json
from pathlib import Path


def filter_dnsmos(dnsmos, metrics, thresholds):
    assert len(metrics) == len(thresholds)
    ret = []
    for uid, scores in dnsmos.items():
        for metric, threshold in zip(metrics, thresholds):
            if scores[metric] < threshold:
                break
        else:
            ret.append(uid)
    return ret


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
        "--json_path",
        type=str,
        required=True,
        help="Path to the json file containing DNSMOS scores for each sample",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to the output file for storing filtered samples",
    )
    parser.add_argument(
        "--score_name",
        type=str,
        choices=("OVRL", "SIG", "BAK", "P808_MOS"),
        action="append",
        help="Path to the output file for storing filtered samples",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        help="Threshold value for filtering samples",
    )
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        dnsmos = json.load(f)

    info = {}
    with open(args.scp_path, "r") as f:
        for line in f:
            uid, fs, audio_path = line.strip().split()
            info[uid] = (fs, audio_path)

    uids = filter_dnsmos(dnsmos, args.score_name, args.threshold)
    print(f"Filtering: {len(dnsmos)} samples -> {len(uids)} samples")

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with Path(args.outfile).open("w") as f:
        for uid in uids:
            if uid in info:
                f.write(f"{uid} {info[uid][0]} {info[uid][1]}\n")
