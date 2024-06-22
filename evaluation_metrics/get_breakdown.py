from collections import defaultdict
import json

import numpy as np


#############################
# Group function definitions
#############################
def group_by_fs(meta):
    """Return a unique group id based on the fs in the given meta dictionary."""
    # group by sampling rate
    fs = int(meta["fs"])
    return f"fs={fs}Hz"


def group_by_snr(meta):
    """Return a unique group id based on SNR in the given meta dictionary."""
    # group every 5 dB
    snr = int(float(meta["snr_dB"]) / 5) * 5
    return f"snr={snr:02d}dB"


def group_by_duration(meta):
    """Return a unique group id based on the duration in the given meta dictionary."""
    # group by sample length (every 5s)
    length = int((float(meta["length"]) / float(meta["fs"])) / 5) * 5
    return f"duration={length:02d}s"


def group_by_corpus(meta):
    """Return a unique group id based on the corpus in the given meta dictionary."""
    # group by corpus name (prefix of speech_sid)
    corpus = meta["speech_sid"].split("_", maxsplit=1)[0]
    return f"corpus={corpus}"


def group_by_rir(meta):
    """Return a unique group id based on the RIR in the given meta dictionary."""
    # group by whether RIR is applied or not
    rir = meta["rir_uid"] != "none"
    return "with_rir" if rir else "no_rir"


def group_by_augmentation(meta):
    """Return a unique group id based on augmentation in the given meta dictionary."""
    # group by augmentation type
    augmentation = meta["augmentation"]
    if augmentation.startswith("bandwidth_limitation"):
        return "bandwidth_limitation"
    if augmentation.startswith("clipping"):
        return "clipping"
    return augmentation


def group_score_by_field(result_dic, meta_dic, group_func):
    """Group scores of different samples in `result_dic`
    by the value in the given field in `meta_dic`.

    Args:
        result_dic (dict): A dictionary containing scores of different samples.
        meta_dic (dict): A dictionary containing meta information of different samples.
        group_func (callable): A function defining how to group the samples.
    Returns:
        ret (dict): A dictionary of groups of scores.
    """
    ret = defaultdict(list)
    for uid, score in result_dic.items():
        group = group_func(meta_dic[uid])
        ret[group].append(score)
    return dict(ret)


#############################
# Main entry
#############################
def main(args):
    result_dic = {}
    is_wer = None
    with open(args.result_scp, "r") as f:
        for line in f:
            uid, score = line.strip().split(maxsplit=1)
            if is_wer is None:
                try:
                    float(score)
                    is_wer = False
                except ValueError:
                    is_wer = True
            score = json.loads(score) if is_wer else float(score)
            if not uid.startswith("fileid"):
                uid = "fileid" + uid.split("fileid", maxsplit=1)[1]
            result_dic[uid] = score

    meta_dic = {}
    with open(args.meta_tsv, "r") as f:
        headers = next(f).strip().split("\t")
        for line in f:
            tup = line.strip().split("\t")
            meta_dic[tup[0]] = dict(zip(headers, tup))

    for group_func in (
        group_by_fs,
        group_by_snr,
        group_by_duration,
        group_by_corpus,
        group_by_rir,
        group_by_augmentation,
    ):
        group_func_name = group_func.__name__.replace("group_by_", "")
        groups = group_score_by_field(result_dic, meta_dic, group_func)
        if group_func in (group_by_fs, group_by_snr, group_by_duration):
            groups = {k: groups[k] for k in sorted(groups.keys())}
        print(f"\n====== Group by {group_func_name} =====\n")
        for group, dic in groups.items():
            msg = f"[Group] {group}\n\t"
            msg += get_average_score(dic, is_wer=is_wer)
            print(msg)


def get_average_score(result_lst, is_wer=False):
    """Return the average score of all samples in the given result list."""
    if is_wer:
        dic = {"delete": 0, "insert": 0, "replace": 0, "equal": 0}
        for score in result_lst:
            for k in dic.keys():
                dic[k] = dic[k] + score[k]
        numerator = dic["replace"] + dic["delete"] + dic["insert"]
        denominator = dic["replace"] + dic["delete"] + dic["equal"]
        wer = numerator / denominator
        msg = f"WER: {wer:.4f}\n"
        for op, count in dic.items():
            msg += f"    {op}: {count}\n"
    else:
        score = np.nanmean(result_lst)
        msg = f"Average score: {score}\n"
    return msg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_scp",
        type=str,
        help="Path to the scp file containing evaluation result of each sample.",
    )
    parser.add_argument(
        "--meta_tsv",
        type=str,
        required=True,
        help="Path to the tsv file containing meta information about each sample",
    )
    args = parser.parse_args()

    main(args)
