import re
from pathlib import Path

from tqdm.contrib.concurrent import process_map


NOISE_WORD = "<NOISE>"


def normalize_transcript(txt, keep_noise=False):
    txt = txt.replace("\\", "")  # Remove backslashes. We don't need the quoting.
    txt = txt.replace(
        "%PERCENT", "PERCENT"
    )  # Normalization for Nov'93 test transcripts.
    txt = txt.replace(".POINT", "POINT")  # Normalization for Nov'93 test transcripts.
    txt = txt.replace(".PERIOD", "PERIOD")
    txt = txt.replace(",COMMA", "COMMA")
    txt = txt.replace(";SEMI-COLON", "SEMICOLON")
    txt = txt.replace('"OPEN-QUOTE', "OPEN-QUOTE")
    txt = txt.replace('"CLOSE-QUOTE', "CLOSE-QUOTE")
    txt = txt.replace('"DOUBLE-QUOTE', "DOUBLE-QUOTE")
    txt = txt.replace("-HYPHEN", "HYPHEN")
    # txt = txt.upper()  # Upcase everything to match the CMU dictionary.
    ret = []
    for w in txt.split(" "):
        if w.startswith("~"):
            # This is used to indicate truncation of an utterance.  Not a word.
            w = re.sub(r"^~+", "", w)
        if re.fullmatch(r"\[\<\w+\]", w) or re.fullmatch(r"\[\w+\>\]", w):
            # E.g. [<door_slam], this means a door slammed in the preceding word. Delete.
            # E.g. [door_slam>], this means a door slammed in the next word. Delete.
            continue
        elif re.fullmatch(r"\[\w+/\]", w) or re.fullmatch(r"\[/\w+\]", w):
            # E.g. [phone_ring/], which indicates the start of this phenomenon.
            # E.g. [/phone_ring], which indicates the end of this phenomenon.
            continue
        elif w == ".":
            # "." is used to indicate a pause.  Silence is optional anyway so not much
            # point including this in the transcript.
            continue
        elif w == "--DASH":
            # This is a common issue; the CMU dictionary has it as -DASH.
            # ret.append("-DASH")
            continue
        elif re.fullmatch(r"\[\w+\]", w):
            # Other noises, e.g. [loud_breath].
            if keep_noise:
                ret.append(NOISE_WORD)
            continue
        elif re.fullmatch(r"<([\w']+)>", w):
            # E.g. replace <and> with and (the <> means verbal deletion of a word)
            # but it's pronounced.
            match = re.fullmatch(r"<([\w']+)>", w)
            ret.append(match.group(1))
            continue

        ret.append(w)
    return " ".join(ret)


def get_transcript(txt):
    ret = []
    with txt.open("r") as f:
        for i, line in enumerate(f, 1):
            transcript = line.strip()
            match = re.fullmatch(r"^(.*)\s*\((\w+)\)$", transcript)
            if not match:
                raise ValueError(
                    f"Failed to match pattern in line {i} in {txt}: {transcript}"
                )
            transcript = match.group(1)
            uid = match.group(2)
            ret.append((uid, normalize_transcript(transcript)))
    return dict(ret)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_scp",
        type=str,
        nargs="+",
        required=True,
        help="Path to the scp files containing WSJ audio IDs in the first column",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        nargs="+",
        help="Path to the directory containing WSJ audios",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        nargs="+",
        required=True,
        help="Path to the output text files for writing transcripts for all samples",
    )
    parser.add_argument("--nj", type=int, default=8, help="Number of parallel workers")
    parser.add_argument(
        "--chunksize", type=int, default=1000, help="Chunk size for each worker"
    )
    args = parser.parse_args()
    assert len(args.audio_scp) == len(args.outfile), (args.audio_scp, args.outfile)

    all_txt = []
    for audio_dir in args.audio_dir:
        all_txt.extend(list(Path(audio_dir).rglob("*.dot")))
    ret = process_map(
        get_transcript,
        all_txt,
        chunksize=args.chunksize,
        max_workers=args.nj,
    )
    ret = {k: v for d in ret for k, v in d.items()}

    for audio_scp, outfile in zip(args.audio_scp, args.outfile):
        outdir = Path(outfile).parent
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as out:
            with open(audio_scp, "r") as f:
                for line in f:
                    uid, path = line.strip().split(maxsplit=1)
                    transcript = ret[uid]
                    out.write(f"{uid} {transcript}\n")
