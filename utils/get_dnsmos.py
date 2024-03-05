import json
from distutils.util import strtobool
from pathlib import Path

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

# The latest espnet is required
from espnet2.enh.layers.dnsmos import DNSMOS_local


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def get_dnsmos(data, dnsmos_model):
    """Expected dictionary structure:

    {
        "speaker_id/youtube_id/audio_id(index)": {
            "audio_path": "/path/to/speaker_id/youtube_id/audio_id.wav",
            "text": "the idea of them tying me to a tree and you're not able to",
            "start": 0.009,
            "end": 4.565,
            "words": [
                {
                    "word": "the", "start": 0.009, "end": 0.17, "score": 0.995
                },
                ...,
                {
                    "word": "to", "start": 2.036, "end": 4.302, "score": 0.747
                }
            ],
            "language": "en",
        },
        ...
    }
    """
    dnsmos = {}
    prev_audio_path, prev_audio = "", None
    for uid, dic in tqdm(data.items()):
        if prev_audio_path == dic["audio_path"]:
            assert prev_audio is not None
        else:
            audio, fs = sf.read(dic["audio_path"])
            if fs != 16000:
                audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
            prev_audio_path = dic["audio_path"]
            prev_audio = audio
        if "start" in dic and "end" in dic:
            st, et = int(fs * dic["start"]), int(fs * dic["end"])
            with torch.no_grad():
                dnsmos_score = dnsmos_model(audio[st:et], fs)
        else:
            with torch.no_grad():
                dnsmos_score = dnsmos_model(audio, fs)
        dnsmos[uid] = {
            f"OVRL": float(dnsmos_score["OVRL"]),
            f"SIG": float(dnsmos_score["SIG"]),
            f"BAK": float(dnsmos_score["BAK"]),
            f"P808_MOS": float(dnsmos_score["P808_MOS"]),
        }
    return dnsmos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Audio related")
    group.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the json file containing audio information of VoxCeleb data",
    )
    group.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to the output json file for writing the DNSMOS information",
    )

    group = parser.add_argument_group("DNSMOS related")
    group.add_argument(
        "--use_gpu",
        type=str2bool,
        default=False,
        help="used when dnsmsos_mode='local'",
    )
    group.add_argument(
        "--convert_to_torch",
        type=str2bool,
        default=False,
        help="used when dnsmsos_mode='local'",
    )
    group.add_argument(
        "--primary_model",
        type=str,
        default="./DNSMOS/sig_bak_ovr.onnx",
        help="Path to the primary DNSMOS model. Required if dnsmsos_mode='local'",
    )
    group.add_argument(
        "--p808_model",
        type=str,
        default="./DNSMOS/model_v8.onnx",
        help="Path to the p808 model. Required if dnsmsos_mode='local'",
    )

    group = parser.add_argument_group("Parallel worker related")
    group.add_argument("--nsplits", type=int, default=1, help="Total number of workers")
    group.add_argument(
        "--job", type=int, default=1, help="Worker index of the current job"
    )
    args = parser.parse_args()

    if not Path(args.primary_model).exists():
        raise ValueError(
            f"The primary model '{args.primary_model}' doesn't exist."
            " You can download the model from https://github.com/microsoft/"
            "DNS-Challenge/tree/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
        )
    if not Path(args.p808_model).exists():
        raise ValueError(
            f"The P808 model '{args.p808_model}' doesn't exist."
            " You can download the model from https://github.com/microsoft/"
            "DNS-Challenge/tree/master/DNSMOS/DNSMOS/model_v8.onnx"
        )
    dnsmos_model = DNSMOS_local(
        args.primary_model,
        args.p808_model,
        use_gpu=args.use_gpu,
        convert_to_torch=args.convert_to_torch,
    )

    if args.json_path.endswith(".scp"):
        data = {}
        with open(args.json_path, "r") as f:
            for line in f:
                tup = line.strip().split()
                if len(tup) == 2:
                    uid, audio_path = tup
                elif len(tup) == 3:
                    uid, fs, audio_path = tup
                else:
                    raise ValueError("Unrecognized data format: %s" % line)
                data[uid] = {"audio_path": audio_path}
    elif args.json_path.endswith(".json"):
        with open(args.json_path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError("Unrecognized file extension: %s" % args.json_path)
    keys = list(data.keys())
    size = len(keys)
    assert 1<= args.job <= args.nsplits <= size, (args.nsplits, args.job, size)
    interval = size // args.nsplits
    start = (args.job - 1) * interval
    end = size if args.job == args.nsplits else start + interval
    data = {k: data[k] for k in keys[start:end]}
    print(f"Processing ({len(data)}/{size}) samples", flush=True)

    out_json = get_dnsmos(data, dnsmos_model)
    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with Path(args.outfile).open("w") as f:
        json.dump(out_json, f)
