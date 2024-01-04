import argparse
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool


# Avaiable sampling rates for bandwidth limitation
SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

RESAMPLE_METHODS = (
    "kaiser_best",
    "kaiser_fast",
    "scipy",
    "polyphase",
    #    "linear",
    #    "zero_order_hold",
    #    "sinc_best",
    #    "sinc_fastest",
    #    "sinc_medium",
)

AUGMENTATIONS = ("bandwidth_limitation", "clipping")


#############################
# Augmentations per sample
#############################
def bandwidth_limitation(fs: int = 16000, res_type="random"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        fs (int): sampling rate in Hz
        res_type (str): resampling method

    Returns:
        res_type (str): adopted resampling method
        fs_new (int): effective sampling rate in Hz
    """
    # resample to a random sampling rate
    fs_opts = [fs_new for fs_new in SAMPLE_RATES if fs_new < fs]
    if fs_opts:
        if res_type == "random":
            res_type = np.random.choice(RESAMPLE_METHODS)
        fs_new = np.random.choice(fs_opts)
        opts = {"res_type": res_type}
    else:
        res_type = "none"
        fs_new = fs
    return res_type, fs_new


def weighted_sample(population, weights, k, replace=True, rng=np.random):
    weights = np.array(weights)
    weights = weights / weights.sum()
    idx = rng.choice(range(len(population)), size=k, replace=replace, p=weights)
    return [population[i] for i in idx]


#############################
# Audio utilities
#############################
def read_audio(filename, force_1ch=False, fs=None):
    audio, fs_ = sf.read(filename, always_2d=True)
    audio = audio[:, :1].T if force_1ch else audio.T
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs)
        return audio, fs
    return audio, fs_


def save_audio(audio, filename, fs):
    if audio.ndim != 1:
        audio = audio[0] if audio.shape[0] == 1 else audio.T
    sf.write(filename, audio, samplerate=fs)


#############################
# Main entry
#############################
def main(args):
    speech_dic = defaultdict(dict)
    for scp in args.speech_scps:
        with open(scp, "r") as f:
            for line in f:
                uid, fs, audio_path = line.strip().split()
                assert uid not in speech_dic[int(fs)], (uid, fs)
                speech_dic[int(fs)][uid] = audio_path

    noise_dic = defaultdict(dict)
    for scp in args.noise_scps:
        with open(scp, "r") as f:
            for line in f:
                uid, fs, audio_path = line.strip().split()
                assert uid not in noise_dic[int(fs)], (uid, fs)
                noise_dic[int(fs)][uid] = audio_path
    used_noise_dic = {fs: {} for fs in noise_dic.keys()}

    rir_dic = None
    if args.rir_scps is not None and args.prob_reverberation > 0.0:
        rir_dic = defaultdict(dict)
        for scp in args.rir_scps:
            with open(scp, "r") as f:
                for line in f:
                    uid, fs, audio_path = line.strip().split()
                    assert uid not in rir_dic[int(fs)], (uid, fs)
                    rir_dic[int(fs)][uid] = audio_path
    used_rir_dic = {fs: {} for fs in rir_dic.keys()}

    f = open(Path(args.log_dir) / "meta.tsv", "w")
    headers = ["id", "noisy_path", "speech_uid", "clean_path", "noise_uid"]
    if args.store_noise:
        headers.append("noise_path")
    headers += ["snr_dB", "rir_uid", "augmentation", "fs", "length"]
    f.write("\t".join(headers) + "\n")

    outdir = Path(args.output_dir)
    snr_range = (args.snr_low_bound, args.snr_high_bound)
    count = 0
    for fs in sorted(speech_dic.keys(), reverse=True):
        for uid, audio_path in tqdm(speech_dic[fs].items()):
            # Load speech sample (Channel, Time)
            with sf.SoundFile(audio_path) as af:
                speech_length = af.frames

            # Select an additional augmentation for each repeat
            opts = {
                "population": args.augmentations,
                "weights": args.weight_augmentations,
                "k": args.repeat_per_utt,
            }
            if args.repeat_per_utt > len(args.augmentations):
                augmentations = weighted_sample(**opts, replace=True)
            else:
                augmentations = weighted_sample(**opts, replace=False)

            for n in range(args.repeat_per_utt):
                info = process_one_sample(
                    args,
                    speech_length,
                    fs,
                    noise_dic=noise_dic,
                    used_noise_dic=used_noise_dic,
                    snr_range=snr_range,
                    store_noise=args.store_noise,
                    rir_dic=rir_dic,
                    used_rir_dic=used_rir_dic,
                    augmentation=augmentations[n],
                    force_1ch=True,
                )
                count += 1
                filename = f"fileid_{count}.wav"
                lst = [
                    f"fileid_{count}",
                    str(outdir / "noisy" / filename),
                    uid,
                    str(outdir / "clean" / filename),
                    info["noise_uid"],
                ]
                if args.store_noise:
                    lst.append(str(outdir / "noise" / filename))
                lst += [
                    str(info["snr"]),
                    info["rir_uid"],
                    info["augmentation"],
                    str(info["fs"]),
                    str(info["length"]),
                ]
                f.write("\t".join(lst) + "\n")
    f.close()


def process_one_sample(
    args,
    speech_length,
    fs,
    noise_dic,
    used_noise_dic,
    snr_range,
    store_noise=False,
    rir_dic=None,
    used_rir_dic=None,
    augmentation="none",
    force_1ch=True,
):
    # select a noise sample
    noise_uid, noise = select_sample(
        fs, noise_dic, used_sample_dic=used_noise_dic, reuse_sample=args.reuse_noise
    )
    if noise_uid is None:
        raise ValueError(f"Noise sample not found for fs={fs}+ Hz")
    snr = np.random.uniform(*snr_range)

    # select a room impulse response (RIR)
    if (
        rir_dic is None
        or args.prob_reverberation <= 0.0
        or np.random.rand() <= args.prob_reverberation
    ):
        rir_uid, rir = None, None
    else:
        rir_uid, rir = select_sample(
            fs, rir_dic, used_sample_dic=used_rir_dic, reuse_sample=args.reuse_rir
        )

    # apply an additional augmentation
    if augmentation == "none":
        pass
    elif augmentation == "bandwidth_limitation":
        res_type, fs_new = bandwidth_limitation(fs=fs, res_type="random")
        augmentation = augmentation + f"-{res_type}->{fs_new}"
    elif augmentation == "clipping":
        pass
    else:
        raise NotImplementedError(augmentation)

    meta = {
        "noise_uid": "none" if noise_uid is None else noise_uid,
        "rir_uid": "none" if rir_uid is None else rir_uid,
        "snr": snr,
        "augmentation": augmentation,
        "fs": fs,
        "length": speech_length,
    }
    return meta


def select_sample(fs, sample_dic, used_sample_dic=None, reuse_sample=False):
    """Randomly select a sample from the given dictionary.

    First try to select an unused sample with the same sampling rate (= fs).
    Then try to select an unused sample with a higher sampling rate (> fs).
    If no unused sample is found and reuse_sample=True,
        try to select a used sample with the same strategy.
    """
    if fs not in sample_dic.keys() or len(sample_dic[fs]) == 0:
        fs_opts = list(sample_dic.keys())
        np.random.shuffle(fs_opts)
        for fs2 in fs_opts:
            if fs2 > fs and len(sample_dic[fs2]) > 0:
                uid = np.random.choice(list(sample_dic[fs2].keys()))
                if used_sample_dic is not None:
                    sample = sample_dic[fs2].pop(uid)
                    used_sample_dic[fs2][uid] = sample
                else:
                    sample = sample_dic[fs2][uid]
                break
        else:
            if reuse_sample:
                return select_sample(fs, used_sample_dic, reuse_sample=False)
            return None, None
    else:
        uid = np.random.choice(list(sample_dic[fs].keys()))
        if used_sample_dic is not None:
            sample = sample_dic[fs].pop(uid)
            used_sample_dic[fs][uid] = sample
        else:
            sample = sample_dic[fs][uid]
    return uid, sample


#############################
# Commandline related
#############################
def get_parser(parser=None):
    if parser is None:

        class ArgumentDefaultsRawTextHelpFormatter(
            argparse.RawTextHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter,
        ):
            pass

        # support --config to specify all arguments in a yaml file
        parser = config_argparse.ArgumentParser(
            description="base parser",
            formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        )

    group = parser.add_argument_group(description="General arguments")
    group.add_argument(
        "--speech_scps",
        type=str,
        nargs="+",
        help="Path to the scp file containing speech samples",
    )
    group.add_argument(
        "--log_dir",
        type=str,
        help="Log directory for storing log and scp files",
    )
    group.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for storing processed audio files",
    )
    group.add_argument(
        "--repeat_per_utt",
        type=int,
        default=1,
        help="Number of times to use each utterance\n"
        "(The final amount of simulated samples will be "
        "`repeat_per_utt` * size(speech_scp))",
    )
    group.add_argument("--seed", type=int, default=0, help="Random seed")

    group = parser.add_argument_group(description="Additive noise related")
    group.add_argument(
        "--noise_scps",
        type=str,
        nargs="+",
        help="Path to the scp file containing noise samples",
    )
    group.add_argument(
        "--snr_low_bound",
        type=float,
        default=-5.0,
        help="Lower bound of signal-to-noise ratio (SNR) in dB",
    )
    group.add_argument(
        "--snr_high_bound",
        type=float,
        default=20.0,
        help="Higher bound of signal-to-noise ratio (SNR) in dB",
    )
    group.add_argument(
        "--reuse_noise",
        type=str2bool,
        default=False,
        help="Whether or not to allow reusing noise samples",
    )
    group.add_argument(
        "--store_noise",
        type=str2bool,
        default=False,
        help="Whether or not to store parallel noise samples",
    )

    group = parser.add_argument_group(description="Reverberation related")
    group.add_argument(
        "--rir_scps",
        type=str,
        nargs="+",
        help="Path to the scp file containing RIR samples\n"
        "(If not provided, reverberation will not be applied)",
    )
    group.add_argument(
        "--prob_reverberation",
        type=float,
        default=0.5,
        help="Probability of randomly adding reverberation to input speech samples",
    )
    group.add_argument(
        "--reuse_rir",
        type=str2bool,
        default=False,
        help="Whether or not to allow reusing RIR samples",
    )

    group = parser.add_argument_group(description="Additional augmentation related")
    group.add_argument(
        "--augmentations",
        default=["none"],
        nargs="+",
        choices=AUGMENTATIONS,
        help="List of mutually-exclusive augmentations to apply to input speech "
        "samples",
    )
    group.add_argument(
        "--weight_augmentations",
        type=float,
        default=[1.0],
        nargs="+",
        help="Non-zero weights of applying each augmentation option to input speech "
        "samples",
    )
    group.add_argument(
        "--clipping_min_quantile",
        type=float,
        default=0.1,
        help="Lower quantile in clipping",
    )
    group.add_argument(
        "--clipping_max_quantile",
        type=float,
        default=0.9,
        help="Higher quantile in clipping",
    )
    parser.set_defaults(required=["speech_scps", "log_dir", "output_dir", "noise_scps"])
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    if args.prob_reverberation > 0:
        assert args.rir_scps
    for w in args.weight_augmentations:
        assert w > 0.0, w
    assert len(args.weight_augmentations) == len(args.augmentations)
    assert 0.0 <= args.clipping_min_quantile < args.clipping_max_quantile <= 1.0

    outdir = Path(args.output_dir)
    (outdir / "clean").mkdir(parents=True, exist_ok=True)
    (outdir / "noisy").mkdir(parents=True, exist_ok=True)
    if args.store_noise:
        (outdir / "noise").mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)