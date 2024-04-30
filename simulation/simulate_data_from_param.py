import re
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import scipy
import soundfile as sf
from tqdm.contrib.concurrent import process_map

from espnet2.train.preprocessor import detect_non_silence

from generate_data_param import get_parser
from rir_utils import estimate_early_rir


#############################
# Augmentations per sample
#############################
def mix_noise(speech_sample, noise_sample, snr=5.0, rng=None):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (Channel, Time)
        noise_sample (np.ndarray): a single noise sample (Channel, Time)
        snr (float): signal-to-nosie ratio (SNR) in dB
        rng (np.random.Generator): random number generator
    Returns:
        noisy_sample (np.ndarray): output noisy sample (Channel, Time)
        noise (np.ndarray): scaled noise sample (Channel, Time)
    """
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        # Repeat noise
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]

    power_speech = (speech_sample[detect_non_silence(speech_sample)] ** 2).mean()
    power_noise = (noise_sample[detect_non_silence(noise_sample)] ** 2).mean()
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample
    noisy_speech = speech_sample + noise
    return noisy_speech, noise


def add_reverberation(speech_sample, rir_sample):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
    Returns:
        reverberant_sample (np.ndarray): output noisy sample (Channel, Time)
    """
    reverberant_sample = scipy.signal.convolve(speech_sample, rir_sample, mode="full")
    return reverberant_sample[:, : speech_sample.shape[1]]


def bandwidth_limitation(speech_sample, fs: int, fs_new: int, res_type="kaiser_best"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        fs (int): sampling rate in Hz
        fs_new (int): effective sampling rate in Hz
        res_type (str): resampling method

    Returns:
        ret (np.ndarray): bandwidth-limited speech sample (1, Time)
    """
    opts = {"res_type": res_type}
    if fs == fs_new:
        return speech_sample
    assert fs > fs_new, (fs, fs_new)
    ret = librosa.resample(speech_sample, orig_sr=fs, target_sr=fs_new, **opts)
    # resample back to the original sampling rate
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return ret[:, : speech_sample.shape[1]]


def clipping(speech_sample, min_quantile: float = 0.0, max_quantile: float = 0.9):
    """Apply the clipping distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        min_quantile (float): lower bound on the quantile of samples to be clipped
        max_quantile (float): upper bound on the quantile of samples to be clipped

    Returns:
        ret (np.ndarray): clipped speech sample (1, Time)
    """
    q = np.array([min_quantile, max_quantile])
    min_, max_ = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    # per-channel clipping
    ret = np.stack(
        [
            np.clip(speech_sample[i], min_[i], max_[i])
            for i in range(speech_sample.shape[0])
        ],
        axis=0,
    )
    return ret


#############################
# Audio utilities
#############################
def read_audio(filename, force_1ch=False, fs=None):
    audio, fs_ = sf.read(filename, always_2d=True)
    audio = audio[:, :1].T if force_1ch else audio.T
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
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
    speech_dic = {}
    for scp in args.speech_scps:
        with open(scp, "r") as f:
            for line in f:
                uid, fs, audio_path = line.strip().split()
                assert uid not in speech_dic, (uid, fs)
                speech_dic[uid] = audio_path

    noise_dic = {}
    for scp in args.noise_scps:
        with open(scp, "r") as f:
            for line in f:
                uid, fs, audio_path = line.strip().split()
                assert uid not in noise_dic, (uid, fs)
                noise_dic[uid] = audio_path
    noise_dic = dict(noise_dic)

    rir_dic = None
    if args.rir_scps is not None:
        rir_dic = {}
        for scp in args.rir_scps:
            with open(scp, "r") as f:
                for line in f:
                    uid, fs, audio_path = line.strip().split()
                    assert uid not in rir_dic, (uid, fs)
                    rir_dic[uid] = audio_path
    rir_dic = dict(rir_dic)

    meta = []
    with open(Path(args.log_dir) / "meta.tsv", "r") as f:
        headers = next(f).strip().split("\t")
        for line in f:
            meta.append(dict(zip(headers, line.strip().split("\t"))))
    process_map(
        partial(
            process_one_sample,
            store_noise=args.store_noise,
            speech_dic=speech_dic,
            noise_dic=noise_dic,
            rir_dic=rir_dic,
        ),
        meta,
        max_workers=args.nj,
        chunksize=args.chunksize,
    )


def process_one_sample(
    info,
    force_1ch=True,
    store_noise=False,
    speech_dic=None,
    noise_dic=None,
    rir_dic=None,
):
    uid = info["id"]
    fs = int(info["fs"])
    snr = float(info["snr_dB"])

    speech = speech_dic[info["speech_uid"]]
    noise = noise_dic[info["noise_uid"]]
    speech_sample = read_audio(speech, force_1ch=force_1ch, fs=fs)[0]
    noise_sample = read_audio(noise, force_1ch=force_1ch, fs=fs)[0]

    rir_uid = info["rir_uid"]
    if rir_uid != "none":
        rir = rir_dic[rir_uid]
        rir_sample = read_audio(rir, force_1ch=force_1ch, fs=fs)[0]
        noisy_speech = add_reverberation(speech_sample, rir_sample)
        # make sure the clean speech is aligned with the input noisy speech
        early_rir_sample = estimate_early_rir(rir_sample, fs=fs)
        speech_sample = add_reverberation(speech_sample, early_rir_sample)
    else:
        noisy_speech = speech_sample

    rng = np.random.default_rng(int(uid.split("_")[-1]))
    noisy_speech, noise_sample = mix_noise(
        noisy_speech, noise_sample, snr=snr, rng=rng
    )

    augmentation = info["augmentation"]
    # apply an additional augmentation
    if augmentation == "none":
        pass
    elif augmentation.startswith("bandwidth_limitation"):
        match = re.fullmatch(f"bandwidth_limitation-(.*)->(\d+)", augmentation)
        res_type, fs_new = match.groups()
        noisy_speech = bandwidth_limitation(
            noisy_speech, fs=fs, fs_new=int(fs_new), res_type=res_type
        )
    elif augmentation.startswith("clipping"):
        match = re.fullmatch(f"clipping\(min=(.*),max=(.*)\)", augmentation)
        min_, max_ = map(float, match.groups())
        noisy_speech = clipping(noisy_speech, min_quantile=min_, max_quantile=max_)
    else:
        raise NotImplementedError(augmentation)

    length = int(info["length"])
    assert noisy_speech.shape[-1] == length, (info, noisy_speech.shape)

    # normalization
    scale = 0.9 / max(
        np.max(np.abs(noisy_speech)),
        np.max(np.abs(speech_sample)),
        np.max(np.abs(noise_sample)),
    )

    save_audio(speech_sample * scale, info["clean_path"], fs)
    save_audio(noisy_speech * scale, info["noisy_path"], fs)
    if store_noise:
        save_audio(noise_sample * scale, info["noise_path"], fs)


if __name__ == "__main__":
    parser = get_parser()
    group = parser.add_argument_group(description="New arguments")
    group.add_argument(
        "--meta_tsv",
        type=str,
        required=True,
        help="Path to the tsv file containing meta information for simulation",
    )
    group.add_argument(
        "--nj",
        type=int,
        default=8,
        help="Number of parallel workers to speed up simulation",
    )
    group.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="Chunk size used in process_map",
    )
    args = parser.parse_args()
    print(args)

    main(args)
