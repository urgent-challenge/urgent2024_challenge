from collections import deque
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import webrtcvad
from tqdm.contrib.concurrent import process_map


def array2bytes(x: np.ndarray):
    if x.dtype.name.startswith("float"):
        return (x * 32768).astype(np.int16).tobytes()
    elif x.dtype.name.startswith("int"):
        return x.astype(np.int16).tobytes()
    else:
        raise TypeError("Invalid data type: %s" % x.dtype.name)


def generate_frame(
    bytes_array: bytes, frame_duration_ms: int, fs: int, process_tail: bool = False
):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields frames of the requested duration.

    Modified from https://github.com/wiseman/py-webrtcvad/blob/master/example.py#L45
    """
    # byte_frame_size = 2 * frame_size
    n = int(fs * (frame_duration_ms / 1000.0) * 2)
    array_length = len(bytes_array)
    assert array_length >= n, (array_length, n)
    offset = 0
    timestamp = 0.0
    # frame duration in second
    duration = (float(n) / fs) / 2.0
    while offset + n <= array_length:
        yield bytes_array[offset : offset + n], timestamp
        timestamp += duration
        offset += n
    if process_tail and array_length % n != 0:
        yield bytes_array[-n:], array_length / 2.0 / fs - duration


def merge_vad_result(
    ret,
    num_padding_frames: int,
    thres_ratio: float = 0.9,
    yield_segments: bool = False,
    frame_duration: float = 0.01,
    verbose: bool = True,
):
    """Merge VAD results following the same way as in
    https://github.com/wiseman/py-webrtcvad/blob/master/example.py#L63

    Args:
        ret (list): VAD result consisting of (frame, timestamp, is_speech)
            will be modified in-place
        num_padding_frames (int): number of frames to be stored in the buffer
        thres_ratio (float): threshold of the minimum ratio of voiced/unvoiced
            frames in the buffer when switching the state
        yield_segments (bool):
            If True, yield voiced segments;
            if False, return the refined VAD results
        frame_duration (float): frame duration in second
        verbose (bool): True to print more details
    Yields:
        voiced_segments
        or
        ret_new (list): refined VAD results
    """
    # We use a deque for our sliding window/ring buffer
    ring_buffer = deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED
    # We start in the NOTTRIGGERED state
    triggered = False

    vad_label = [is_speech for _, _, is_speech in ret]
    voiced_frames = []
    for n, (frame, timestamp, is_speech) in enumerate(ret):
        if verbose:
            print("1" if is_speech else "0", end="")
        if not triggered:
            vad_label[n] = False
            ring_buffer.append(n)
            num_voiced = len([idx for idx in ring_buffer if ret[idx][2]])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > thres_ratio * ring_buffer.maxlen:
                triggered = True
                if verbose:
                    print("+(%s)" % ret[ring_buffer[0]][1], end="")
                # Label all frames we see as speech from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for idx in ring_buffer:
                    vad_label[idx] = True
                    voiced_frames.append(ret[idx][0])
                ring_buffer.clear()
        else:
            vad_label[n] = True
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append(n)
            num_unvoiced = len([idx for idx in ring_buffer if not ret[idx][2]])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter the NOTTRIGGERED state.
            if num_unvoiced > thres_ratio * ring_buffer.maxlen:
                if verbose:
                    print("-(%s)" % (timestamp + frame_duration), end="")
                triggered = False
                if yield_segments:
                    yield b"".join(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []
    if verbose:
        if triggered:
            print("-(%s)" % (timestamp + frame_duration), end="")
        print("")
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if yield_segments and voiced_frames:
        yield b"".join(voiced_frames)
    else:
        ret_new = []
        for idx, (frame, timestamp, _) in enumerate(ret):
            ret_new.append((frame, timestamp, vad_label[idx]))
        yield ret_new


def compute_vad(
    speech,
    mode: int = 1,
    fs: int = 16000,
    frame_size: int = 10,
    ref_channel: int = 0,
    strict: bool = True,
):
    """Perform Voice Activity Detection (VAD) on the input waveform.

    Args:
        speech (array-like): waveform of shape (T,) or (T, C)
        mode (int): aggressiveness mode for VAD, an integer between 0 and 3
            0 is the least aggressive about filtering out non-speech
            3 is the most aggressive
        fs (int): sample rate in Hz, one of (8000, 16000, 32000, 48000)
        frame_size (int): frame size in millisecond, one of (10, 20, 30)
        ref_channel (int): select the specified channel if speech is multi-channel
        strict (bool):
            True to preserve frame-level VAD results;
            False to use a speech-nonspeech transition buffer to smooth VAD results
    Returns:
        vad_mask (array-like): VAD mask in the time domain (T,)
    """
    assert mode in (0, 1, 2, 3), mode
    # The WebRTC VAD only accepts 16-bit mono PCM audio,
    # sampled at 8000, 16000, 32000 or 48000 Hz.
    assert fs in (8000, 16000, 32000, 48000), fs
    # A frame must be either 10, 20, or 30 ms in duration
    assert frame_size in (10, 20, 30), frame_size
    # frame size in #samples
    frame_size_pts = int(fs * (frame_size / 1000.0))

    speech = np.asarray(speech)
    if speech.ndim == 2:
        speech = speech[..., ref_channel]
    speech_bytes = array2bytes(speech)
    assert len(speech_bytes) == 2 * len(speech), (len(speech_bytes), len(speech))

    vad = webrtcvad.Vad(mode)
    vad_result = [
        (frame, timestamp, vad.is_speech(frame, fs))
        for frame, timestamp in generate_frame(
            speech_bytes, frame_size, fs, process_tail=True
        )
    ]
    assert len(vad_result) * frame_size_pts >= len(speech), (
        len(vad_result) * frame_size_pts,
        len(speech),
    )
    assert len(vad_result) * frame_size_pts - frame_size_pts < len(speech)
    if not strict:
        vad_result = list(
            merge_vad_result(
                vad_result,
                10,
                yield_segments=False,
                frame_duration=(frame_size / 1000.0),
                verbose=False,
            )
        )[0]

    vad_mask = np.concatenate(
        [np.full(frame_size_pts, is_speech) for _, _, is_speech in vad_result]
    )[: len(speech)]
    assert len(vad_mask) == len(speech), (len(vad_mask), len(speech))
    return vad_mask


def filter_by_vad(utt_tup, vad_mode=2, threshold=0.3):
    uid, fs, audio_path = utt_tup
    wav, fs = sf.read(audio_path)
    if fs == 22050:
        wav = librosa.resample(wav, orig_sr=fs, target_sr=16000)
        fs = 16000
    if fs == 24000:
        wav = librosa.resample(wav, orig_sr=fs, target_sr=16000)
        fs = 16000
    elif fs == 44100:
        wav = librosa.resample(wav, orig_sr=fs, target_sr=48000)
        fs = 48000
    vad_mask = compute_vad(wav, mode=2, fs=fs, strict=True)
    if vad_mask.mean() < threshold:
        return None
    return uid


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
        "--outfile",
        type=str,
        required=True,
        help="Path to the output scp file for storing filtered samples",
    )
    parser.add_argument(
        "--vad_mode",
        type=int,
        default=2,
        choices=(0, 1, 2, 3),
        help="aggressiveness mode for VAD, an integer between 0 and 3",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Threshold value of speech percentages for filtering samples",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=8,
        help="Number of parallel workers to speed up simulation",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200,
        help="Chunk size used in process_map",
    )
    args = parser.parse_args()

    info = {}
    with open(args.scp_path, "r") as f:
        for line in f:
            uid, fs, audio_path = line.strip().split()
            info[uid] = (uid, fs, audio_path)

    uids = process_map(
        partial(
            filter_by_vad,
            vad_mode=args.vad_mode,
            threshold=args.threshold,
        ),
        list(info.values()),
        max_workers=args.nj,
        chunksize=args.chunksize,
    )
    uids = [uid for uid in uids if uid is not None]
    print(f"Filtering: {len(info)} samples -> {len(uids)} samples")
    filtered_uids = set(list(info.keys())).difference(set(uids))
    print(f"Filtered uids ({len(filtered_uids)}): {filtered_uids}")

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with Path(args.outfile).open("w") as f:
        for uid in uids:
            f.write(f"{uid} {info[uid][1]} {info[uid][2]}\n")

