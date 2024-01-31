import argparse
import concurrent.futures
import io
import multiprocessing
import os
import json
import datetime
from pathlib import Path

import soxr
import torch
import torchaudio
import webdataset
from tqdm import tqdm

# torchaudio.set_audio_backend("soundfile")

mp_context = multiprocessing.get_context("spawn")


class Resampler:
    def __init__(self, new_freq):
        self.new_freq = new_freq

    def __call__(self, x, fs):
        if self.new_freq is None:
            return x, fs

        if fs == self.new_freq:
            return x, fs

        x_npy = x.numpy()
        if x_npy.ndim == 2:
            x_npy = x_npy.T
        x_npy = soxr.resample(x_npy, fs, self.new_freq)
        if x_npy.ndim == 2:
            x_npy = x_npy.T
        x = torch.from_numpy(x_npy)

        return x, self.new_freq


def pack_for_wds(key, audio, fs):
    audio_buffer = io.BytesIO()
    torchaudio.save(
        audio_buffer,
        audio,
        fs,
        bits_per_sample=16,
        encoding="PCM_S",
        format="wav",
    )
    audio_buffer.seek(0)
    audio_bytes = audio_buffer.read()

    return {
        "__key__": key,
        "audio.wav": audio_bytes,
    }


def write_block(
    files,
    block_idx,
    output_path,
    samplerate,
    keep_channels,
):
    infos = {}

    resampler = Resampler(new_freq=samplerate)

    tar_num = f"{block_idx:04d}.tar"
    with webdataset.TarWriter(str(output_path / tar_num)) as sink:
        for idx, (key, path) in enumerate(files):
            audio, fs = torchaudio.load(path)

            if not keep_channels:
                audio = audio[:1, :]

            if samplerate is not None:
                if samplerate < fs:
                    audio, fs = resampler(audio, fs)
                else:
                    # skip files that will result in data loss (e.g. 16kHz -> 24kHz)
                    continue

            sink.write(pack_for_wds(key, audio, fs))

            infos[key] = {
                "tar": f"{output_path.parent}/{tar_num}",
                "original": str(path),
                "len_s": audio.shape[-1] / fs,
            }
    return infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a dataset summarized in .scp files into WebDataset format"
    )
    parser.add_argument(
        "--scps", nargs="+", required=True, type=Path, help="path to scp files"
    )
    parser.add_argument("output_path", type=Path, help="path to output directory")
    parser.add_argument(
        "--sampling-rate",
        default=24000,
        type=int,
        help="Target sampling rate. "
        "If provided, files with higher sampling rate will be resampled to the target. "
        "Those with lower sampling rate will be discarded.",
    )
    parser.add_argument("--keep-channels", action="store_true")
    parser.add_argument("--files-per-tar", default=250, type=int)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    parser.add_argument("--max-tars", type=int)
    args = parser.parse_args()

    meta_dir = args.output_path / "metadata"
    meta_dir.mkdir(exist_ok=True, parents=True)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers, mp_context=mp_context
    ) as executor:
        future_to_block = {}

        if args.max_tars is not None:
            num_blocks = args.max_tars

        infos = {}
        meta_files = {}
        num_tars = 0
        lengths_s = {"total": 0.0}

        for scp in args.scps:
            scp_name = scp.stem
            shard_dir = args.output_path / f"{scp_name}"
            shard_dir.mkdir(exist_ok=True, parents=True)
            meta_files[scp_name] = meta_dir / f"{scp_name}.json"
            infos[scp_name] = {}
            lengths_s[scp_name] = 0.0

            with open(scp) as f:
                # initialize
                block_idx = 0
                files = []

                for line in tqdm(f, desc=f"reading {scp_name}"):
                    key, fs, path = line.strip().split()

                    # skip sample with lower sampling rate
                    fs = int(fs)
                    if args.sampling_rate is not None and fs < args.sampling_rate:
                        continue

                    # add for processing
                    files.append((key, path))

                    if len(files) == args.files_per_tar:
                        # batch ready for packing into tar
                        future = executor.submit(
                            write_block,
                            files,
                            block_idx,
                            shard_dir,
                            args.sampling_rate,
                            args.keep_channels,
                        )
                        future_to_block[future] = (scp_name, block_idx)

                        # update
                        files = []
                        block_idx += 1
                        num_tars += 1

                        if num_tars == args.max_tars:
                            break

                if len(files) > 0:
                    future = executor.submit(
                        write_block,
                        files,
                        block_idx,
                        shard_dir,
                        args.sampling_rate,
                        args.keep_channels,
                    )
                    future_to_block[future] = (scp_name, block_idx)
                    num_tars += 1

            if num_tars == args.max_tars:
                break

        for future in tqdm(
            concurrent.futures.as_completed(future_to_block),
            total=len(future_to_block),
            desc="waiting...",
        ):
            (scp_name, block) = future_to_block[future]
            try:
                block_infos = future.result()
                block_time = sum([info["len_s"] for info in block_infos.values()])
                infos[scp_name].update(block_infos)
                lengths_s[scp_name] += block_time
                lengths_s["total"] += block_time
            except Exception as exc:
                print(f"{block} generated an exception: {exc}")
            else:
                # print(f"{split} block {block} successfully terminated")
                pass

        for scp_name, data in infos.items():
            with open(meta_files[scp_name], "w") as f:
                json.dump(infos[scp_name], f, indent=2)
        for scp_name, len_s in lengths_s.items():
            with open(meta_dir / f"{scp_name}-length.json", "w") as f:
                dt = datetime.timedelta(seconds=int(len_s))
                json.dump({"length": str(dt)}, f, indent=2)
