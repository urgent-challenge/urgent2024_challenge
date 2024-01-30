import json
import re
from datetime import datetime, timedelta
from math import inf
from pathlib import Path


def get_subset_split(segments, data, min_duration=3.0, silence_thres=2.0):
    dic = {}
    orig_dur, final_dur = 0.0, 0.0
    for group in group_by_sample(segments):
        video_id = group[0]
        if video_id in data:
            dur_before, dur_after = segment_one_sample(
                dic,
                video_id,
                merge_overlapped_segments(group[1:]),
                data[video_id],
                min_duration=min_duration,
                silence_thres=silence_thres,
            )
            orig_dur += dur_before
            final_dur += dur_after
        else:
            raise ValueError(f"Could not find '{video_id}' in the scp file")
    orig_dur = timedelta(seconds=orig_dur)
    final_dur = timedelta(seconds=final_dur)
    print(f"Total duration: {orig_dur} -> {final_dur}")
    return dic


def group_by_sample(segments):
    """Group all segments by their sample ID (video_id).

    This function assumes that `segments` is sorted by video_id.
    """
    ret = []
    prev = None
    for segment in segments:
        annotation_id, video_id = segment[:2]
        if prev is None or prev[0] != video_id:
            if prev is not None:
                ret.append(prev)
            prev = [video_id, segment[2:]]
        else:
            prev.append(segment[2:])
    if prev is not None:
        ret.append(prev)
    return ret


def merge_overlapped_segments(segments):
    """Merged overlapped segments into one."""
    ret = []
    for start, end in sorted(segments, key=lambda x: x[0]):
        if len(ret) == 0:
            ret.append((start, end))
        if ret[-1][0] <= start <= ret[-1][1]:
            ret[-1] = (ret[-1][0], max(ret[-1][1], end))
        else:
            ret.append((start, end))
    return ret


def segment_one_sample(
    dic,
    uid,
    group,
    audio_path,
    min_duration=1.0,
    max_duration=inf,
    silence_thres=0.5,
    segment_duration_thres=inf,
):
    i = 1
    dur_after = 0.0
    st, prev_et = None, None
    segments = []
    for seg_info in group:
        start, end = seg_info
        if st is None:
            st = start
        if prev_et is None:
            prev_et = end
        dur = end - start
        if dur > segment_duration_thres or start - prev_et > silence_thres:
            # too long for a single segment or too long silence
            if segments and prev_et - st >= min_duration:
                dic[f"{uid}({i})"] = prepare_segment(audio_path, st, prev_et, segments)
                dur_after += (dic[f"{uid}({i})"]["end"] - dic[f"{uid}({i})"]["start"])
                i += 1
            st, prev_et = start, end
            segments = [seg_info]
        elif end > prev_et and end - st > max_duration:
            # current sample gets too long if this segment is added
            if segments and prev_et - st >= min_duration:
                dic[f"{uid}({i})"] = prepare_segment(audio_path, st, prev_et, segments)
                dur_after += (dic[f"{uid}({i})"]["end"] - dic[f"{uid}({i})"]["start"])
                i += 1
            st, prev_et = start, end
            segments = [seg_info]
        else:
            segments.append(seg_info)
            prev_et = end

    dur = segments[-1][1] - segments[0][0] if segments else -1
    if dur > min_duration:
        dic[f"{uid}({i})"] = prepare_segment(
            audio_path, segments[0][0], segments[-1][1], segments
        )
        dur_after += (dic[f"{uid}({i})"]["end"] - dic[f"{uid}({i})"]["start"])
        i += 1

    dur_before = group[-1][1] - group[0][0]
    return dur_before, dur_after


def prepare_segment(audio_path, start, end, segments):
    assert end > start, (start, end)
    assert start == segments[0][0], (start, segments[0][0])
    assert end == segments[-1][1], (end, segments[-1][1])
    return {"audio_path": audio_path, "start": start, "end": end}


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
        help="Path to the csv file containing information about the EPIC-SOUNDS subset",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to the output scp file for storing subset samples",
    )
    args = parser.parse_args()

    segments = []
    count = 0
    t0 = datetime.strptime("00:00:00.0", "%H:%M:%S.%f")
    with open(args.csv_path, "r") as f:
        headers = f.readline().strip().split(",")
        aid_idx = headers.index("annotation_id")
        vid_idx = headers.index("video_id")
        sts_idx = headers.index("start_timestamp")
        ets_idx = headers.index("stop_timestamp")
        # st_idx = headers.index("start_sample")
        # et_idx = headers.index("stop_sample")
        desc_idx = headers.index("description")
        cls_idx = headers.index("class")
        for line in f:
            tup = line.strip().split(",")
            description = tup[desc_idx]
            if tup[cls_idx] == "human" and description in ("human", "unlabelled"):
                count += 1
                continue
            if re.search(
                r"\b(speech|chat|speak|talk|voice)", description, re.IGNORECASE
            ):
                count += 1
                continue
            st = (datetime.strptime(tup[sts_idx], "%H:%M:%S.%f") - t0).total_seconds()
            et = (datetime.strptime(tup[ets_idx], "%H:%M:%S.%f") - t0).total_seconds()
            segments.append(
                (
                    tup[aid_idx],
                    tup[vid_idx],
                    st,
                    et,
                    # int(tup[st_idx]),
                    # int(tup[et_idx]),
                )
            )
    print(f"Filtered {count} segments that may contain human speech")

    data = {}
    with open(args.scp_path, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split(maxsplit=1)
            data[uid] = audio_path

    ret = get_subset_split(segments, data)
    print(f"New split: {len(ret)} segmented samples")

    outdir = Path(args.outfile).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with Path(args.outfile).open("w") as f:
        json.dump(ret, f, indent=2)
