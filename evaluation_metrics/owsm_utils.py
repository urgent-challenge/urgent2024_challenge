#!/usr/bin/env python3
import librosa

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def owsm_predict(
    model,
    speech,
    fs: int,
    src_lang: str = "eng",
    beam_size: int = 5,
    long_form: bool = False,
    text_prev: str = "",
):
    """Generate predictions using the OWSM model.

    Args:
        model (torch.nn.Module): OWSM model.
        speech (np.ndarray): speech signal < 120s (time,)
        fs (int): sampling rate in Hz.
        src_lang (str): source language in ISO 639-2 Code.
        beam_size (int): beam size used in beam search.
        long_form (bool): perform long-form decoding for audios longer than 30s.
            If an exception happens, it will fall back to standard decoding on the
            initial 30s.
        text_prev (str): generation will be conditioned on this prompt if provided.
    Returns:
        text (str): predicted text
    """
    task_sym = "<asr>"
    model.beam_search.beam_size = int(beam_size)

    assert fs == TARGET_FS, (fs, TARGET_FS)

    # Detect language using the first 30s of speech
    if src_lang == "none":
        from espnet2.bin.s2t_inference_language import Speech2Language as Speech2Lang

        speech2lang = Speech2Lang.from_pretrained(
            model_tag="espnet/owsm_v3.1_ebf",
            device=model.device,
            nbest=1,
        )
        src_lang = speech2lang(
            librosa.util.fix_length(speech, size=(TARGET_FS * CHUNK_SIZE))
        )[0][0].strip()[1:-1]
    lang_sym = f"<{src_lang}>"

    # ASR or ST
    if long_form:  # speech will be padded in decode_long()
        try:
            model.maxlenratio = -300
            utts = model.decode_long(
                speech,
                condition_on_prev_text=False,
                init_text=text_prev,
                end_time_threshold="<29.00>",
                lang_sym=lang_sym,
                task_sym=task_sym,
            )

            text = []
            for t1, t2, res in utts:
                text.append(
                    f"[{format_timestamp(seconds=t1)} --> "
                    f"{format_timestamp(seconds=t2)}] {res}"
                )
            text = "\n".join(text)

            return text
        except:
            print(
                "An exception occurred in long-form decoding. "
                "Fall back to standard decoding (only first 30s)"
            )

    # assuming 10 tokens per second
    model.maxlenratio = -min(300, int((len(speech) / TARGET_FS) * 10))

    speech = librosa.util.fix_length(speech, size=(TARGET_FS * CHUNK_SIZE))
    text = model(speech, text_prev, lang_sym=lang_sym, task_sym=task_sym)[0][-2]

    return text


# Copied from Whisper utils
def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )
