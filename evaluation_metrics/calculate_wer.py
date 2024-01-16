import torch
import librosa

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.bin.s2t_inference_language import Speech2Text as Speech2Lang

if not torch.cuda.is_available():
    raise RuntimeError("Please use GPU for better speed")

# model_path = "owsm_v3/exp/s2t_train_s2t_transformer_conv2d_size1024_e24_d24_lr2.5e-4_warmup10k_finetune_raw_bpe50000/valid.acc.ave_5best.till50epoch.pth"
model_path = "owsm_v3.1_ebf/exp/s2t_train_s2t_ebf_conv2d_size1024_e18_d18_piecewise_lr2e-4_warmup60k_flashattn_raw_bpe50000/valid.total_count.ave_5best.till45epoch.pth"
device = "cuda"  # if torch.cuda.is_available() else "cpu"

model_path_or_tag = "espnet/owsm_v3.1_ebf"
speech2text = Speech2Text.from_pretrained(
    s2t_model_file=model_path_or_tag,
    device=device,
    category_sym="<eng>",
    beam_size=5,
    # quantize_s2t_model=not torch.cuda.is_available(),
    # quantize_dtype="float16",
)

speech2lang = Speech2Lang.from_pretrained(
    s2t_model_file=model_path_or_tag,
    device=device,
    nbest=1,
    # quantize_s2t_model=not torch.cuda.is_available(),
    # quantize_dtype="float16",
)


iso_codes = ['abk', 'afr', 'amh', 'ara', 'asm', 'ast', 'aze', 'bak', 'bas', 'bel', 'ben', 'bos', 'bre', 'bul', 'cat', 'ceb', 'ces', 'chv', 'ckb', 'cmn', 'cnh', 'cym', 'dan', 'deu', 'dgd', 'div', 'ell', 'eng', 'epo', 'est', 'eus', 'fas', 'fil', 'fin', 'fra', 'frr', 'ful', 'gle', 'glg', 'grn', 'guj', 'hat', 'hau', 'heb', 'hin', 'hrv', 'hsb', 'hun', 'hye', 'ibo', 'ina', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kab', 'kam', 'kan', 'kat', 'kaz', 'kea', 'khm', 'kin', 'kir', 'kmr', 'kor', 'lao', 'lav', 'lga', 'lin', 'lit', 'ltz', 'lug', 'luo', 'mal', 'mar', 'mas', 'mdf', 'mhr', 'mkd', 'mlt', 'mon', 'mri', 'mrj', 'mya', 'myv', 'nan', 'nep', 'nld', 'nno', 'nob', 'npi', 'nso', 'nya', 'oci', 'ori', 'orm', 'ory', 'pan', 'pol', 'por', 'pus', 'quy', 'roh', 'ron', 'rus', 'sah', 'sat', 'sin', 'skr', 'slk', 'slv', 'sna', 'snd', 'som', 'sot', 'spa', 'srd', 'srp', 'sun', 'swa', 'swe', 'swh', 'tam', 'tat', 'tel', 'tgk', 'tgl', 'tha', 'tig', 'tir', 'tok', 'tpi', 'tsn', 'tuk', 'tur', 'twi', 'uig', 'ukr', 'umb', 'urd', 'uzb', 'vie', 'vot', 'wol', 'xho', 'yor', 'yue', 'zho', 'zul']
lang_names = ['Abkhazian', 'Afrikaans', 'Amharic', 'Arabic', 'Assamese', 'Asturian', 'Azerbaijani', 'Bashkir', 'Basa (Cameroon)', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Catalan', 'Cebuano', 'Czech', 'Chuvash', 'Central Kurdish', 'Mandarin Chinese', 'Hakha Chin', 'Welsh', 'Danish', 'German', 'Dagaari Dioula', 'Dhivehi', 'Modern Greek (1453-)', 'English', 'Esperanto', 'Estonian', 'Basque', 'Persian', 'Filipino', 'Finnish', 'French', 'Northern Frisian', 'Fulah', 'Irish', 'Galician', 'Guarani', 'Gujarati', 'Haitian', 'Hausa', 'Hebrew', 'Hindi', 'Croatian', 'Upper Sorbian', 'Hungarian', 'Armenian', 'Igbo', 'Interlingua (International Auxiliary Language Association)', 'Indonesian', 'Icelandic', 'Italian', 'Javanese', 'Japanese', 'Kabyle', 'Kamba (Kenya)', 'Kannada', 'Georgian', 'Kazakh', 'Kabuverdianu', 'Khmer', 'Kinyarwanda', 'Kirghiz', 'Northern Kurdish', 'Korean', 'Lao', 'Latvian', 'Lungga', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Ganda', 'Luo (Kenya and Tanzania)', 'Malayalam', 'Marathi', 'Masai', 'Moksha', 'Eastern Mari', 'Macedonian', 'Maltese', 'Mongolian', 'Maori', 'Western Mari', 'Burmese', 'Erzya', 'Min Nan Chinese', 'Nepali (macrolanguage)', 'Dutch', 'Norwegian Nynorsk', 'Norwegian Bokmål', 'Nepali (individual language)', 'Pedi', 'Nyanja', 'Occitan (post 1500)', 'Oriya (macrolanguage)', 'Oromo', 'Odia', 'Panjabi', 'Polish', 'Portuguese', 'Pushto', 'Ayacucho Quechua', 'Romansh', 'Romanian', 'Russian', 'Yakut', 'Santali', 'Sinhala', 'Saraiki', 'Slovak', 'Slovenian', 'Shona', 'Sindhi', 'Somali', 'Southern Sotho', 'Spanish', 'Sardinian', 'Serbian', 'Sundanese', 'Swahili (macrolanguage)', 'Swedish', 'Swahili (individual language)', 'Tamil', 'Tatar', 'Telugu', 'Tajik', 'Tagalog', 'Thai', 'Tigre', 'Tigrinya', 'Toki Pona', 'Tok Pisin', 'Tswana', 'Turkmen', 'Turkish', 'Twi', 'Uighur', 'Ukrainian', 'Umbundu', 'Urdu', 'Uzbek', 'Vietnamese', 'Votic', 'Wolof', 'Xhosa', 'Yoruba', 'Yue Chinese', 'Chinese', 'Zulu']

task_codes = ['asr', 'st_ara', 'st_cat', 'st_ces', 'st_cym', 'st_deu', 'st_eng', 'st_est', 'st_fas', 'st_fra', 'st_ind', 'st_ita', 'st_jpn', 'st_lav', 'st_mon', 'st_nld', 'st_por', 'st_ron', 'st_rus', 'st_slv', 'st_spa', 'st_swe', 'st_tam', 'st_tur', 'st_vie', 'st_zho']
task_names = ['Automatic Speech Recognition', 'Translate to Arabic', 'Translate to Catalan', 'Translate to Czech', 'Translate to Welsh', 'Translate to German', 'Translate to English', 'Translate to Estonian', 'Translate to Persian', 'Translate to French', 'Translate to Indonesian', 'Translate to Italian', 'Translate to Japanese', 'Translate to Latvian', 'Translate to Mongolian', 'Translate to Dutch', 'Translate to Portuguese', 'Translate to Romanian', 'Translate to Russian', 'Translate to Slovenian', 'Translate to Spanish', 'Translate to Swedish', 'Translate to Tamil', 'Translate to Turkish', 'Translate to Vietnamese', 'Translate to Chinese']

lang2code = dict(
    [("Unknown", "none")] + sorted(list(zip(lang_names, iso_codes)), key=lambda x: x[0])
)
task2code = dict(sorted(list(zip(task_names, task_codes)), key=lambda x: x[0]))

code2lang = dict([(v, k) for k, v in lang2code.items()])


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


def predict(
    audio_path,
    src_lang: str,
    task: str,
    beam_size,
    long_form: bool,
    text_prev: str,
):
    speech2text.task_id = speech2text.converter.token2id[f"<{task2code[task]}>"]
    speech2text.beam_search.beam_size = int(beam_size)

    # Our model is trained on 30s and 16kHz
    _sr = 16000
    _dur = 30
    speech, rate = librosa.load(
        audio_path, sr=_sr
    )  # speech has shape (len,); resample to 16k Hz

    # Detect language using the first 30s of speech
    lang_code = lang2code[src_lang]
    if lang_code == "none":
        lang_code = speech2lang(librosa.util.fix_length(speech, size=(_sr * _dur)))[0][
            0
        ].strip()[1:-1]
    speech2text.category_id = speech2text.converter.token2id[f"<{lang_code}>"]

    # ASR or ST
    if long_form:  # speech will be padded in decode_long()
        try:
            speech2text.maxlenratio = -300
            utts = speech2text.decode_long(
                speech,
                segment_sec=_dur,
                fs=_sr,
                condition_on_prev_text=False,
                init_text=text_prev,
                start_time="<0.00>",
                end_time_threshold="<29.50>",
            )

            text = []
            for t1, t2, res in utts:
                text.append(
                    f"[{format_timestamp(seconds=t1)} --> {format_timestamp(seconds=t2)}] {res}"
                )
            text = "\n".join(text)

            return code2lang[lang_code], text
        except:
            print(
                "An exception occurred in long-form decoding. Fall back to standard decoding (only first 30s)"
            )

    speech2text.maxlenratio = -min(
        300, int((len(speech) / rate) * 10)
    )  # assuming 10 tokens per second
    speech = librosa.util.fix_length(speech, size=(_sr * _dur))
    text = speech2text(speech, text_prev)[0][3]

    return code2lang[lang_code], text


demo = gr.Interface(
    predict,
    inputs=[
        gr.Audio(
            type="filepath",
            label="Input Speech (<120s)",
            max_length=120,
            sources=["microphone", "upload"],
            show_download_button=True,
            show_share_button=True,
        ),
        gr.Dropdown(
            choices=list(lang2code),
            value="English",
            label="Language",
            info="Language of input speech. Select 'Unknown' (1st option) to detect it automatically.",
        ),
        gr.Dropdown(
            choices=list(task2code),
            value="Automatic Speech Recognition",
            label="Task",
            info="Task to perform on input speech.",
        ),
        gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            value=5,
            label="Beam Size",
            info="Beam size used in beam search.",
        ),
        gr.Checkbox(
            label="Long Form (Experimental)",
            info="Perform long-form decoding for audios that are longer than 30s. If an exception happens, it will fall back to standard decoding on the initial 30s.",
        ),
        gr.Text(
            label="Text Prompt (Optional)",
            info="Generation will be conditioned on this prompt if provided",
        ),
    ],
    outputs=[
        gr.Text(
            label="Predicted Language",
            info="Language identification is performed if language is unknown.",
        ),
        gr.Text(label="Predicted Text", info="Best hypothesis."),
    ],
    title=TITLE,
    description=DESCRIPTION,
    allow_flagging="never",
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for running inference",
    )
    args = parser.parse_args()
