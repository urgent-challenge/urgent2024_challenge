## Data simulation scripts

### Usage

If you only want to simulate the official training / validation data for the URGENT 2024 Challenge, just follow the [`prepare_espnet_data.sh`](https://github.com/urgent-challenge/urgent2024_challenge/tree/main/simulation) to run [generate_data_param.py](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/generate_data_param.py) and [simulate_data_from_param.py](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/simulate_data_from_param.py) respectively.


If you want to simulate paired degraded-clean speech data using your custom configuration, you can follow the steps below:

0. Generate data files for your speech, noise, and RIR (if used) corpora:
    ```bash
    # generate `.scp`, `.utt2spk`, and `.text` data files for each corpus
    mkdir -p data

    python -c '
    from pathlib import Path
    import soundfile as sf

    uids = set()
    def get_unique_uid(uid, i=0):
        uid2 = uid
        while uid2 in uids:
            i += 1
            uid2 = f"{uid}({i})"
        uids.add(uid2)
        return uid2

    #----- Define the args below -----
    ext = "wav"
    corpus_name = "xxx"
    corpus_dir = "/path/to/your/speech/or/noise/or/rir/corpus"
    #---------------------------------

    audios = Path(corpus_dir).rglob(f"*.{ext}")
    with open(f"data/{corpus_name}.scp", "w") as f1:
        with open(f"data/{corpus_name}.utt2spk", "w") as f2:
            with open(f"data/{corpus_name}.text", "w") as f3:
                for p in audios:
                    uid = get_unique_uid(p.stem)
                    f1.write(f"{uid} {sf.info(p).samplerate} {p}\n")
                    f2.write(f"{uid} {uid}\n")
                    f3.write(f"{uid} <not-available>\n")
    '
    ```

1. Prepare a simulation configuration file like [conf/simulation_train.yaml](https://github.com/urgent-challenge/urgent2024_challenge/blob/main/conf/simulation_train.yaml).

    <details><summary>Click to expand a template configuration</summary><div>

    ```yaml
    speech_scps:
    - /path1/to/speech_corpus1.scp
    - /path1/to/speech_corpus2.scp
    - ...

    speech_utt2spk:
    - /path1/to/speech_corpus1.utt2spk
    - /path1/to/speech_corpus2.utt2spk
    - ...

    speech_text:
    - /path1/to/speech_corpus1.text
    - /path1/to/speech_corpus2.text
    - ...

    log_dir: simulation_dir/log  # for storing meta.tsv
    output_dir: simulation_dir  # for storing generated audios
    repeat_per_utt: 1  # How many times to reuse each speech sample
    seed: 0  # random seed for reproducibility

    noise_scps:
    - /path/to/noise_corpus1.scp
    - /path/to/noise_corpus2.scp
    - ...
    snr_low_bound: -5.0   # lowest SNR in simulation
    snr_high_bound: 20.0  # highest SNR in simulation
    reuse_noise: true   # whether to allow using each noise sample for multiple times
    store_noise: false  # whether to store the generated noise audio files

    # If you don't need reverberation, simply use
    #
    # rir_scps: null
    rir_scps
    - /path/to/rir_corpus1.scp
    - /path/to/rir_corpus2.scp
    - ...
    prob_reverberation: 0.5  # apply RIRs with a probability
    reuse_rir: true  # whether to allow using each RIR sample for multiple times
    
    # If you only want to generate noisy speech data, simply use
    #
    # augmentations: [none]
    # weight_augmentations: [1.0]
    
    augmentations:
    - none  # the first augmentation (do nothing)
    - bandwidth_limitation  # the first augmentation (bandwidth limitation)
    - clipping  # the first augmentation (clipping)
    weight_augmentations:
    - 1.0  # weight for randomly selecting the first augmentation
    - 1.0  # weight for randomly selecting the second augmentation
    - 1.0  # weight for randomly selecting the third augmentation
    # The args below are only used for the clipping distortion
    clipping_min_quantile: [0.0, 0.1]
    clipping_max_quantile: [0.9, 1.0]
    ```
    </div></details>

2. Generate the corresponding simulation configuration file using the following command:
    > This step only reads the meta information of all audio files to prepare a reproduciable configuration for simulation.
    >
    > It does not generate any audio files. But a meta.tsv file will be generated in `$log_dir` (defined in the configuration file in step 1).

    ```bash
    # Replace "conf/simulation_train.yaml" with your custom configuration

    # meta.tsv will be generated in `$log_dir` ("simulation_train/log/" in this example)
    python simulation/generate_data_param.py --config conf/simulation_train.yaml
    ```
3. Simulate the paired degraded-clean speech data using the following command:
    > This step generates the paired degraded-clean speech data based on the simulation configuration file generated in step 2.
    >
    > The total number of simulated samples will be `$repeat_per_utt` × number of source speech samples.

    ```bash
    # Replace "conf/simulation_train.yaml" with your custom configuration
    # Replace "simulation_train/log/meta.tsv" with your custom meta.tsv file
    # Adjust the value of `--nj` to the number of CPU cores you want to use
    python simulation/simulate_data_from_param.py \
        --config conf/simulation_train.yaml \
        --meta_tsv simulation_train/log/meta.tsv \
        --nj 8 \
        --chunksize 100
    ```

After finishing the last step, the following directory structure can be found under `$output_dir` (defined in the configuration file in step 1):
```
📁 `$output_dir`/
│
├── 📁 log
│   └── meta.tsv
│
├── 📁 clean
│   │── fileid_1.flac
│   │── fileid_2.flac
│   └── ...
│
├── 📁 noisy
│   │── fileid_1.flac
│   │── fileid_2.flac
│   └── ...
│
└── 📁 noise (optional, only generated when `$store_noise` is true)
    │── fileid_1.flac
    │── fileid_2.flac
    └── ...
```

> To further prepare data files for model training and evaluation in the [ESPnet](https://github.com/espnet/espnet) toolkit, you can use the following script.
> <details><summary>Click to expand the script</summary><div>
>
> ```bash
> #----- Define the args below -----
> meta_tsv_file=simulation_train/log/meta.tsv
> subset_dir=data/train
> #---------------------------------
> mkdir -p "${subset_dir}"
>
> awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="noisy_path") {n=i; break}} next} NR>1{print($1" "$n)}' "${meta_tsv_file}" | sort -u > "${subset_dir}"/wav.scp 
> awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="speech_sid") {n=i; break}} next} NR>1{print($1" "$n)}' "${meta_tsv_file}" | sort -u > "${subset_dir}"/utt2spk
> utils/utt2spk_to_spk2utt.pl "${subset_dir}"/utt2spk > "${subset_dir}"/spk2utt
> awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="text") {n=i; break}} next} NR>1{print($1" "$n)}' "${meta_tsv_file}" | sort -u > "${subset_dir}"/text
> awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="clean_path") {n=i; break}} next} NR>1{print($1" "$n)}' "${meta_tsv_file}" | sort -u > "${subset_dir}"/spk1.scp 
> awk -F"\t" 'NR==1{for(i=1; i<=NF; i++) {if($i=="fs") {n=i; break}} next} NR>1{print($1" "$n)}' "${meta_tsv_file}" | sort -u > "${subset_dir}"/utt2fs
> awk '{print($1" 1ch_"$2"Hz")}' "${subset_dir}"/utt2fs > "${subset_dir}"/utt2category
> ```
>
> </summary><div>