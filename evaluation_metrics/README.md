## Evaluation metrics

This folder contains the objective evaluation metrics used in the URGENT Challenge. The metrics are used to evaluate the performance of the models on the evaluation datasets. The metrics include:

<table class="tg">
<thead>
<tr>
    <th class="tg-uzvj">Category</th>
    <th class="tg-g7sd">Metric</th>
    <th class="tg-uzvj">Need Reference Signals?</th>
    <th class="tg-uzvj">Supported Sampling Frequencies</th>
    <th class="tg-uzvj">Value Range</th>
    <th class="tg-uzvj">Run on CPU or GPU?</th>
</tr>
</thead>
<tbody>
<tr>
    <td class="tg-r6l2" rowspan="2">Non-intrusive SE metrics</td>
    <td class="tg-rt8k"><a href="calculate_nonintrusive_dnsmos.py">DNSMOS</a> ↑</td>
    <td class="tg-51oy">❌</td>
    <td class="tg-51oy">16 kHz</td>
    <td class="tg-51oy">[1, 5]</td>
    <td class="tg-51oy">CPU</td>
</tr>
<tr>
    <td class="tg-0a7q"><a href="calculate_nonintrusive_nisqa.py">NISQA</a> ↑</td>
    <td class="tg-xwyw"><span style="font-weight:400;font-style:normal;text-decoration:none">❌</span></td>
    <td class="tg-xwyw">48 kHz</td>
    <td class="tg-xwyw">[1, 5]</td>
    <td class="tg-xwyw">CPU</td>
</tr>
<tr>
    <td class="tg-kyy7" rowspan="6">Intrusive SE metrics</td>
    <!-- <td class="tg-d459"><a href="http://www.polqa.info" style="color:#e97c36;">POLQA</a> ↑</td> -->
    <!-- <td class="tg-kyy7">✔</td>
    <td class="tg-kyy7"><span style="font-weight:400;font-style:normal;text-decoration:none">8~48 kHz</span></td>
    <td class="tg-kyy7"><span style="font-weight:400;font-style:normal;text-decoration:none">[1, 5]</span></td> -->
</tr>
<tr>
    <td class="tg-d459"><a href="calculate_intrusive_se_metrics.py">PESQ</a> ↑</td>
    <td class="tg-kyy7">✔</td>
    <td class="tg-kyy7"><span style="font-weight:400;font-style:normal;text-decoration:none">{8, 16} kHz</span></td>
    <td class="tg-kyy7"><span style="font-weight:400;font-style:normal;text-decoration:none">[-0.5, 4.5]</span></td>
    <td class="tg-kyy7">CPU</td>
</tr>
<tr>
    <td class="tg-r2ra"><a href="calculate_intrusive_se_metrics.py">ESTOI</a> ↑</td>
    <td class="tg-ligs">✔</td>
    <td class="tg-ligs"><span style="font-weight:400;font-style:normal;text-decoration:none">10 kHz</span></td>
    <td class="tg-ligs">[0, 1]</td>
    <td class="tg-ligs">CPU</td>
</tr>
<tr>
    <td class="tg-d459"><a href="calculate_intrusive_se_metrics.py">SDR</a> ↑</td>
    <td class="tg-kyy7">✔</td>
    <td class="tg-kyy7">Any</td>
    <td class="tg-kyy7">(-∞, +∞)</td>
    <td class="tg-kyy7">CPU</td>
</tr>
<tr>
    <td class="tg-r2ra"><a href="calculate_intrusive_se_metrics.py">MCD</a> ↓</td>
    <td class="tg-ligs">✔</td>
    <td class="tg-ligs">Any</td>
    <td class="tg-ligs">[0, +∞)</td>
    <td class="tg-ligs">CPU</td>
</tr>
<tr>
    <td class="tg-d459"><a href="calculate_intrusive_se_metrics.py">LSD</a> ↓</td>
    <td class="tg-kyy7">✔</td>
    <td class="tg-kyy7">Any</td>
    <td class="tg-kyy7">[0, +∞)</td>
    <td class="tg-kyy7">CPU</td>
</tr>
<tr>
    <td class="tg-rq3n" rowspan="2">Downstream-task-independent metrics</td>
    <td nowrap class="tg-mfxt"><a href="calculate_speechbert_score.py">SpeechBERTScore</a> ↑</td>
    <td class="tg-rq3n">✔</td>
    <td class="tg-rq3n">16 kHz</td>
    <td class="tg-rq3n">[-1, 1]</td>
    <td class="tg-rq3n">CPU or GPU</td>
</tr>
<tr>
    <td class="tg-qmuc"><a href="calculate_phoneme_similarity.py">LPS</a> ↑</td>
    <td class="tg-r6l2">✔</td>
    <td class="tg-r6l2">16 kHz</td>
    <td class="tg-r6l2"><span style="font-weight:400;font-style:normal;text-decoration:none">(-∞, 1]</span></td>
    <td class="tg-r6l2">CPU or GPU</td>
</tr>
<tr>
    <td class="tg-ligs" rowspan="2">Downstream-task-dependent metrics</td>
    <td class="tg-r2ra"><a href="calculate_speaker_similarity.py">SpkSim</a> ↑</td>
    <td class="tg-ligs">✔</td>
    <td class="tg-ligs">16 kHz</td>
    <td class="tg-ligs">[-1, 1]</td>
    <td class="tg-ligs">CPU or GPU</td>
</tr>
<tr>
    <td class="tg-d459"><a href="calculate_wer.py">WAcc</a> (=1-WER) ↑</td>
    <td class="tg-kyy7">❌</td>
    <td class="tg-kyy7">16 kHz</td>
    <td class="tg-kyy7">(-∞, 1]</td>
    <td class="tg-kyy7">CPU or GPU</td>
</tr>
<!-- <tr>
    <td class="tg-r6l2" rowspan="1">Subjective SE metrics</td>
    <td class="tg-rt8k"><a href="https://github.com/microsoft/P.808" style="color:#e97c36;">MOS</a> ↑</td>
    <td class="tg-51oy">❌</td>
    <td class="tg-51oy">Any</td>
    <td class="tg-51oy">[1, 5]</td>
</tr> -->
</tbody>
</table>

## Usage

1. Make sure that you have prepared the enhanced speech and their corresponding clean reference signals in the same folder, and that each paired enhanced-reference samples have the same sampling frequencies and file names. For example, the enhanced speech samples are stored in `enhanced/` folder and the clean reference samples are in `clean/` folder. The folder structure will look like this:
    ```
    📁 /path/to/your/data/
    ├── 📁 enhanced/
    │   ├── 🔈 fileid_1.flac
    │   ├── 🔈 fileid_2.flac
    │   └── ...
    └── 📁 clean/
        ├── 🔈 fileid_1.flac
        ├── 🔈 fileid_2.flac
        └── ...
    ```

2. Prepare the scp files for both enhanced and clean reference signals. The scp files should contain the paths to the audio files and their corresponding sampling frequencies. The scp files should look like this:
    ```
    # enhanced.scp
    fileid_1 /path/to/your/data/enhanced/fileid_1.flac
    fileid_2 /path/to/your/data/enhanced/fileid_2.flac
    ...
    
    # reference.scp
    fileid_1 /path/to/your/data/clean/fileid_1.flac
    fileid_2 /path/to/your/data/clean/fileid_2.flac
    ...
    ```

    You can prepare the scp files using the following command:
    ```bash
    find /path/to/your/data/enhanced/ -name "*.flac" | \
        awk -F'/' '{print $NF" "$0}' | sed 's/.flac / /' | \
        LC_ALL=C sort -u > enhanced.scp
    
    find /path/to/your/data/clean/ -name "*.flac" | \
        awk -F'/' '{print $NF" "$0}' | sed 's/.flac / /' | \
        LC_ALL=C sort -u > reference.scp
    ```

    For WER evaluation, you will need to additional provide the reference text file. The text file should contain the transcriptions of the clean reference signals. The text file should look like this:
    ```
    # reference.text
    fileid_1 This is the transcription of the first speech sample
    fileid_2 This is the transcription of the second speech sample
    ...
    ```

    Alternatively, you can also use the `meta.tsv` file provided in the [evaluation dataset](https://urgent-challenge.github.io/urgent2024/data/). The `meta.tsv` file should at least contain the two columns (`id` and `text`) for the WER evaluation. The `meta.tsv` file should look like this:
    
    | id | ... | text |
    |:---:|:---:|:---:|
    |fileid_1|...|Please call Stella.|
    |fileid_2|...|Please call Stella.|
    ...
    |fileid_1000|...|&lt;not-available&gt;|

    > [!NOTE]  
    > The `id` column should match the IDs defined in the first column of the scp files. The `text` column should contain the transcriptions of the clean reference signals. The `...` columns can be any other columns (if available) in the `meta.tsv` file.
    >
    > Samples with the `text` column filled with `<not-available>` will be ignored during the WER evaluation.

3. Run the evaluation script with the following command:

    **CPU-only metrics:**

    ```bash
    #!/bin/bash
    nj=16  # Number of parallel CPU jobs for speedup
    python=python3

    output_prefix=scoring_results

    # PESQ, ESTOI, SDR, MCD, LSD
    ${python} calculate_intrusive_se_metrics.py \
        --ref_scp reference.scp \
        --inf_scp enhanced.scp \
        --output_dir "${output_prefix}"/scoring_cpu \
        --nj ${nj} \
        --chunksize 60
    ```

    **GPU-supported metrics:**

    <details><summary>DNSMOS</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    ref_scp=reference.scp
    inf_scp=enhanced.scp

    mkdir -p DNSMOS/
    wget -c -O DNSMOS/sig_bak_ovr.onnx https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
    wget -c -O DNSMOS/model_v8.onnx https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/model_v8.onnx

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_nonintrusive_dnsmos.py \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_dnsmos \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx} \
            --convert_to_torch ${gpu_inference} \
            --primary_model ./DNSMOS/sig_bak_ovr.onnx \
            --p808_model ./DNSMOS/model_v8.onnx

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_dnsmos/DNSMOS_OVRL.${i}.scp
        done > "${output_prefix}"/scoring_dnsmos/DNSMOS_OVRL.scp
    fi
    ```

    </div></details>

    <details><summary>NISQA</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    ref_scp=reference.scp
    inf_scp=enhanced.scp

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_nonintrusive_nisqa.py \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_nisqa \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx} \
            --nisqa_model ../lib/NISQA/weights/nisqa.tar

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_nisqa/NISQA_MOS.${i}.scp
        done > "${output_prefix}"/scoring_nisqa/NISQA_MOS.scp
    fi
    ```

    </div></details>

    <details><summary>SpeechBERTScore</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    ref_scp=reference.scp
    inf_scp=enhanced.scp

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_speechbert_score.py \
            --ref_scp "${ref_scp}" \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_speech_bert_score \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_speech_bert_score/SpeechBERTScore.${i}.scp
        done > "${output_prefix}"/scoring_speech_bert_score/SpeechBERTScore.scp
    fi
    ```

    </div></details>

    <details><summary>SpeakerSimilarity</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    ref_scp=reference.scp
    inf_scp=enhanced.scp

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_speaker_similarity.py \
            --ref_scp "${ref_scp}" \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_speaker_similarity \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_speaker_similarity/SpeakerSimilarity.${i}.scp
        done > "${output_prefix}"/scoring_speaker_similarity/SpeakerSimilarity.scp
    fi
    ```

    </div></details>
    
    <details><summary>PhonemeSimilarity</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    ref_scp=reference.scp
    inf_scp=enhanced.scp

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_phoneme_similarity.py \
            --ref_scp "${ref_scp}" \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_phoneme_similarity \
            --device ${_device} \
            --nsplits ${nj} \
            --job ${idx}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_phoneme_similarity/PhonemeSimilarity.${i}.scp
        done > "${output_prefix}"/scoring_phoneme_similarity/PhonemeSimilarity.scp
    fi
    ```

    </div></details>

    <details><summary>WER</summary><div>

    ```bash
    #!/bin/bash
    nj=8  # Number of parallel CPU/GPU jobs for speedup
    python=python3

    # Whether to use GPU for inference
    gpu_inference=true
    if ${gpu_inference}; then
        _device="cuda"
    else
        _device="cpu"
    fi

    ref_text=reference.text
    # Alternatively, you can also use the "meta.tsv" file for `ref_text`
    # ref_text=/path/to/meta.tsv
    inf_scp=enhanced.scp

    pids=() # initialize pids
    for idx in $(seq ${nj}); do
    (

        # Run each parallel job on a different GPU (if $gpu_inference = true)
        CUDA_VISIBLE_DEVICES=$((${idx} - 1)) ${python} calculate_wer.py \
            --meta_tsv "${ref_text}" \
            --inf_scp "${inf_scp}" \
            --output_dir "${output_prefix}"/scoring_wer \
            --device ${_device} \
            --nsplits ${nj} \
            --job JOB

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
    echo "Finished"

    if [ ${nj} -gt 1 ]; then
        for i in $(seq ${nj}); do
            cat "${output_prefix}"/scoring_wer/WER.${i}.scp
        done > "${output_prefix}"/scoring_wer/WER.scp
    fi
    ```

    </div></details>