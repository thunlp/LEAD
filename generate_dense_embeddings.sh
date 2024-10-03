#!/bin/bash

checkpoint_path="your path here"
# checkpoint_path="/yeesuanAI06/thunlp/gaocheng/LEAD/train/checkpoint_2048_Lawformer_train_0728_cail2022/dpr_biencoder.39"
pretrained_model_cfg="your path here"
# pretrained_model_cfg="/yeesuanAI06/thunlp/gaocheng/LEAD/Lawformer_model"
test_dataset="lecard"
# test_dataset="CAIL2022"
LeCaRD_root_folder="your path here"
# LeCaRD_root_folder="/yeesuanAI06/thunlp/gaocheng/LEAD/LeCaRD/data/candidates/similar_case"
CAIL_root_folder="your path here"
# CAIL_root_folder="/yeesuanAI06/thunlp/gaocheng/CAIL2022/stage2"
output_folder="your path here"
# output_folder="/yeesuanAI06/thunlp/gaocheng/LEAD/encoded_for_LeCaRD/encoded_cail2022_train_39epochs_test/LeCaRD"

base_command="python -m torch.distributed.launch --nproc_per_node 8 --use_env generate_dense_embeddings.py \
    model_file=${checkpoint_path} \
    ctx_src=lecard_short \
    encoder.sequence_length=512 \
    encoder.pretrained_model_cfg=${pretrained_model_cfg} \
    is_DPR_checkpoint=True \
    from_pretrained=True \
    shard_id=0 num_shards=1 "

if [ $test_dataset = "lecard" ]; then
    root_folder=${LeCaRD_root_folder}
    output_folder=${output_folder}
    dir_list="candidates1 candidates2"
elif [ $test_dataset = "CAIL2022" ]; then
    root_folder=${CAIL_root_folder}
    output_folder=${output_folder}
    dir_list="candidates_stage2_valid"
else
    echo "Invalid test dataset"
    exit 1
fi

if [ ! -d "$output_folder" ]; then
    mkdir -p "$output_folder"
fi

for folder_name in $dir_list; do
    folder_path="${root_folder}/${folder_name}"
    output_path="${output_folder}/${folder_name}"

    if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
    fi
    $COUNT=0
    for subfolder in $(ls "$folder_path"); do
        COUNT=$((COUNT + 1))
        echo "Subfolder name: $subfolder"
        echo "COUNT: $COUNT"
        subfolder_path="${folder_path}/${subfolder}"
        subfolder_output="${output_path}/${subfolder}"
        echo $subfolder_output
        if [ -f "${subfolder_output}_0" ]; then
            echo "File exists, skip"
            continue
        fi
        command="${base_command} \
            ctx_sources.lecard_short.query_ridx_path=${subfolder_path} \
            out_file=${subfolder_output}"
        
        $command
    done
done