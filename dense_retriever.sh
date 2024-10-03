#!/bin/bash

test_dataset="lecard"
# test_dataset="CAIL2022"
checkpoint_path="your path here"
# checkpoint_path="/yeesuanAI06/thunlp/gaocheng/LEAD/train/checkpoint_2048_Lawformer_train_0728_cail2022/dpr_biencoder.39"
pretrained_model_cfg="your path here"
# pretrained_model_cfg="/yeesuanAI06/thunlp/gaocheng/LEAD/Lawformer_model"
encoded_ctx_dir="your path here"
# encoded_ctx_dir="/yeesuanAI06/thunlp/gaocheng/LEAD/encoded/encoded_cail2022_train_39epochs/LeCaRD"
LeCaRD_root_folder="your path here"
# LeCaRD_root_folder="/yeesuanAI06/thunlp/gaocheng/LEAD/LeCaRD/data/candidates/similar_case"
CAIL_root_folder="your path here"
# CAIL_root_folder="/yeesuanAI06/thunlp/gaocheng/CAIL2022/stage2"

out_file="your path here"
# out_file="/yeesuanAI06/thunlp/gaocheng/LEAD/result/result_LeCard/results_Lawformer_cail2022_train_39epochs_test"
if [ ! -d $out_file ]; then
	mkdir $out_file
fi


if [ $test_dataset = "lecard" ]; then
    base_dir=${LeCaRD_root_folder}
    encoded_ctx_dir=${encoded_ctx_dir}
    dir_list="candidates1 candidates2"
	# for simplified LeCaRD
	query_key="q_short"
	# for LeCaRD
	# query_key="q"
	query_path="/yeesuanAI06/thunlp/gaocheng/LEAD/LeCaRD/data/query/query_simplify.json"
elif [ $test_dataset = "CAIL2022" ]; then
    base_dir=${CAIL_root_folder}
	encoded_ctx_dir=${encoded_ctx_dir}
    dir_list="candidates_stage2_valid"
	query_key="q_short"
	# query_key="q"
	query_path="/yeesuanAI06/thunlp/gaocheng/LEAD/CAIL2022/stage2/query_stage2_valid_onlystage2_40_simplified.json"
else
    echo "Invalid test dataset"
    exit 1
fi


# -m torch.distributed.launch --nproc_per_node 8 --use_env
base_command="python -m torch.distributed.launch --nproc_per_node 8 --use_env dense_retriever.py \
    model_file=${checkpoint_path}
	is_DPR_checkpoint=True \
	from_pretrained=True \
	qa_dataset=lecard_short \
	datasets.lecard_short.file=${query_path} \
	datasets.lecard_short.question_attr=${query_key} \
    ctx_datatsets=[lecard_short] \
	encoder.sequence_length=512 \
    encoder.pretrained_model_cfg=${pretrained_model_cfg} \
	out_file=${out_file}/"

# 遍历所有子目录
$COUNT=0
for folder_name in $dir_list; do
    folder_path="${base_dir}/${folder_name}"
	for sub_dir in $(ls "$folder_path"); do
		COUNT=$((COUNT + 1))
        echo "Subfolder name: $sub_dir"
        echo "COUNT: $COUNT"
		if [ -f "${out_file}/${sub_dir}.json" ]; then
            echo "File exists, skip"
            continue
        fi
		command="$base_command$sub_dir.json \
			ctx_sources.lecard_short.query_ridx_path=${folder_path}/${sub_dir} \
			datasets.lecard_short.ridx=$sub_dir \
			encoded_ctx_files=[$encoded_ctx_dir/$folder_name/${sub_dir}_0]"
		$command
	done
done
