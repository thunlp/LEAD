log_dir=train/log
CUDA_LAUNCH_BLOCKING=1


data_name="your data name here"
# data_name="checkpoint_2048_xs30original_fp16_train_globalmask_1e-5"
output_dir=train/${data_name}
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

# change it to your path
pretrain_model="your path here"
# pretrain_model="/yeesuanAI06/thunlp/gaocheng/LEAD/Lawformer_model"
train_data="your path here"
# train_data="/yeesuanAI06/thunlp/gaocheng/LEAD/LEAD_data/xs_100060_30_train.json"
dev_data="your path here"
# dev_data="/yeesuanAI06/thunlp/gaocheng/LEAD/LEAD_data/xs_100060_30_dev.json"

# NOTE: During training, pay attention to whether the false negative masking strategy is needed. If your dataset does not label charges or you do not need to use this strategy, please uncomment out line 288 in LEAD/dpr/models/biencoder.py
torchrun --nproc_per_node 8 \
    train_dense_encoder.py \
    fp16=True \
    seed=42 \
    val_av_rank_start_epoch=40 \
    train.warmup_steps=306 \
    train.learning_rate=1e-5 \
    train.num_train_epochs=40 \
    train.batch_size=16 \
    train.dev_batch_size=64 \
    train.gradient_accumulation_steps=1 \
    encoder.sequence_length=512 \
    encoder.pretrained_model_cfg=${pretrain_model} \
    checkpoint_file_name=dpr_biencoder \
    train_datasets=[Law_data_train] \
    datasets.Law_data_train.file=${train_data} \
    dev_datasets=[Law_data_dev] \
    datasets.Law_data_dev.file=${dev_data} \
    train=biencoder_LCR \
    output_dir=${output_dir} \
    | tee ${log_dir}/${data_name}.log
