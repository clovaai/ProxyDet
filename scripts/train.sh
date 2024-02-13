#/bin/bash
env_list=(
    "num_gpus"
    "seed"
    "config_file"
    "train_datasets"
    "dataset_bs"
    "num_workers"
    "eval_only"
    "resume"
    "out_dir"
    "eval_period"
    "eval_start"
    "pretrained_weight"
    "zs_weight_path"
    "cmm_base_alpha"
    "cmm_novel_beta"
)

default_train_datasets="None" # defaults to config
default_dataset_bs="None" # defaults to config
default_num_gpus="8"
default_seed="0"
default_config_file="./configs/ProxyDet_R50_Lbase_INL.yaml"
default_num_workers="2"
default_eval_only="False"
default_resume="False"
default_out_dir="None"
default_eval_period="0"
default_eval_start="0"
default_pretrained_weight="None"
default_zs_weight_path="datasets/metadata/lvis_v1_clip_a+cname.npy"
default_cmm_base_alpha="0.15"
default_cmm_novel_beta="0.35"

for env_name in ${env_list[@]};do
    if [[ -v "$env_name" ]]
    then
        eval $env_name='$'$env_name
    else
        eval $env_name='$'default_$env_name
    fi
done

config_file_underbar=$(echo ${config_file} | tr '/' '_')
date=$(date '+%Y_%m_%d_%H_%M_%S')

# set train datasets
if [ ${train_datasets} == "None" ]
then
    train_datasets_arg=""
else
    train_datasets_arg="DATASETS.TRAIN ${train_datasets}"
fi

# set datset bs
if [ ${dataset_bs} == "None" ]
then
    dataset_bs_arg=""
else
    dataset_bs_arg="DATALOADER.DATASET_BS ${dataset_bs}"
fi

# set default dir
if [ ${out_dir} == "None" ]
then
    out_dir="/mnt/tmp/${config_file_underbar}#_${date}"
fi

# set eval options
if [ ${eval_only} == "True" ]
then
    eval_only_arg="--eval-only"
else
    eval_only_arg=""
fi

# set pretrained checkpoint
if [ ${pretrained_weight} == "None" ]
then
    pretrained_weight_arg=""
else
    pretrained_weight_arg="MODEL.WEIGHTS ${pretrained_weight}"
fi

# set resume options
if [ ${resume} == "True" ]
then
    resume_arg="--resume"
else
    resume_arg=""
fi

python \
    train_net.py \
    --num-gpus ${num_gpus} \
    --config-file ${config_file} \
    ${train_datasets_arg} \
    ${dataset_bs_arg} \
    ${eval_only_arg} \
    ${resume_arg} \
    TEST.EVAL_PERIOD ${eval_period} \
    EVAL_START ${eval_start} \
    SEED ${seed} \
    MODEL.DEVICE "cuda" \
    ${pretrained_weight_arg} \
    DATALOADER.NUM_WORKERS ${num_workers} \
    MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH ${zs_weight_path} \
    MODEL.ROI_HEADS.CMM.BASE_ALPHA ${cmm_base_alpha} \
    MODEL.ROI_HEADS.CMM.NOVEL_BETA ${cmm_novel_beta} \
    OUTPUT_DIR ${out_dir} | tee "${config_file_underbar}#_${date}.log"
