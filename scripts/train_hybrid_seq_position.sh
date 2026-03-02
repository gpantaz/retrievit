#!/bin/bash
model_config=$1
learning_rate=$2
per_device_train_batch_size=$3
per_device_eval_batch_size=$4
gradient_accumulation_steps=${5}
is_prefix=${6}
seed=${7}

config=$(basename "$model_config")

python train.py \
	--model-class "hybrid_seq" \
	--model-config ${model_config} \
	--task position_retrieval \
	--is_prefix ${is_prefix} \
	--train-dataset-size 20000000 \
	--validation-dataset-size 100000 \
	--seq-len 200 \
	--output-dir storage/models/position_retrieval-${config}-lr${learning_rate}-seqlen200-prefix${is_prefix}-seed${seed}-og \
	--per-device-train-batch-size ${per_device_train_batch_size} \
	--per-device-eval-batch-size ${per_device_eval_batch_size} \
	--gradient-accumulation-steps ${gradient_accumulation_steps} \
	--num-train-epochs 1 \
	--save-safetensors False \
	--load-best-model False \
	--save-strategy no \
	--learning-rate ${learning_rate} \
	--seed ${seed} \
	--data-seed ${seed} \
	--dataloader-num-worker 4 \
	--upload_embeddings_after_training True \
	--upload_embeddings_during_training True \
	--upload_full_model_after_training True \
	--hf_repo_id gpantaz/retrievit \
	--project-name retrievit_position_retrieval \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.1 \
	--run-name ${config}-lr${learning_rate}-seqlen200-prefix${is_prefix}-seed${seed}-og