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
	--model-class "mamba" \
	--model-config ${model_config} \
	--task n_gram_retrieval \
	--is_prefix ${is_prefix} \
	--train-dataset-size 30000000 \
	--validation-dataset-size 10000 \
	--vocab-size 30 \
	--min-seq-len 50 \
	--seq-len 100 \
	--retrieval_n_gram_size 5 \
	--retrieval_query_n_gram_size 2 \
	--output-dir storage/models/ngram-${config}-lr${learning_rate}-ngram5-query2-seqlen100-minseqlen50-prefix${is_prefix}-seed${seed} \
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
	--project-name retrievit_ngram_retrieval_test_duplicate \
	--upload_embeddings_after_training False \
	--upload_embeddings_during_training False \
	--upload_full_model_after_training False \
	--early_stopping_threshold 0.99 \
	--do_test_duplicate True \
	--max_train_steps 10000000 \
	--run-name ${config}-lr${learning_rate}-ngram5-query2-seqlen100-minseqlen50-prefix${is_prefix}-seed${seed}
