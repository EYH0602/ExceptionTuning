#!/bin/bash

model='codebert-base'
ifraw='False'
error_prompt='error is in line:'
n_class=22

CUDA_VISIBLE_DEVICES=3,4 python3 run.py \
                --output_dir=./savedmodels \
                --model_type=roberta_cls \
                --config_name=../microsoft/$model \
                --model_name_or_path=../microsoft/$model \
                --tokenizer_name=../microsoft/$model \
                --do_train \
                --do_test \
                --train_data_file=../dataset/train_err_cls.jsonl \
                --eval_data_file=../dataset/valid_err_cls.jsonl \
                --test_data_file=../dataset/test_err_cls.jsonl \
                --epoch 1 \
                --block_size 512 \
                --train_batch_size 32 \
                --eval_batch_size 64 \
                --learning_rate 2e-5 \
                --max_grad_norm 1.0 \
                --evaluate_during_training \
                --error_prompt "$error_prompt" \
                --need_raw $ifraw \
                --nohang \
                --nocrash \
                --n_class $n_class \
                --seed 123456 2>&1| tee ./exception_train.log.txt &
