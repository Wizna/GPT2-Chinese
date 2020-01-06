#!/usr/bin/env bash
python generate.py \
  --device 0 \
  --tokenizer_path cache/vocab_small.txt \
  --model_path model/model_epoch35 \
  --topp 0.9 \
  --topk 8 \
  --temperature 1.0 \
  --nsamples 1 \
  --prefix pre_context.txt \
  --repetition_penalty 1.1 \
  --original_lyric original_lyric.txt \
  --batch_size 1