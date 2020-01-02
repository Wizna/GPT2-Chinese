#!/usr/bin/env bash
python generate.py \
  --device 0 \
  --length 900 \
  --tokenizer_path cache/vocab_user.txt \
  --model_path 'C:\\Users\\Ruiming Huang\\Desktop\\chkp8' \
  --prefix "[CLS][MASK]" \
  --topp 1.0 \
  --temperature 1.0 \
  --nsamples 3 \


