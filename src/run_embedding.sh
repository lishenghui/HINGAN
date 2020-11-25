#!/usr/bin/env bash

#python ./recgan.py --out_path="../../results/recgan_default.txt"
for i in 2 4 8 16 32 48 64 96 128 256 512
  do
    python3 ./hingan.py --emb_dim=$i --out_path="../results/embedding_size_${i}.txt"
done