#!/bin/bash

add_style=(0)
seed=(169 196 256 289 324 400 529 676 841 1024)

for a in ${add_style[@]}; do
    for s in ${seed[@]}; do
        python train.py \
            --layers 32 \
            -add_style ${a} \
            -seed ${s} \
            -GPU 1
    done
done




