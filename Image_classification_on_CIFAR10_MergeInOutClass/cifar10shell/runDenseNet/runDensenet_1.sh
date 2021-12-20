#!/bin/bash

seed=(169 196 256 289 324 400 529 676 841 1024)

add_style=(1)
style_position=(3)
alpha1=(0.2)
alpha2=(0.4)

for a in ${add_style[@]}; do
    for pos in ${style_position[@]}; do
        for al1 in ${alpha1[@]}; do
            for al2 in ${alpha2[@]}; do
                for se in ${seed[@]}; do
                    python train.py \
                        --model densenet_bc \
                        --layer 100 \
                        -add_style ${a} \
                        -style_position ${pos} \
                        -styleratio1 ${al1} \
                        -styleratio2 ${al2} \
                        -seed ${se} \
                        # -GPU 2
                done
            done
        done
    done
done
