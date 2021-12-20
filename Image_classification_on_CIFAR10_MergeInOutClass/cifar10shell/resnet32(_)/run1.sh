#!/bin/bash

add_style=(0)
seed=(169 196 256 289 324)

for a in ${add_style[@]}; do
    for s in ${seed[@]}; do
        python train.py \
            -add_style ${a} \
            -seed ${s} \
            -GPU 2
    done
done

seed=(169 196 256 289 324 400 529 676 841 1024)
add_style=(1)
style_position=(3)
alpha1=(0.1 0.2)
alpha2=(0.1)

for a in ${add_style[@]}; do
    for pos in ${style_position[@]}; do
        for al1 in ${alpha1[@]}; do
            for al2 in ${alpha2[@]}; do
                for se in ${seed[@]}; do
                    python train.py \
                        -add_style ${a} \
                        -style_position ${pos} \
                        -styleratio1 ${al1} \
                        -styleratio2 ${al2} \
                        -seed ${se} \
                        -GPU 2
                done
            done
        done
    done
done
