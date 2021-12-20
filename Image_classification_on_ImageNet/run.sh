# seed=(169 196 256)
seed=(169)
add_style=(0)

for a in ${add_style[@]}; do
    for s in ${seed[@]}; do
        python imagenet_DDP.py \
        --model resnet50 \
        --batch-size 128 \
        --lr 0.2 \
        --epochs 300 \
        --lambda_0 7.5 \
        --workers 32 \
        --dist-url 'tcp://127.0.0.1:12345' \
        --dist-backend 'nccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --seed ${s}
    done
done


# add_style=(1)
# style_position=(3)
# alpha1=(0.2)
# alpha2=(0.4)

# for a in ${add_style[@]}; do
#     for pos in ${style_position[@]}; do
#         for al1 in ${alpha1[@]}; do
#             for al2 in ${alpha2[@]}; do
#                 for se in ${seed[@]}; do
#                     python train.py \
#                         --model se_resnet \
#                         --layers 110 \
#                         -add_style ${a} \
#                         -style_position ${pos} \
#                         -styleratio1 ${al1} \
#                         -styleratio2 ${al2} \
#                         -seed ${se} \
#                         -GPU 2
#                 done
#             done
#         done
#     done
# done