#!/bin/bash

python test_widerface.py \
    --trained_model='outputs/Efficientnet-b4_v1/Efficientnet-b4_iter_82000_2.8391_.pth' \
    --network='Efficientnet-b4' \
    --save_folder='./widerface_evaluate/predictions/' \
    --confidence_threshold=0.7 \
    --nms_threshold=0.35 \
    --save_image

python test_widerface.py \
    --trained_model='outputs/Efficientnet-b0_v1/Efficientnet-b0_iter_85000_2.9441_.pth' \
    --network='Efficientnet-b0' \
    --save_folder='./widerface_evaluate/predictions/' \
    --confidence_threshold=0.7 \
    --nms_threshold=0.35 \
    --save_image

# python test_widerface.py \
#     --trained_model='outputs/resnet18_v1/Resnet18_iter_21000_2.6661_.pth' \
#     --network='resnet18' \
#     --save_folder='./widerface_evaluate/predictions/' \
#     --confidence_threshold=0.7 \
#     --nms_threshold=0.35 \
#     --save_image

python test_widerface.py \
    --trained_model='outputs/resnet18_v1/Resnet18_iter_18000_2.8730_.pth' \
    --network='resnet18' \
    --save_folder='./widerface_evaluate/predictions/' \
    --confidence_threshold=0.7 \
    --nms_threshold=0.35 \
    --save_image

# python test_widerface.py \
#     --trained_model='outputs/resnet34_v1/Resnet34_iter_21000_2.5562_.pth' \
#     --network='resnet34' \
#     --save_folder='./widerface_evaluate/predictions/' \
#     --confidence_threshold=0.7 \
#     --nms_threshold=0.35 \
#     --save_image

python test_widerface.py \
    --trained_model='outputs/resnet34_v1/Resnet34_iter_18000_2.8420_.pth' \
    --network='resnet34' \
    --save_folder='./widerface_evaluate/predictions/' \
    --confidence_threshold=0.7 \
    --nms_threshold=0.35 \
    --save_image
