#!/bin/bash
python test_widerface.py \
    --trained_model='outputs/resnet18_v1/Resnet18_iter_17000_3.0301_.pth' \
    --network='resnet18' \
    --save_folder='./widerface_evaluate/Resnet18_iter_17000_3.0301_/' \
    --confidence_threshold=0.7 \
    --nms_threshold=0.35

python test_widerface.py \
    --trained_model='outputs/resnet18_v1/Resnet18_iter_18000_2.8730_.pth' \
    --network='resnet18' \
    --save_folder='./widerface_evaluate/Resnet18_iter_18000_2.8730_/' \
    --confidence_threshold=0.7 \
    --nms_threshold=0.35

python test_widerface.py \
    --trained_model='outputs/resnet18_v1/Resnet18_iter_20000_2.7374_.pth' \
    --network='resnet18' \
    --save_folder='./widerface_evaluate/Resnet18_iter_20000_2.7374_/' \
    --confidence_threshold=0.7 \
    --nms_threshold=0.35

python test_widerface.py \
    --trained_model='outputs/resnet18_v1/Resnet18_iter_21000_2.6661_.pth' \
    --network='resnet18' \
    --save_folder='./widerface_evaluate/Resnet18_iter_21000_2.6661_/' \
    --confidence_threshold=0.7 \
    --nms_threshold=0.35
