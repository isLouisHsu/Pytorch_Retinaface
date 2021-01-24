#!/bin/bash

python test_ecust_hsfd.py \
    --network=resnet34_hsfd \
    --trained_model=outputs/resnet34_hsfd_with_finetune/Resnet34_iter_1200_0.2563_.pth

python test_ecust_hsfd.py \
    --network=resnet34_hsfd_not_finetune \
    --trained_model=outputs/resnet34_hsfd_not_finetune_not_finetune/Resnet34_iter_4900_0.2644_.pth
