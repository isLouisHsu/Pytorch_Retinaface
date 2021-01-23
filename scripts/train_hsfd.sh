python train_hsfd.py \
    --network=resnet34_hsfd \
    --version=with_finetune \
    --lr=1e-3 \
    --weight_decay=1e-3

python train_hsfd.py \
    --network=resnet34_hsfd_not_finetune \
    --version=not_finetune \
    --lr=1e-3 \
    --weight_decay=1e-3