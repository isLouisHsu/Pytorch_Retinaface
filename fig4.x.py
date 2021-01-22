import cv2
import numpy as np
import albumentations as A
from utils import _utils
from data import WiderFaceDetection, train_preproc, train_transformers
from data import data_augment as da

def transformers(img_size):
    return A.Compose([
            ## A.RandomResizedCrop(img_size, img_size),
            A.CenterCrop(img_size, img_size, p=1.0),
            A.HorizontalFlip(p=1.0),
            A.ShiftScaleRotate(p=1.0),
            A.ColorJitter(p=1.0),
            A.Cutout(p=1.0),
        ], 
        bbox_params=A.BboxParams('pascal_voc', label_fields=['category_ids']),
        keypoint_params=A.KeypointParams('xy')
    )


def process(image, annotations):
    _, height, width = image.shape
    boxes = annotations[:, : 4]
    landmarks = annotations[:, 4: 14]
    boxes[:, 0::2] *= width
    boxes[:, 1::2] *= height
    landmarks[:, 0::2] *= width
    landmarks[:, 1::2] *= height
    landmarks = landmarks.reshape(-1, 5, 2)
    image = image * 128.0 + 127.5
    image = image.numpy().transpose(1, 2, 0).astype(np.uint8)
    return image, boxes, landmarks

_utils.seed_everything(99)
training_dataset = '../data/widerface/WIDER_train/label.txt'
trainset = WiderFaceDetection(training_dataset, transformers=transformers(480), mode='train')

# for i in range(len(trainset)):
#     try:
#         image, annotations = trainset[i]
#         print(i)
#         image, boxes, landmarks = process(image, annotations)
#         image = da.visualize(image, boxes, landmarks)
#         cv2.imshow('', image)
#         cv2.imwrite('/home/louishsu/Desktop/%d.jpg' % i, image)
#         cv2.waitKey(0)
#     except Exception as e:
#         print(e)
#         continue


image, annotations = trainset[2]
image, boxes, landmarks = process(image, annotations)
image = da.visualize(image, boxes, landmarks)
cv2.imshow('', image)
cv2.imwrite('/home/louishsu/Desktop/1.jpg', image)
cv2.waitKey(0)

for i in range(10):
    try:
        image, annotations = trainset[2]
        image, boxes, landmarks = process(image, annotations)
        image = da.visualize(image, boxes, landmarks)
        cv2.imshow('', image)
        cv2.imwrite('/home/louishsu/Desktop/%d.jpg' % i, image)
        cv2.waitKey(0)
    except:
        continue
print()