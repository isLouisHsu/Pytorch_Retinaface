import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from data.data_augment import visualize
from itertools import chain

def load_datacube(datadir, bright_off=0):
    imgs = []
    for i in range(1, 26):
        filename = os.path.join(datadir, f'{i}.jpg')
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        imgs.append(img)
    imgs = np.stack(imgs)
    imgs = imgs + bright_off
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs

class EcustHsfdDetection(data.Dataset):
    def __init__(self, txt_path, 
            used_channels,
            preproc=None, transformers=None, 
            mode='train', valid_size=0.2,
            label_file='labels.txt', image_dir='Original_Image_jpg/',
    ):
        self.used_channels = [c - 1 for c in used_channels]
        self.preproc = preproc
        self.transformers = transformers
        imgs_path = []
        words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace(label_file, image_dir) + path
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        words.append(labels)
        f.close()

        # split training and validation data
        all_idx = [i for i in range(len(imgs_path))]
        train_idx, valid_idx = train_test_split(all_idx, test_size=0.2, shuffle=True)
        idx = train_idx if mode == 'train' else valid_idx
        self.imgs_path = [imgs_path[i] for i in idx]
        self.words = [words[i] for i in idx]

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        # img = cv2.imread(self.imgs_path[index])
        img = load_datacube(self.imgs_path[index])[..., self.used_channels]
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        annotations = np.array(annotations)

        if self.preproc:
            img, annotations = self.preproc(img, annotations)

        if self.transformers:
            bboxes = [label[:4].tolist() for label in annotations]
            category_ids = [0 for _ in range(len(bboxes))]
            keypoints = [label[4: 14].reshape(-1, 2).tolist() for label in annotations]
            keypoints_all = list(chain(*keypoints))
            idx_with_landm = [i for i, kp in enumerate(keypoints_all) if kp[0] >= 0]
            keypoints_all_dropped = np.array(keypoints_all)[idx_with_landm]
            transformed = self.transformers(image=img, 
                bboxes=bboxes, category_ids=category_ids, 
                keypoints=keypoints_all_dropped
            )

            img_t = transformed['image']
            bboxes_t = np.array(transformed['bboxes'])
            keypoints_t = np.array(transformed['keypoints'])
            # parse
            keypoints_t_new = [[-1, -1] for i in range(len(keypoints_all))]
            j = 0
            for i, idx in enumerate(idx_with_landm):
                # FIXME: 该扩增方法会丢失关键点
                keypoints_t_new[idx] = keypoints_t[i]
            keypoints_t = keypoints_t_new
            keypoints_t = np.array(keypoints_t).reshape(-1, 5, 2)
            # img = visualize(img, bboxes, keypoints)
            # img = visualize(img_t, bboxes_t, keypoints_t)
            # cv2.imshow('', img)
            # cv2.waitKey(0)
            
            height, width, _ = img_t.shape
            bboxes_t[:, 0::2] /= width
            bboxes_t[:, 1::2] /= height
            keypoints_t[..., 0] /= width
            keypoints_t[..., 1] /= height
            for i, (bbox, keypoint) in enumerate(zip(bboxes_t, keypoints_t)):
                annotations[i, :4] = bbox
                annotations[i, 4: 14] = keypoint.reshape(-1)
            img = np.transpose((img_t - 127.5) / 128.0, (2, 0, 1))

        # plt.imshow(img.astype(np.uint8).transpose(1, 2, 0))
        # plt.show()
        # cv2.imshow('', img.astype(np.uint8).transpose(1, 2, 0))
        # cv2.waitKey(0)

        return torch.from_numpy(img), annotations

def detection_collate_hsfd(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
