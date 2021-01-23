import os
import cv2
import sys
sys.path.append('../band_selection/')
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from data import cfg_re34, visualize
from detector import RetinaFaceDetector
from dataset import EcustHSFDCroppedDataset as EcustHSFDDataset

CFG = {
    'resnet34': cfg_re34,
}

def main(args):
    dataset = EcustHSFDDataset(args.datadir, 'train') + \
        EcustHSFDDataset(args.datadir, 'valid') + \
        EcustHSFDDataset(args.datadir, 'test')
    dataset.return_sample = True

    fp1 = open(args.annotation_file, 'w')
    fp2 = open(os.path.join(os.path.dirname(args.annotation_file), 'not_detected.txt'), 'w')
    
    detector = RetinaFaceDetector(CFG[args.network], weights_path=args.weights_path)
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        image = sample.load_image()
        
        is_detected = False
        for j in range(image.shape[-1]):
            image_ = np.repeat(image[..., [j]], 3, axis=-1)
            scores, dets, landms = detector.detect(image_, keep_top_k=1)
            if scores.size == 0: continue

            scores = scores[0]
            dets = dets[0].reshape(-1).astype(np.int).tolist()
            dets[2] -= dets[0]; dets[3] -= dets[1]
            landms = np.concatenate([landms[0], np.zeros((5, 1))], axis=-1).reshape(-1).tolist()
            label = [0. for _ in range(20)]
            label[-1] = scores
            label[:4] = dets
            label[4: -1] = landms
            
            label = ' '.join(list(map(str, label)))
            image_path = sample.image_path.lstrip(args.datadir)
            fp1.write(f'# {image_path:s}\n')
            fp1.write(f'{label}\n')

            # image_ = visualize(image_, dets, landms, scores)
            # cv2.imshow('', image_)
            # cv2.waitKey(0)

            is_detected = True
            break
        if not is_detected:
            fp2.write(f'{sample.image_path}\n')
    
    fp1.close()
    fp2.close()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, 
        default='../data/ecust_hsfd/Original_Image_jpg_indoor/Original_Image_jpg/')
    parser.add_argument('--network', type=str, 
        default='resnet34')
    parser.add_argument('--weights_path', type=str, 
        default='outputs/resnet34_v1/Resnet34_iter_21000_2.5562_.pth')
    parser.add_argument('--annotation_file', type=str, 
        default='../data/ecust_hsfd/Original_Image_jpg_indoor/labels.txt')
    args = parser.parse_args()
    main(args)