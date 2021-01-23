import cv2
import torch
import numpy as np
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from utils.timer import Timer


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class RetinaFaceDetector(RetinaFace):

    def __init__(self, cfg, weights_path=None, device='cuda:0'):
        super().__init__(cfg, phase='test')
        self.device = torch.device(device)
        self = self.to(self.device).eval()
        if weights_path:
            self = load_model(self, weights_path, True)
        
    def _process_image(self, img, origin_size=True, target_size=480, max_size=2150):
        im_shape = img.shape        # (H, W, C)
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        img = (img - 127.5) / 128.0
        # img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img, resize          # (1, C, H, W)
    
    @torch.no_grad()
    def detect(self, img, 
            origin_size=True,
            target_size=480, 
            max_size=2150,
            confidence_threshold=0.7, 
            nms_threshold=0.35, 
            top_k=5000, 
            keep_top_k=750):

        img, resize = self._process_image(
            img, origin_size, target_size, max_size)
        img = img.to(self.device)
        _, _, im_height, im_width = img.size()

        # anchor
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)
        prior_data = priors.data

        # forward
        loc, conf, landms = self(img)
        
        # decoder output
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        scale = torch.Tensor([
            im_width, im_height, 
            im_width, im_height,
        ]).to(self.device)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([
            im_width, im_height, 
            im_width, im_height, 
            im_width, im_height, 
            im_width, im_height, 
            im_width, im_height, 
        ]).to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:keep_top_k, :]
        # landms = landms[:keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)
        scores = dets[:, -1]                    # (N,)
        dets = dets[:, :-1]                     # (N, 4)
        landms = landms.reshape(-1, 5, 2)       # (N, 5, 2)
        return scores, dets, landms


if __name__ == '__main__':

    from torchstat import stat
    from data.data_augment import visualize
    from data import cfg_mnet, cfg_re18, cfg_re34, cfg_re50, cfg_eff_b0, cfg_eff_b4

    with torch.no_grad():
        for cfg in [cfg_re18, cfg_re34]:
        # for cfg in [cfg_eff_b0, cfg_eff_b4]:
            model = RetinaFaceDetector(cfg, device='cpu')
            # model(torch.rand(1, 3, 480, 480))
            stat(model, input_size=(3, 480, 480))

    # image = cv2.imread('../data/widerface/WIDER_val/images/0--Parade/0_Parade_Parade_0_275.jpg', cv2.IMREAD_COLOR)
    # image = cv2.imread('../data/widerface/WIDER_val/images/0--Parade/0_Parade_marchingband_1_1004.jpg', cv2.IMREAD_COLOR)
    image = cv2.imread('/home/louishsu/Desktop/0_Parade_marchingband_1_849.jpg', cv2.IMREAD_COLOR)
    
    # detector = RetinaFaceDetector(cfg=cfg_re18, weights_path='outputs/resnet18_v1/Resnet18_iter_21000_2.6661_.pth')
    detector = RetinaFaceDetector(cfg=cfg_eff_b0, weights_path='outputs/Efficientnet-b0_v1/Efficientnet-b0_iter_85000_2.9441_.pth')

    timer = Timer()
    timer.tic()
    scores, dets, landms = detector.detect(image, confidence_threshold=0.5)
    timer.toc()
    print(f"Cost {timer.total_time:f}s")

    image = visualize(image, dets, landms, scores)

    cv2.imwrite('/home/louishsu/Desktop/res.jpg', image)
    cv2.imshow('', image)
    cv2.waitKey(0)
