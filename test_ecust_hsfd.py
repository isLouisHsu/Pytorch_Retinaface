from __future__ import print_function
import os
import cv2
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from data import EcustHsfdDetection, valid_preproc, load_datacube
from data import cfg_mnet, cfg_re18, cfg_re34, cfg_re50, cfg_eff_b0, cfg_eff_b4, cfg_re34_hsfd_finetune, cfg_re34_hsfd_not_finetune
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from utils._utils import seed_everything

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--seed', default=99)
# parser.add_argument('-m', '--trained_model', required=True, type=str, 
#                     help='Trained state_dict file path to open, eg. `./weights/Resnet50_Final.pth`')
# parser.add_argument('--network', required=True, type=str, 
#                     help='Backbone network mobile0.25 or resnet50, eg. `resnet50`')
# parser.add_argument('--save_folder', required=True, type=str, 
#                     help='Dir to save txt results, eg. `./widerface_evaluate/widerface_txt/`')
parser.add_argument('-m', '--trained_model', default='outputs/resnet34_hsfd_with_finetune/Resnet34_iter_900_0.2767_.pth', type=str, 
                    help='Trained state_dict file path to open, eg. `./weights/Resnet50_Final.pth`')
parser.add_argument('--network', default='resnet34_hsfd', type=str, 
                    help='Backbone network mobile0.25 or resnet50, eg. `resnet50`')
parser.add_argument('--save_folder', default='./widerface_evaluate/predictions/', type=str, 
                    help='Dir to save txt results, eg. `./widerface_evaluate/widerface_txt/`')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='../data/widerface/WIDER_val/images/', type=str, help='dataset path')
parser.add_argument('--dataset_file', default='../data/ecust_hsfd/Original_Image_jpg_indoor/labels.txt', help='Training dataset directory')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=1, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()
seed_everything(args.seed)
args.save_folder = os.path.join(args.save_folder, args.network)

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


@torch.no_grad()
def main():
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet18":
        cfg = cfg_re18
    elif args.network == "resnet34":
        cfg = cfg_re34
    elif args.network == "resnet50":
        cfg = cfg_re50
    elif args.network == "Efficientnet-b0":
        cfg = cfg_eff_b0
    elif args.network == "Efficientnet-b4":
        cfg = cfg_eff_b4
    elif args.network == "resnet34_hsfd":
        cfg = cfg_re34_hsfd_finetune
    elif args.network == "resnet34_hsfd_not_finetune":
        cfg = cfg_re34_hsfd_not_finetune

    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # # testing dataset
    # testset_folder = args.dataset_folder
    # # testset_list = args.dataset_folder[:-7] + "wider_val.txt"
    # # with open(testset_list, 'r') as fr:
    # #     test_dataset = fr.read().split()
    # test_dataset = []
    # for event in os.listdir(testset_folder):
    #     subdir = os.path.join(testset_folder, event)
    #     img_names = os.listdir(subdir)
    #     for img_name in img_names:
    #         test_dataset.append([event, os.path.join(subdir, img_name)])
    # num_images = len(test_dataset)

    used_channels = cfg['used_channels']
    img_dim = cfg['image_size']
    test_dataset = EcustHsfdDetection(args.dataset_file, used_channels, 
        preproc=valid_preproc(img_dim, None), mode='valid')
    num_images = len(test_dataset)
    datadir = '/'.join(args.dataset_file.split('/')[:-1])

    pred_file = os.path.join(f'{args.save_folder:s}_pred.txt')
    gt_file = os.path.join(f'{args.save_folder:s}_gt.txt')
    fp1 = open(pred_file, 'w')
    fp2 = open(gt_file, 'w')


    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset.imgs_path):
        if i % 100 == 0:
            torch.cuda.empty_cache()

        # image_path = testset_folder + img_name
        img_raw = load_datacube(img_name)[..., used_channels]
        # img_raw = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = img_dim
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            
            img = np.stack([
                cv2.resize(img[..., i], None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR) \
                for i in range(img.shape[-1])
            ], axis=-1)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = (img - 127.5) / 128.0
        # img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        prediction = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # --------------------------------------------------------------------
        # save_name = os.path.join(args.save_folder, img_name.split('/')[-1].split('.')[0] + ".txt")
        # dirname = os.path.dirname(save_name)
        # if not os.path.isdir(dirname):
        #     os.makedirs(dirname)
        # with open(save_name, "w") as fd:
        #     bboxs = dets
        #     file_name = os.path.basename(save_name)[:-4] + "\n"
        #     bboxs_num = str(len(bboxs)) + "\n"
        #     fd.write(file_name)
        #     fd.write(bboxs_num)
        #     for box in bboxs:
        #         x = int(box[0])
        #         y = int(box[1])
        #         w = int(box[2]) - int(box[0])
        #         h = int(box[3]) - int(box[1])
        #         confidence = str(box[4])
        #         line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
        #         fd.write(line)
        
        fp1.write(f"# {img_name.lstrip(datadir).lstrip('/')}\n")
        if dets.shape[0] > 0:
            
            dets = prediction[0][:4].astype(np.int).tolist()
            dets[2] -= dets[0]; dets[3] -= dets[1]
            landms = prediction[0][4: 14]
            scores = prediction[0][14]

            label = [0. for _ in range(20)]
            label[-1] = scores
            label[:4] = dets
            label[4: -1] = landms
            label = ' '.join(list(map(str, label)))
            fp1.write(f'{label}\n')

        gt_label = ' '.join(list(map(str, test_dataset.words[i][0])))
        fp2.write(f"# {img_name.lstrip(datadir).lstrip('/')}\n")
        fp2.write(f'{gt_label}\n')
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # # save image
        # if args.save_image:
        #     for b in dets:
        #         if b[4] < args.vis_thres:
        #             continue
        #         text = "{:.4f}".format(b[4])
        #         b = list(map(int, b))
        #         cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        #         cx = b[0]
        #         cy = b[1] + 12
        #         cv2.putText(img_raw, text, (cx, cy),
        #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        #         # landms
        #         cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        #         cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        #         cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        #         cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        #         cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        #     # save image
        #     if not os.path.exists("./results/"):
        #         os.makedirs("./results/")
        #     name = "./results/" + str(i) + ".jpg"
        #     cv2.imwrite(name, img_raw)
    
    fp1.close()


if __name__ == "__main__":
    main()