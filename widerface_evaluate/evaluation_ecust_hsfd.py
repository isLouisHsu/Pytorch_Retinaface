# """
# WiderFace evaluation code
# author: wondervictor
# mail: tianhengcheng@gmail.com
# copyright@wondervictor
# """

# import os
# import tqdm
# import pickle
# import argparse
# import numpy as np
# from scipy.io import loadmat
# from bbox import bbox_overlaps
# from IPython import embed
# from collections import defaultdict
# from matplotlib import pyplot as plt


# def get_gt_boxes(gt_dir):
#     """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

#     gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
#     hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
#     medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
#     easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

#     facebox_list = gt_mat['face_bbx_list']
#     event_list = gt_mat['event_list']
#     file_list = gt_mat['file_list']

#     hard_gt_list = hard_mat['gt_list']
#     medium_gt_list = medium_mat['gt_list']
#     easy_gt_list = easy_mat['gt_list']

#     return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


# def get_gt_boxes_from_txt(gt_path, cache_dir):

#     cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
#     if os.path.exists(cache_file):
#         f = open(cache_file, 'rb')
#         boxes = pickle.load(f)
#         f.close()
#         return boxes

#     f = open(gt_path, 'r')
#     state = 0
#     lines = f.readlines()
#     lines = list(map(lambda x: x.rstrip('\r\n'), lines))
#     boxes = {}
#     print(len(lines))
#     f.close()
#     current_boxes = []
#     current_name = None
#     for line in lines:
#         if state == 0 and '--' in line:
#             state = 1
#             current_name = line
#             continue
#         if state == 1:
#             state = 2
#             continue

#         if state == 2 and '--' in line:
#             state = 1
#             boxes[current_name] = np.array(current_boxes).astype('float32')
#             current_name = line
#             current_boxes = []
#             continue

#         if state == 2:
#             box = [float(x) for x in line.split(' ')[:4]]
#             current_boxes.append(box)
#             continue

#     f = open(cache_file, 'wb')
#     pickle.dump(boxes, f)
#     f.close()
#     return boxes


# def read_pred_file(filepath):

#     with open(filepath, 'r') as f:
#         lines = f.readlines()
#         img_file = lines[0].rstrip('\n\r')
#         lines = lines[2:]

#     # b = lines[0].rstrip('\r\n').split(' ')[:-1]
#     # c = float(b)
#     # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
#     boxes = []
#     for line in lines:
#         line = line.rstrip('\r\n').split(' ')
#         if line[0] == '':
#             continue
#         # a = float(line[4])
#         boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
#     boxes = np.array(boxes)
#     # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
#     return img_file.split('/')[-1], boxes


# def get_preds(pred_dir):
#     events = [event for event in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, event))]
#     boxes = dict()
#     pbar = tqdm.tqdm(events)

#     for event in pbar:
#         pbar.set_description('Reading Predictions ')
#         event_dir = os.path.join(pred_dir, event)
#         event_images = os.listdir(event_dir)
#         current_event = dict()
#         for imgtxt in event_images:
#             imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
#             current_event[imgname.rstrip('.jpg')] = _boxes
#         boxes[event] = current_event
#     return boxes


# def norm_score(pred):
#     """ norm score
#     pred {key: [[x1,y1,x2,y2,s]]}
#     """

#     max_score = 0
#     min_score = 1

#     for _, k in pred.items():
#         for _, v in k.items():
#             if len(v) == 0:
#                 continue
#             _min = np.min(v[:, -1])
#             _max = np.max(v[:, -1])
#             max_score = max(_max, max_score)
#             min_score = min(_min, min_score)

#     diff = max_score - min_score
#     for _, k in pred.items():
#         for _, v in k.items():
#             if len(v) == 0:
#                 continue
#             v[:, -1] = (v[:, -1] - min_score)/diff


# def image_eval(pred, gt, ignore, iou_thresh):
#     """ single image evaluation
#     pred: Nx5
#     gt: Nx4
#     ignore:
#     """

#     _pred = pred.copy()
#     _gt = gt.copy()
#     pred_recall = np.zeros(_pred.shape[0])
#     recall_list = np.zeros(_gt.shape[0])
#     proposal_list = np.ones(_pred.shape[0])

#     _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
#     _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
#     _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
#     _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

#     overlaps = bbox_overlaps(_pred[:, :4], _gt)

#     for h in range(_pred.shape[0]):

#         gt_overlap = overlaps[h]
#         max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
#         if max_overlap >= iou_thresh:
#             if ignore[max_idx] == 0:
#                 recall_list[max_idx] = -1
#                 proposal_list[h] = -1
#             elif recall_list[max_idx] == 0:
#                 recall_list[max_idx] = 1

#         r_keep_index = np.where(recall_list == 1)[0]
#         pred_recall[h] = len(r_keep_index)
#     return pred_recall, proposal_list


# def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
#     pr_info = np.zeros((thresh_num, 2)).astype('float')
#     for t in range(thresh_num):

#         thresh = 1 - (t+1)/thresh_num
#         r_index = np.where(pred_info[:, 4] >= thresh)[0]
#         if len(r_index) == 0:
#             pr_info[t, 0] = 0
#             pr_info[t, 1] = 0
#         else:
#             r_index = r_index[-1]
#             p_index = np.where(proposal_list[:r_index+1] == 1)[0]
#             pr_info[t, 0] = len(p_index)
#             pr_info[t, 1] = pred_recall[r_index]
#     return pr_info


# def dataset_pr_info(thresh_num, pr_curve, count_face):
#     _pr_curve = np.zeros((thresh_num, 2))
#     for i in range(thresh_num):
#         _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
#         _pr_curve[i, 1] = pr_curve[i, 1] / count_face
#     return _pr_curve


# def voc_ap(rec, prec):

#     # correct AP calculation
#     # first append sentinel values at the end
#     mrec = np.concatenate(([0.], rec, [1.]))
#     mpre = np.concatenate(([0.], prec, [0.]))

#     # compute the precision envelope
#     for i in range(mpre.size - 1, 0, -1):
#         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

#     # to calculate area under PR curve, look for points
#     # where X axis (recall) changes value
#     i = np.where(mrec[1:] != mrec[:-1])[0]

#     # and sum (\Delta recall) * prec
#     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap


# def evaluation(pred_path, gt_path, iou_thresh=0.5):
#     facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
#     pred = get_preds(pred_path)
#     norm_score(pred)
#     event_num = len(event_list)
#     thresh_num = 1000
#     settings = ['easy', 'medium', 'hard']
#     setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
#     aps = []

#     # plt.figure()
#     for setting_id in range(3):
#         # different setting
#         gt_list = setting_gts[setting_id]
#         count_face = 0
#         pr_curve = np.zeros((thresh_num, 2)).astype('float')
#         # [hard, medium, easy]
#         pbar = tqdm.tqdm(range(event_num))
#         for i in pbar:
#             pbar.set_description('Processing {}'.format(settings[setting_id]))
#             event_name = str(event_list[i][0][0])
#             img_list = file_list[i][0]
#             pred_list = pred[event_name]
#             sub_gt_list = gt_list[i][0]
#             # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
#             gt_bbx_list = facebox_list[i][0]

#             for j in range(len(img_list)):
#                 pred_info = pred_list[str(img_list[j][0][0])]

#                 gt_boxes = gt_bbx_list[j][0].astype('float')
#                 keep_index = sub_gt_list[j][0]
#                 count_face += len(keep_index)

#                 if len(gt_boxes) == 0 or len(pred_info) == 0:
#                     continue
#                 ignore = np.zeros(gt_boxes.shape[0])
#                 if len(keep_index) != 0:
#                     ignore[keep_index-1] = 1
#                 pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

#                 _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

#                 pr_curve += _img_pr_info
#         pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

#         propose = pr_curve[:, 0]
#         recall = pr_curve[:, 1]

#         # plt.subplot(int(f'31{setting_id + 1:d}'))
#         # plt.plot(recall, propose)
#         # print(propose, recall)
#         np.savetxt(os.path.join(pred_path, f'propose_{settings[setting_id]}.txt'), propose)
#         np.savetxt(os.path.join(pred_path, f'recall_{settings[setting_id]}.txt'), recall)

#         ap = voc_ap(recall, propose)
#         aps.append(ap)

#     print("==================== Results ====================")
#     print("Easy   Val AP: {}".format(aps[0]))
#     print("Medium Val AP: {}".format(aps[1]))
#     print("Hard   Val AP: {}".format(aps[2]))
#     print("=================================================")

#     # plt.show()


# def _load_annotation(txt_path):
#     imgs_path = []
#     words = []
#     f = open(txt_path, 'r')
#     lines = f.readlines()
#     isFirst = True
#     labels = []
#     for line in lines:
#         line = line.rstrip()
#         if line.startswith('#'):
#             if isFirst is True:
#                 isFirst = False
#             else:
#                 labels_copy = labels.copy()
#                 words.append(labels_copy)
#                 labels.clear()
#             path = line[2:]
#             imgs_path.append(path)
#         else:
#             line = line.split(' ')
#             label = [float(x) for x in line]
#             labels.append(label)
#     words.append(labels)
#     return dict(zip(imgs_path, words))


# def evaluate_ecust_hsfd(pred_path, gt_path, thresh_num=1000, iou_thresh=0.5):

#     preds = _load_annotation(pred_path)
#     # preds = {k: [_v[:4] + _v[-1] for _v in v] for k, v in preds.items()}
#     preds = {k: np.array([_v[:4] + [_v[4]] for _v in v]) for k, v in preds.items()}
#     gts = _load_annotation(gt_path)
#     gts = {k: np.array([_v[:4] for _v in v]) for k, v in gts.items()}

#     for imgpath in gts.keys():
#         pred_info = preds[imgpath]
#         gt_boxes = gts[imgpath]
#         ignore = np.zeros(gt_boxes.shape[0])
#         pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

#         count_face = len(gt_boxes)
#         pr_curve = np.zeros((thresh_num, 2)).astype('float')
#         _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
#         pr_curve += _img_pr_info
#         pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

#         print()


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-p', '--pred', default='./predictions/resnet34_hsfd_pred.txt')
#     parser.add_argument('-g', '--gt', default='./predictions/resnet34_hsfd_gt.txt')
#     # parser.add_argument('-p', '--pred', default='./predictions/resnet34_hsfd_not_finetune_pred.txt')
#     # parser.add_argument('-g', '--gt', default='./predictions/resnet34_hsfd_not_finetune_gt.txt')
#     args = parser.parse_args()

#     evaluate_ecust_hsfd(args.pred, args.gt)












