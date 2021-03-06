from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from tqdm import tqdm
from data import (WiderFaceDetection, detection_collate, 
    preproc, train_preproc, valid_preproc, 
    train_transformers, valid_transformers,
    cfg_mnet, cfg_re18, cfg_re34, cfg_re50, cfg_eff_b0, cfg_eff_b4)
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace

from utils import _utils

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--seed', default=99)
parser.add_argument('--version', default='v1')
parser.add_argument('--training_dataset', default='../data/widerface/WIDER_train/label.txt', help='Training dataset directory')
# parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--network', default='resnet18', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--network', default='resnet34', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--network', default='Efficientnet-b0', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--network', default='Efficientnet-b4', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./outputs/', help='Location to save checkpoint models')
parser.add_argument('--valid_steps', default=1000, help='Validation steps')
parser.add_argument('--verbose_steps', default=100, help='Validation steps')
args = parser.parse_args()

args.save_folder = os.path.join(args.save_folder, f'{args.network}_{args.version}')
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
_utils.seed_everything(args.seed)
logger = _utils.init_logger(__name__, log_file=os.path.join(args.save_folder, 'log.txt'))

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

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder
valid_steps = args.valid_steps
verbose_steps = args.verbose_steps


def train():

    net = RetinaFace(cfg=cfg)
    logger.info("Printing net...")
    logger.info(net)

    if args.resume_net is not None:
        logger.info('Loading resume network...')
        state_dict = torch.load(args.resume_net)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    cudnn.benchmark = True

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    net.train()
    epoch = 0 + args.resume_epoch
    logger.info('Loading Dataset...')

    trainset = WiderFaceDetection(training_dataset, preproc=train_preproc(img_dim, rgb_mean), mode='train')
    validset = WiderFaceDetection(training_dataset, preproc=valid_preproc(img_dim, rgb_mean), mode='valid')
    # trainset = WiderFaceDetection(training_dataset, transformers=train_transformers(img_dim), mode='train')
    # validset = WiderFaceDetection(training_dataset, transformers=valid_transformers(img_dim), mode='valid')
    trainloader = data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
    validloader = data.DataLoader(validset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
    logger.info(f'Totally {len(trainset)} training samples and {len(validset)} validating samples.')

    epoch_size = math.ceil(len(trainset) / batch_size)
    max_iter = max_epoch * epoch_size
    logger.info(f'max_epoch: {max_epoch:d} epoch_size: {epoch_size:d}, max_iter: {max_iter:d}')

    # optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = _utils.get_linear_schedule_with_warmup(optimizer, int(0.1 * max_iter), max_iter)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    best_loss_val = float('inf')
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            # batch_iterator = iter(tqdm(trainloader, total=len(trainloader)))
            batch_iterator = iter(trainloader)
            # if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
            #     torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1
            torch.cuda.empty_cache()

        if (valid_steps > 0) and (iteration > 0) and (iteration % valid_steps == 0):
            net.eval()
            # validation
            loss_l_val = 0.
            loss_c_val = 0.
            loss_landm_val = 0.
            loss_val = 0.
            # for val_no, (images, targets) in tqdm(enumerate(validloader), total=len(validloader)):
            for val_no, (images, targets) in enumerate(validloader):
                # load data
                images = images.cuda()
                targets = [anno.cuda() for anno in targets]
                # forward
                with torch.no_grad():
                    out = net(images)
                    loss_l, loss_c, loss_landm = criterion(out, priors, targets)
                    loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
                loss_l_val += loss_l.item()
                loss_c_val += loss_c.item()
                loss_landm_val += loss_landm.item()
                loss_val += loss.item()
            loss_l_val /= len(validloader)
            loss_c_val /= len(validloader)
            loss_landm_val /= len(validloader)
            loss_val /= len(validloader)
            logger.info('[Validating] Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Total: {:.4f} Loc: {:.4f} Cla: {:.4f} Landm: {:.4f}'
                .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                epoch_size, iteration + 1, max_iter, 
                loss_val, loss_l_val, loss_c_val, loss_landm_val))
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                pth = os.path.join(save_folder, cfg['name'] + '_iter_' + str(iteration) + f'_{loss_val:.4f}_' + '.pth')
                torch.save(net.state_dict(), pth)
                logger.info(f'Best validating loss: {best_loss_val:.4f}, model saved as {pth:s})')
            net.train()

        load_t0 = time.time()
        # if iteration in stepvalues:
        #     step_index += 1
        # lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        scheduler.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        if iteration % verbose_steps == 0:
            logger.info('[Training] Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Total: {:.4f} Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                epoch_size, iteration + 1, max_iter, 
                loss.item(), loss_l.item(), loss_c.item(), loss_landm.item(), 
                scheduler.get_last_lr()[-1], batch_time, str(datetime.timedelta(seconds=eta))))

    # torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
