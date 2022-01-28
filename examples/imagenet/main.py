"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import PIL
from glob import glob
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from CustomDataset import *
from sklearn.model_selection import train_test_split
import pandas as pd
from efficientnet_pytorch import EfficientNet
import numpy as np
from tqdm import tqdm
def csv_reader(enc_dec_mode=0):
    csv_features = [
        "내부 온도 1 평균",
        "내부 온도 1 최고",
        "내부 온도 1 최저",
        "내부 습도 1 평균",
        "내부 습도 1 최고",
        "내부 습도 1 최저",
        "내부 이슬점 평균",
        "내부 이슬점 최고",
        "내부 이슬점 최저",
    ]
    csv_files = sorted(glob("/DL_data_big/DACON_2022_LeafChallenge/leaf_data/data/train/*/*.csv"))

    # temp_csv = pd.read_csv(csv_files[0])[csv_features]
    # max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # # feature 별 최대값, 최솟값 계산
    # for csv in tqdm(csv_files[1:]):
    #     temp_csv = pd.read_csv(csv)[csv_features]
    #     temp_csv = temp_csv.replace("-", np.nan).dropna()
    #     if len(temp_csv) == 0:
    #         continue
    #     temp_csv = temp_csv.astype(float)
    #     temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
    #     max_arr = np.max([max_arr, temp_max], axis=0)
    #     min_arr = np.min([min_arr, temp_min], axis=0)

    # feature 별 최대값, 최솟값 dictionary 생성
    min_arr = np.array([3.4, 3.4, 3.3, 23.7, 25.9, 0, 0.1, 0.2, 0.0])
    max_arr = np.array([46.8, 47.1, 46.6, 100, 100, 100, 34.5, 34.7, 34.4])

    csv_feature_dict = {
        csv_features[i]: [min_arr[i], max_arr[i]] for i in range(len(csv_features))
    }
    # 변수 설명 csv 파일 참조
    crop = {"1": "딸기", "2": "토마토", "3": "파프리카", "4": "오이", "5": "고추", "6": "시설포도"}
    disease = {
        "1": {
            "a1": "딸기잿빛곰팡이병",
            "a2": "딸기흰가루병",
            "b1": "냉해피해",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "2": {
            "a5": "토마토흰가루병",
            "a6": "토마토잿빛곰팡이병",
            "b2": "열과",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "3": {
            "a9": "파프리카흰가루병",
            "a10": "파프리카잘록병",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "4": {
            "a3": "오이노균병",
            "a4": "오이흰가루병",
            "b1": "냉해피해",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "5": {
            "a7": "고추탄저병",
            "a8": "고추흰가루병",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "6": {"a11": "시설포도탄저병", "a12": "시설포도노균병", "b4": "일소피해", "b5": "축과병"},
    }
    risk = {"1": "초기", "2": "중기", "3": "말기"}

    label_description = {}  # classification 111 number ex) '딸기_다량원소결핍 (P)_말기'

    label_description_crop = {}
    label_description_disease = {}
    label_description_risk = {}
    for key, value in disease.items():
        label_description[f"{key}_00_0"] = f"{crop[key]}_정상"
        for disease_code in value:
            for risk_code in risk:
                label = f"{key}_{disease_code}_{risk_code}"
                label_crop = f"{key}"
                label_disease = f"{disease_code}"
                label_risk = f"{risk_code}"

                label_description[
                    label
                ] = f"{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}"
                label_description_crop[label_crop] = f"{crop[key]}"
                label_description_disease[
                    label_disease
                ] = f"{disease[key][disease_code]}"
                label_description_risk[label_risk] = f"{risk[risk_code]}"

    label_description_disease["00"] = "정상"
    label_description_risk["0"] = "정상"

    # ex) '1_00_0' : 0
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_encoder_crop = {key: idx for idx, key in enumerate(label_description_crop)}
    label_encoder_disease = {
        key: idx for idx, key in enumerate(label_description_disease)
    }
    label_encoder_risk = {key: idx for idx, key in enumerate(label_description_risk)}

    # ex) '0' : '1_00_0'
    label_decoder = {val: key for key, val in label_encoder.items()}
    label_decoder_crop = {val: key for key, val in label_encoder_crop.items()}
    label_decoder_disease = {val: key for key, val in label_encoder_disease.items()}
    label_decoder_risk = {val: key for key, val in label_encoder_risk.items()}

    # print(label_decoder)
    if enc_dec_mode == 0:
        return csv_feature_dict, label_encoder, label_decoder
    else:
        return (
            csv_feature_dict,
            [label_encoder_crop, label_encoder_disease, label_encoder_risk],
            [label_decoder_crop, label_decoder_disease, label_decoder_risk],
        )


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch,num_classes=111)

    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    train2 = sorted(glob("/DL_data_big/DACON_2022_LeafChallenge/leaf_data/data/train/*"))
    test = sorted(glob("/DL_data_big/DACON_2022_LeafChallenge/leaf_data/data/test/*"))
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
    else:
        image_size = args.image_size

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(image_size),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    labelsss = pd.read_csv("/works/EfficientNet-PyTorch/examples/imagenet/train.csv")["label"]
    train, val = train_test_split(train2, test_size=0.2, stratify=labelsss)
    _, label_encoder,label_decoder= csv_reader(enc_dec_mode=0)
    train_dataset = CustomDataset(train, label_encoder)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=False)

    # val_transforms = transforms.Compose([
    #     transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    print('Using image size', image_size)

    val_dataset = CustomDataset(val,label_encoder)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16, shuffle=False)


    test_dataset = CustomDataset(test,label_encoder,mode="test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=16, shuffle=False)
    if args.evaluate:
        res = validate(val_dataloader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_on(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_dataloader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        


def train_on(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # for i, (images, target) in enumerate(train_loader):
    for i, dict_ in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = dict_['img'].cuda(args.gpu, non_blocking=True)
        target = dict_['label'].cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        # for i, (images, target) in enumerate(val_loader):
        for i, dict_ in enumerate(val_loader):
            if args.gpu is not None:
                images = dict_['img'].cuda(args.gpu, non_blocking=True)
            target = dict_['label'].cuda(args.gpu, non_blocking=True)
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def predict(dataset,model,args):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    results = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item["img"].to(args.gpu)
        # seq = batch_item["csv_feature"].to(args.gpu)
        with torch.no_grad():
            output = model(img)
        output = (
            torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
        )
        results.extend(output)
    return results

def save_checkpoint(state, is_best, filename='checkpoint_eff_mt.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_eff_mt.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
