import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

# 读取vgg文件中的网络名字；__dict__为vgg类属性；islower()选择小写字母；name.startswith选择开头字母；callable选择可访问属性
model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))


# argparse.ArgumentParser控制输入参数；metavar作用与help信息输出；help作为输入提示；choices选择参数；default默认参数；type类型
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)


best_prec1 = 0


def main():
    global args, best_prec1
    # 获取用户输入参数
    args = parser.parse_args()


    # 检查存储路径是否存在，如果不存在新建文件夹
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 获取用户选择的模型
    model = vgg.__dict__[args.arch]()
    # 多gpu分布式训练：torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
    # module (Module) – module to be parallelized
    # device_ids (list of python:int or torch.device) – CUDA devices (default: all devices)
    # output_device (int or torch.device) – device location of output (default: device_ids[0])
    model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        # 使用cpu，将操作tensor放在cpu上
        model.cpu()
    else:
        # 使用gpu，将操作tensor放在gpu上
        model.cuda()

    # optionally resume from a checkpoint
    # 读取上次训练的检查节点
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # torch.load 读取一个存储的map；map_location：重新指定map所在位置（cpu or gpu）
            checkpoint = torch.load(args.resume)
            # 迭代次数、最佳预测结果更新
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # load_state_dict将导入的模型参数赋值到self中
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # torch.backends.cudnn.benchmark=True ：
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定。
    cudnn.benchmark = True

    # torchvision.transforms.Normalize(mean, std)：输入图片预处理给定均值和方差（mean，std）,下面为三通道方式
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # torch.utils.data.DataLoader：读取数据集（在这里是读取训练集）
    # class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0,
        # collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
    # dataset (Dataset) – 从中​​加载数据的数据集。
    # batch_size (int, optional) – 批训练的数据个数(默认: 1)。
    # shuffle (bool, optional) – 设置为True在每个epoch重新排列数据（默认值：False,一般打乱比较好）。
    # **sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
    # num_workers (int, 可选) – 用于数据加载的子进程数。0表示数据将在主进程中加载​​（默认值：0）
    # **collate_fn (callable, optional) – 合并样本列表以形成小批量。
    # pin_memory (bool, optional) – 如果为True，数据加载器在返回前将张量复制到CUDA固定内存中。
    # drop_last (bool, optional) – 如果数据集大小不能被batch_size整除，设置为True可删除最后一个不完整的批处理。如果设为False并且数据集的大小不能被batch_size整除，则最后一个batch将更小。(默认: False)
    train_loader = torch.utils.data.DataLoader(
        # torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
        # 读取数据集或从互联网下载数据集，这里采用CIFAR10，其他数据集的参数也一样
        # cifar-10-batches-py的根目录
        # train : true为训练集，false为测试集
        # transform : 对PILimage的预处理函数
            # transforms.Compose：transforms函数集合
            # https://pytorch.org/docs/master/torchvision/transforms.html?highlight=transforms
            # transforms.RandomHorizontalFlip(p=0.5)：p概率水平翻转
            # torchvision.transforms.RandomCrop(size, padding=0)：随机中心点切取图片
            # torchvision.transforms.ToTensor：将ndarray转换为张量tensor
            # normalize：归一化
            # download：允许下载
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # 读取验证集
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    # 定义损失函数和优化器（损失函数为交叉熵（CrossEntropyLoss），优化器为带momentum和weight_decay的随机梯度下降）
    criterion = nn.CrossEntropyLoss()
    # 将损失函数放在cpu或者gpu上
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()

    # 使用半精度 16-bit
    if args.half:
        model.half()
        criterion.half()

    # torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    # params：用于迭代优化的参数
    # lr：learning rate
    # momentum：动量因子
    # weight_decay：l2范数因子
    # nesterov：是否使用nesterov momentum
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 如果选择需要初始时validation？
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # 对于每个epoch循环
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        # 如果当前epochs验证准确率最高，更新模型权重
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cpu == False:
            input,target = input.cuda(),target.cuda() #原文为 input = input.cuda(async=True)。python3.7已移除async字段

        if args.half:
            input = input.half()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """

    # batch_time：batch的运行时间
    # losses：代价函数值
    # top1：准确度
    # AverageMeter：计算平均值、合、更新的工具类
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    # 更新到验证模式:相反则用torch.nn.train
    model.eval()

    end = time.time()
    # 从验证集val_loader里获取图片input和标注target
    for i, (input, target) in enumerate(val_loader):
        # 使用cpu or gpu
        if args.cpu == False:
            input,target = input.cuda(),target.cuda()
        # 使用半精度 or not
        if args.half:
            input = input.half()

        # compute output
        # 将推断函数包含在torch.no_grad()，表示不需要求导，加快运行速度
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    # 学习率更新，每30epochs学习率更新为之前1/2
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # 求取准确度
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
