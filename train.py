import os
import argparse
from copy import deepcopy
import shutil
import time

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import torch.backends.cudnn as cudnn

from opts import opts
from models import wideresnet
from trainable import TrainCIFAR
from dataset import get_dataloaders, Augmentation
from lib.augmentations import fa_reduced_cifar_repro


def main():
    args = opts().parse()
    train_sampler, train_loader, val_loader, test_loader = get_dataloaders(
        dataset='cifar10', batch=args.batch_size, dataroot="../../data/cifar10",
        aug=args.aug, cutout=args.cutout
    )
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # if not args.baseline:
    #     transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar_repro()))
    #
    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.CIFAR10(
    #         root="../../data/cifar10", train=True, download=True,
    #         transform=transform_train
    #     ),
    #     batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True,drop_last=True
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.CIFAR10(
    #         root="../../data/cifar10", train=False, download=True,
    #         transform=transform_test
    #     ),
    #     batch_size=args.batch_size*2,shuffle=False,num_workers=4,pin_memory=True,drop_last=False
    # )
    if args.gpu > 0:
        cudnn.benchmark = True

    model = wideresnet.Wide_ResNet(40, 2, 0.3, 10).to(args.device)
    _, best_score = train(model, train_loader, test_loader, args)



def train(model, train_loader, val_loader, args):
    file_name = f"checkpoint/trian_{args.log}_cp.pth.tar"
    best_name = f"checkpoint/train_{args.log}_bt.pth.tar"
    log_name = f"checkpoint/log_{args.log}.txt"
    best_acc = -1
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        loss_t, acc_t = train_epoch(model, train_loader, criterion, optimizer, args)
        loss, acc = validate(model, val_loader, criterion, args)

        log = f"Epoch[{epoch}/{args.epochs}]\t" \
              f"Train Loss: {loss_t:.4f}\tAccuracy: {acc_t:.2f}\tLR: {optimizer.param_groups[0]['lr']:.6f}\n" \
              f"\t\t  Val Loss: {loss:.4f}\tAccuracy: {acc:.2f}\t{time.ctime()}"
        print(log)
        with open(log_name, "a") as f:
            f.write(log)
            f.write("\n")
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        state = {
            'state_dict': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_checkpoint(state, is_best, file_name, best_name)
    with open(f"./checkpoint/acc_{args.log}.txt", "a") as f:
        f.write(f"Best Accuracy: {best_acc:.2f}")
        f.write("\n")

    log = f"=======================================\n" \
          f"          FINISHED TRAINING\n" \
          f"         BEST SCORE: {best_acc:.4f}\n" \
          f"======================================="
    print(log)
    with open(log_name, "a") as f:
        f.write(log)
        f.write("\n")
    return model, best_acc


def train_epoch(model, train_loader, criterion, optimizer, args):
    train_loss = 0
    correct = 0
    total = 0
    data_time, model_time = 0, 0
    end = time.time()
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        data_time += time.time() - end
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(f"{i}/{len(train_loader)}\t", loss.item(), correct/total)
        # break

        model_time += time.time() - end
        end = time.time()

    # print(f"data_time: {data_time/(i+1):.5f}\tmodel_time: {model_time/(i+1):.5f}")

    return train_loss/(i+1), 100.*correct/total


def validate(model, val_loader, criterion, args):
    val_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return val_loss/(i+1), 100.*correct/total


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    if epoch > 160:
        lr = 0.0008
    elif epoch > 120:
        lr = 0.004
    elif epoch > 60:
        lr = 0.02
    else:
        lr = 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__=="__main__":
    main()