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
from dataset import get_dataloaders


def main():
    args = opts().parse()
    train_sampler, train_loader, val_loader, test_loader = get_dataloaders(
        dataset='cifar10', batch=args.batch_size, dataroot="../../data/cifar10",
        aug=args.aug, cutout=args.cutout, K=args.K
    )

    if args.gpu > 0:
        cudnn.benchmark = True

    model = wideresnet.Wide_ResNet(40, 2, 0.3, 10).to(args.device)
    _, best_score = train(model, train_loader, test_loader, args)



def train(model, train_loader, val_loader, args):
    file_name = f"checkpoint/trian_{args.log}_cp.pth.tar"
    best_name = f"checkpoint/train_{args.log}_bt.pth.tar"
    log_name = f"checkpoint/log_{args.log}.txt"
    best_acc, best_epoch = -1, -1
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr/(5**3))
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        loss_t, acc_t = train_epoch(model, train_loader, criterion, optimizer, args)
        loss, acc = validate(model, val_loader, criterion, args)
        scheduler.step()

        log = f"Epoch[{epoch}/{args.epochs}]\t" \
              f"Train Loss: {loss_t:.4f}\tAccuracy: {acc_t:.2f}\tLR: {optimizer.param_groups[0]['lr']:.6f}\n" \
              f"\t\t  Val Loss: {loss:.4f}\tAccuracy: {acc:.2f}\t{time.ctime()}"
        print(log)
        with open(log_name, "a") as f:
            f.write(log)
            f.write("\n")
        is_best = acc > best_acc
        best_epoch = epoch if is_best else best_epoch
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
          f"            EPOCH  : {best_epoch}\n" \
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