import os
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision

import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.logger import DEFAULT_LOGGERS
from hyperopt import hp

from opts import opts
from models import wideresnet
from trainable import TrainCIFAR
from dataset import get_dataloaders
from lib.common import AverageMeter
from lib.augmentations import get_candidate_augment
from logger import JsonLogger


def main():

    args = opts().parse()
    train_sampler, train_loader, val_loader, test_loader = get_dataloaders(
        dataset='reduced_cifar10', batch=args.batch_size, dataroot="../../data/cifar10",
        split=0.2, split_idx=0, horovod=False
    )

    if not args.pass_train:
        model = wideresnet.Wide_ResNet(40, 2, 0.3, 10).to(args.device)
        train(model, train_loader, val_loader, args)
        del model
    del train_sampler, train_loader, test_loader

    for b in range(args.B):

        print(f"\n\n########### START SEARCHING : B{b} ############\n\n")

        ray.init()

        aug1, aug2 = get_candidate_augment(T=args.T)


        space = {
            "p1": hp.uniform("p1", 0, 1),
            "p2": hp.uniform("p2", 0, 1),
            "value1": hp.uniform("value1", 0, 1),
            "value2": hp.uniform("value2", 0, 1),
        }

        config = {
            "resources_per_trial": {
                "cpu": args.cpu,
                "gpu": args.gpu
            },
            "num_samples": args.num_samples,
            "config": {
                "args": args,
                "dataloader": val_loader,
                "aug1": aug1[0].__name__,
                "aug2": aug2[0].__name__,
                "iteration": 1,
            },
            "stop": {
                "training_iteration": 1
            }
        }

        algo = HyperOptSearch(
            space,
            max_concurrent=args.max_concurrent,
            metric="mean_accuracy",
            mode="max")
        scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", metric="mean_accuracy", mode="max")
        tune.run(TrainCIFAR,
                 search_alg=algo,
                 scheduler=scheduler,
                 # loggers=[DEFAULT_LOGGERS, JsonLogger],
                 loggers=[JsonLogger],
                 verbose=args.verbose,
                 **config)

        ray.shutdown()



def train(model, train_loader, val_loader, args):
    best_acc = -1
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4, nesterov=True)
    model.train()
    for epoch in range(args.epochs):
        loss_t, acc_t = train_epoch(model, train_loader, criterion, optimizer, args)
        loss, acc = validate(model, val_loader, criterion, args)
        print(f"Epoch[{epoch}/{args.epochs}]\t"
              f"Train Loss: {loss_t:.4f}\tAccuracy: {acc_t:.2f}\n"
              f"\t\t  Val Loss: {loss:.4f}\tAccuracy: {acc:.2f}")
        if acc > best_acc:
            print('Saving..')
            state = {
                'state_dict': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth.tar')
            best_acc = acc
    with open("./checkpoint/acc.txt", "a") as f:
        f.write(f"Best Accuracy: {best_acc:.2f}")
        f.write("\n")
    return model


def train_epoch(model, train_loader, criterion, optimizer, args):
    train_loss = 0
    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(train_loader):
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

    return train_loss/(i+1), 100.*correct/total

def validate(model, val_loader, criterion, args):
    val_loss = 0
    correct = 0
    total = 0
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

if __name__=="__main__":
    main()
