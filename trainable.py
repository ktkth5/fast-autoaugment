import os
import random
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision

import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.suggest.hyperopt import HyperOptSearch

from hyperopt import hp

from opts import opts
from models import wideresnet
from dataset import get_dataloaders
from lib.common import AverageMeter
from lib.augmentations import get_candidate_augment, apply_augment


class TrainCIFAR(Trainable):
    def _setup(self, config):
        args = config.pop("args")
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.val_loader = config.pop("dataloader")
        augs = [[config["aug1"], config["p1"], config["value1"]],
                [config["aug2"], config["p2"], config["value2"]]]
        self.val_loader.dataset.transform.transforms.insert(0, Augmentation(augs))

        # self.model = wideresnet.Wide_ResNet(40, 2, 0.3, 10).to(args.device)
        # if args.aws:
        #     cp = torch.load("/home/ubuntu/Develope/fast-autoaugment/checkpoint/ckpt.pth.tar")
        # elif args.my_home:
        #     cp = torch.load("/home/kento/Develope/my_products/fast-autoaugment/checkpoint/ckpt.pth.tar")
        # else:
        #     cp = torch.load("/Users/kento/Development/my_pc/fast-autoaugment/checkpoint/ckpt.pth.tar")
        # self.model.load_state_dict(cp["state_dict"])
        self.model = config["model"].to(args.device)

        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.args = args

    def _train(self):
        return self._test()

    def _test(self):
        val_loss = 0
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return {"mean_accuracy": 100.*correct/total, "mean_loss": val_loss/(i+1)}

    def _save(self, checkpoint_dir):
        return

    def _restore(self, checkpoint):
        return


class Augmentation(object):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img):
        for name, pr, level in self.policy:
            if random.random() > pr:
                continue
            img = apply_augment(img, name, level)
        return img
