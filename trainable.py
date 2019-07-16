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

from lib.augmentations import apply_augment


class TrainCIFAR(Trainable):
    def _setup(self, config):
        args = config.pop("args")
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.val_loader = config.pop("dataloader")
        augs = [[config["aug1"], config["p1"], config["value1"]],
                [config["aug2"], config["p2"], config["value2"]]]
        self.val_loader.dataset.transform.transforms.insert(0, Augmentation(augs))
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
            for _ in range(3):
                for i, (inputs, targets) in enumerate(self.val_loader):
                    inputs = inputs.to(self.args.device)
                    targets = targets.to(self.args.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        return {"mean_accuracy": 100.*correct/total, "mean_loss": val_loss/(i+1)/2}

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
