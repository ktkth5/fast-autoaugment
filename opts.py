import argparse
import torch

class hp(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # training condition
        self.parser.add_argument("--epochs", default=1000, type=int)
        self.parser.add_argument("--lr", default=0.00225, type=float)
        self.parser.add_argument("--batch_size", default=32, type=int)
        self.parser.add_argument("--workers", default=1, type=int)
        self.parser.add_argument("--pin_memory", action="store_true")

    def parse(self):
        args = self.parser.parse_args()
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return args
