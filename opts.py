import argparse
import torch

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # main condition
        self.parser.add_argument("--B", default=200, type=int, help="search depth")
        self.parser.add_argument("--T", default=2, type=int, help="length of sub policy")
        self.parser.add_argument("--num_samples", default=3, type=int, help="sampling number in hyperopt")
        self.parser.add_argument("--max_concurrent", default=4, type=int, help="max trials in concurrent")
        self.parser.add_argument("--verbose", default=0, type=int, help="verbose")
        self.parser.add_argument("--K", default=10, type=int, help="Top k best policy")
        self.parser.add_argument("--split_idx", default=0, type=int, help="index of kfold")

        # training condition
        self.parser.add_argument("--epochs", default=1, type=int)
        self.parser.add_argument("--lr", default=0.1, type=float)
        self.parser.add_argument("--batch_size", default=16, type=int)
        self.parser.add_argument("--workers", default=1, type=int)
        self.parser.add_argument("--pin_memory", action="store_true")

        # other condition
        self.parser.add_argument("--pass_train", action="store_true")
        self.parser.add_argument("--gpu", default=0, type=float)
        self.parser.add_argument("--cpu", default=4, type=float)
        self.parser.add_argument("--aws", action="store_true")
        self.parser.add_argument("--my_home", action="store_true")
        self.parser.add_argument("--baseline", action="store_true")
        self.parser.add_argument("--cutout", default=0, type=float)
        self.parser.add_argument("--aug", default="default")
        self.parser.add_argument("--log", default="default")

    def parse(self):
        args = self.parser.parse_args()
        args.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu>0 else "cpu")
        return args
