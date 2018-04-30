import argparse
import os
import sys
import torch.backends.cudnn as cudnn
import yaml
from network import GANNetwork

cudnn.benchmark = True


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(_AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--config', default='default', help='yaml name to use in config.yml')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    parser = _get_parser()
    opt = _AttrDict(yaml.load(open('config.yml'))[parser.config])
    opt.dataset = parser.dataset
    opt.dataroot = parser.dataroot
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    gan_net = GANNetwork(opt)
    gan_net.train()
