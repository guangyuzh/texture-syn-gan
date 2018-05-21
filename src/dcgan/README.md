# Deep Convolution Generative Adversarial Networks

This example implements the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

The implementation is very close to the Torch implementation [dcgan.torch](https://github.com/soumith/dcgan.torch)

After every 100 training iterations, the files `real_samples.png` and `fake_samples.png` are written to disk
with the samples from the generative model.

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

## Downloading the dataset
You can download the LSUN dataset by cloning [this repo](https://github.com/fyu/lsun) and running
```
python download.py -c bedroom
```

## Usage
```
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT --config CONFIG

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | imagenet | folder | lfw
  --dataroot DATAROOT   path to dataset
  --config CONFIG       yaml name to use in config.yml
