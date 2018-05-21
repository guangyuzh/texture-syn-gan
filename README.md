# Periodic Spatial GAN for Texture Synthesis

### Usage
```bash
cd src/dcgan
python main.py --dataroot $PATH_TO_DTD_DATA --config archpc
```
To change configurations for hyperparameters, add definitions in `src/dcgan/config.yml`, and use the newly added yaml name (e.g., `archpc`) for the `--config` flag.

### Datasets
* [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

### Resources
* [PyTorch DCGAN example](https://github.com/pytorch/examples/tree/master/dcgan) (replicated [
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434))
* [Stabilizing Adversarial Nets with Prediction Methods](https://openreview.net/forum?id=Skj8Kag0Z&noteId=rkLymJTSf), ICLR 2018, [GitHub](https://github.com/sanghoon/prediction_gan)
* [From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

