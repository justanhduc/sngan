# Spectral Normalized GAN
A Theano implementation of Spectral Normalized GAN

## Requirements

[Theano](http://deeplearning.net/software/theano/)

[neuralnet](https://github.com/justanhduc/neuralnet)

[Visdom](https://github.com/facebookresearch/visdom) (required if you want to monitor the training live on browser)

## Usages

```
python sngan.py path-to-CIFAR-dataset (--bs 64) (--n_ters 100000) (--gpu 0) (--n_dis 5) (--valid_freq 500) (--adam_alpha 0.0002) (--adam_beta1 0) (--adam_beta2 0) (--use_visdom 0)
```

## Samples

The following samples are obtained after 20 iters.

![samples @20k](https://github.com/justanhduc/sngan/blob/master/samples/samples.jpg)

## Credits

This implementation attempts to reproduce the results from

```
@article{yoshida2017spectral,
  title={Spectral Norm Regularization for Improving the Generalizability of Deep Learning},
  author={Yoshida, Yuichi and Miyato, Takeru},
  journal={arXiv preprint arXiv:1705.10941},
  year={2017}
}
```

The original implementation can be found [here](https://github.com/pfnet-research/chainer-gan-lib).
