# A General and Adaptive Robust Loss Function

This directory contains reference code for the paper
[A General and Adaptive Robust Loss Function](https://arxiv.org/abs/1701.03077),
Jonathan T. Barron CVPR, 2019

The code is implemented in Pytorch, and is a port of the TensorFlow
implementation at:
https://github.com/google-research/google-research/tree/master/robust_loss. The
required packages are listed in `requirements.txt`.

To use this code, include `general.py` or `adaptive.py` and call the loss
function. `general.py` implements the "general" form of the loss, which assumes
you are prepared to set and tune hyperparameters yourself, and `adaptive.py`
implements the "adaptive" form of the loss, which tries to adapt the
hyperparameters automatically and also includes support for imposing losses in
different image representations. The probability distribution underneath the
adaptive loss is implemented in `distribution.py`.

If you use this code, please cite it:
```
@article{BarronCVPR2019,
  Author = {Jonathan T. Barron},
  Title = {A General and Adaptive Robust Loss Function},
  Journal = {CVPR},
  Year = {2019}
}
```
