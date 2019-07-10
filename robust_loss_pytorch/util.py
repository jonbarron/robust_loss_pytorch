# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
import torch_dct


def log_safe(x):
  """The same as torch.log(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.log(torch.min(x, torch.tensor(33e37).to(x)))


def log1p_safe(x):
  """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))


def exp_safe(x):
  """The same as torch.exp(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.exp(torch.min(x, torch.tensor(87.5).to(x)))


def expm1_safe(x):
  """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))


def inv_softplus(y):
  """The inverse of tf.nn.softplus()."""
  y = torch.as_tensor(y)
  return torch.where(y > 87.5, y, torch.log(torch.expm1(y)))


def logit(y):
  """The inverse of tf.nn.sigmoid()."""
  y = torch.as_tensor(y)
  return -torch.log(1. / y - 1.)


def affine_sigmoid(logits, lo=0, hi=1):
  """Maps reals to (lo, hi), where 0 maps to (lo+hi)/2."""
  if not lo < hi:
    raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
  logits = torch.as_tensor(logits)
  lo = torch.as_tensor(lo)
  hi = torch.as_tensor(hi)
  alpha = torch.sigmoid(logits) * (hi - lo) + lo
  return alpha


def inv_affine_sigmoid(probs, lo=0, hi=1):
  """The inverse of affine_sigmoid(., lo, hi)."""
  if not lo < hi:
    raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
  probs = torch.as_tensor(probs)
  lo = torch.as_tensor(lo)
  hi = torch.as_tensor(hi)
  logits = logit((probs - lo) / (hi - lo))
  return logits


def affine_softplus(x, lo=0, ref=1):
  """Maps real numbers to (lo, infinity), where 0 maps to ref."""
  if not lo < ref:
    raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
  x = torch.as_tensor(x)
  lo = torch.as_tensor(lo)
  ref = torch.as_tensor(ref)
  shift = inv_softplus(torch.tensor(1.))
  y = (ref - lo) * torch.nn.Softplus()(x + shift) + lo
  return y


def inv_affine_softplus(y, lo=0, ref=1):
  """The inverse of affine_softplus(., lo, ref)."""
  if not lo < ref:
    raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
  y = torch.as_tensor(y)
  lo = torch.as_tensor(lo)
  ref = torch.as_tensor(ref)
  shift = inv_softplus(torch.tensor(1.))
  x = inv_softplus((y - lo) / (ref - lo)) - shift
  return x


def students_t_nll(x, df, scale):
  """The NLL of a Generalized Student's T distribution (w/o including TFP)."""
  x = torch.as_tensor(x)
  df = torch.as_tensor(df)
  scale = torch.as_tensor(scale)
  log_partition = torch.log(torch.abs(scale)) + torch.lgamma(
      0.5 * df) - torch.lgamma(0.5 * df + torch.tensor(0.5)) + torch.tensor(
          0.5 * np.log(np.pi))
  return 0.5 * ((df + 1.) * torch.log1p(
      (x / scale)**2. / df) + torch.log(df)) + log_partition


# A constant scale that makes tf.image.rgb_to_yuv() volume preserving.
_VOLUME_PRESERVING_YUV_SCALE = 1.580227820074


def rgb_to_syuv(rgb):
  """A volume preserving version of tf.image.rgb_to_yuv().

  By "volume preserving" we mean that rgb_to_syuv() is in the "special linear
  group", or equivalently, that the Jacobian determinant of the transformation
  is 1.

  Args:
    rgb: A tensor whose last dimension corresponds to RGB channels and is of
      size 3.

  Returns:
    A scaled YUV version of the input tensor, such that this transformation is
    volume-preserving.
  """
  rgb = torch.as_tensor(rgb)
  kernel = torch.tensor([[0.299, -0.14714119, 0.61497538],
                         [0.587, -0.28886916, -0.51496512],
                         [0.114, 0.43601035, -0.10001026]]).to(rgb)
  yuv = torch.reshape(
      torch.matmul(torch.reshape(rgb, [-1, 3]), kernel), rgb.shape)
  return _VOLUME_PRESERVING_YUV_SCALE * yuv


def syuv_to_rgb(yuv):
  """A volume preserving version of tf.image.yuv_to_rgb().

  By "volume preserving" we mean that rgb_to_syuv() is in the "special linear
  group", or equivalently, that the Jacobian determinant of the transformation
  is 1.

  Args:
    yuv: A tensor whose last dimension corresponds to scaled YUV channels and is
      of size 3 (ie, the output of rgb_to_syuv()).

  Returns:
    An RGB version of the input tensor, such that this transformation is
    volume-preserving.
  """
  yuv = torch.as_tensor(yuv)
  kernel = torch.tensor([[1, 1, 1], [0, -0.394642334, 2.03206185],
                         [1.13988303, -0.58062185, 0]]).to(yuv)
  rgb = torch.reshape(
      torch.matmul(torch.reshape(yuv, [-1, 3]), kernel), yuv.shape)
  return rgb / _VOLUME_PRESERVING_YUV_SCALE


def image_dct(image):
  """Does a type-II DCT (aka "The DCT") on axes 1 and 2 of a rank-3 tensor."""
  image = torch.as_tensor(image)
  dct_y = torch.transpose(torch_dct.dct(image, norm='ortho'), 1, 2)
  dct_x = torch.transpose(torch_dct.dct(dct_y, norm='ortho'), 1, 2)
  return dct_x


def image_idct(dct_x):
  """Inverts image_dct(), by performing a type-III DCT."""
  dct_x = torch.as_tensor(dct_x)
  dct_y = torch_dct.idct(torch.transpose(dct_x, 1, 2), norm='ortho')
  image = torch_dct.idct(torch.transpose(dct_y, 1, 2), norm='ortho')
  return image


def compute_jacobian(f, x):
  """Computes the Jacobian of function `f` with respect to input `x`."""
  vec = lambda z: torch.reshape(z, [-1])
  jacobian = []
  for i in range(np.prod(x.shape)):
    var_x = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
    y = vec(f(var_x))[i]
    y.backward()
    jacobian.append(np.array(vec(var_x.grad)))
  jacobian = np.stack(jacobian, 1)
  return jacobian
