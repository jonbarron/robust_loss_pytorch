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
r"""Implements the adaptive form of the loss.

You should only use this function if 1) you want the loss to change it's shape
during training (otherwise use general.py) or 2) you want to impose the loss on
a wavelet or DCT image representation, a only this function has easy support for
that.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from robust_loss_pytorch import distribution
from robust_loss_pytorch import util
from robust_loss_pytorch import wavelet


class AdaptiveLossFunction(nn.Module):
  """The adaptive loss function on a matrix.

  This class behaves differently from general.lossfun() and
  distribution.nllfun(), which are "stateless", allow the caller to specify the
  shape and scale of the loss, and allow for arbitrary sized inputs. This
  class only allows for rank-2 inputs for the residual `x`, and expects that
  `x` is of the form [batch_index, dimension_index]. This class then
  constructs free parameters (torch Parameters) that define the alpha and scale
  parameters for each dimension of `x`, such that all alphas are in
  (`alpha_lo`, `alpha_hi`) and all scales are in (`scale_lo`, Infinity).
  The assumption is that `x` is, say, a matrix where x[i,j] corresponds to a
  pixel at location j for image i, with the idea being that all pixels at
  location j should be modeled with the same shape and scale parameters across
  all images in the batch. If the user wants to fix alpha or scale to be a
  constant,
  this can be done by setting alpha_lo=alpha_hi or scale_lo=scale_init
  respectively.
  """

  def __init__(self,
               num_dims,
               float_dtype,
               device,
               alpha_lo=0.001,
               alpha_hi=1.999,
               alpha_init=None,
               scale_lo=1e-5,
               scale_init=1.0):
    """Sets up the loss function.

    Args:
      num_dims: The number of dimensions of the input to come.
      float_dtype: The floating point precision of the inputs to come.
      device: The device to run on (cpu, cuda, etc).
      alpha_lo: The lowest possible value for loss's alpha parameters, must be
        >= 0 and a scalar. Should probably be in (0, 2).
      alpha_hi: The highest possible value for loss's alpha parameters, must be
        >= alpha_lo and a scalar. Should probably be in (0, 2).
      alpha_init: The value that the loss's alpha parameters will be initialized
        to, must be in (`alpha_lo`, `alpha_hi`), unless `alpha_lo` == `alpha_hi`
        in which case this will be ignored. Defaults to (`alpha_lo` +
        `alpha_hi`) / 2
      scale_lo: The lowest possible value for the loss's scale parameters. Must
        be > 0 and a scalar. This value may have more of an effect than you
        think, as the loss is unbounded as scale approaches zero (say, at a
        delta function).
      scale_init: The initial value used for the loss's scale parameters. This
        also defines the zero-point of the latent representation of scales, so
        SGD may cause optimization to gravitate towards producing scales near
        this value.
    """
    super(AdaptiveLossFunction, self).__init__()

    if not np.isscalar(alpha_lo):
      raise ValueError('`alpha_lo` must be a scalar, but is of type {}'.format(
          type(alpha_lo)))
    if not np.isscalar(alpha_hi):
      raise ValueError('`alpha_hi` must be a scalar, but is of type {}'.format(
          type(alpha_hi)))
    if alpha_init is not None and not np.isscalar(alpha_init):
      raise ValueError(
          '`alpha_init` must be None or a scalar, but is of type {}'.format(
              type(alpha_init)))
    if not alpha_lo >= 0:
      raise ValueError('`alpha_lo` must be >= 0, but is {}'.format(alpha_lo))
    if not alpha_hi >= alpha_lo:
      raise ValueError('`alpha_hi` = {} must be >= `alpha_lo` = {}'.format(
          alpha_hi, alpha_lo))
    if alpha_init is not None and alpha_lo != alpha_hi:
      if not (alpha_init > alpha_lo and alpha_init < alpha_hi):
        raise ValueError(
            '`alpha_init` = {} must be in (`alpha_lo`, `alpha_hi`) = ({} {})'
            .format(alpha_init, alpha_lo, alpha_hi))
    if not np.isscalar(scale_lo):
      raise ValueError('`scale_lo` must be a scalar, but is of type {}'.format(
          type(scale_lo)))
    if not np.isscalar(scale_init):
      raise ValueError(
          '`scale_init` must be a scalar, but is of type {}'.format(
              type(scale_init)))
    if not scale_lo > 0:
      raise ValueError('`scale_lo` must be > 0, but is {}'.format(scale_lo))
    if not scale_init >= scale_lo:
      raise ValueError('`scale_init` = {} must be >= `scale_lo` = {}'.format(
          scale_init, scale_lo))

    self.num_dims = num_dims
    if float_dtype == np.float32:
      float_dtype = torch.float32
    if float_dtype == np.float64:
      float_dtype = torch.float64
    self.float_dtype = float_dtype
    self.device = device
    if isinstance(device, int) or\
       (isinstance(device, str) and 'cuda' in device) or\
       (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.set_device(self.device)

    self.distribution = distribution.Distribution()

    if alpha_lo == alpha_hi:
      # If the range of alphas is a single item, then we just fix `alpha` to be
      # a constant.
      self.fixed_alpha = torch.tensor(
          alpha_lo, dtype=self.float_dtype,
          device=self.device)[np.newaxis, np.newaxis].repeat(1, self.num_dims)
      self.alpha = lambda: self.fixed_alpha
    else:
      # Otherwise we construct a "latent" alpha variable and define `alpha`
      # As an affine function of a sigmoid on that latent variable, initialized
      # such that `alpha` starts off as `alpha_init`.
      if alpha_init is None:
        alpha_init = (alpha_lo + alpha_hi) / 2.
      latent_alpha_init = util.inv_affine_sigmoid(
          alpha_init, lo=alpha_lo, hi=alpha_hi)
      self.register_parameter(
          'latent_alpha',
          torch.nn.Parameter(
              latent_alpha_init.clone().detach().to(
                  dtype=self.float_dtype,
                  device=self.device)[np.newaxis, np.newaxis].repeat(
                      1, self.num_dims),
              requires_grad=True))
      self.alpha = lambda: util.affine_sigmoid(
          self.latent_alpha, lo=alpha_lo, hi=alpha_hi)

    if scale_lo == scale_init:
      # If the difference between the minimum and initial scale is zero, then
      # we just fix `scale` to be a constant.
      self.fixed_scale = torch.tensor(
          scale_init, dtype=self.float_dtype,
          device=self.device)[np.newaxis, np.newaxis].repeat(1, self.num_dims)
      self.scale = lambda: self.fixed_scale
    else:
      # Otherwise we construct a "latent" scale variable and define `scale`
      # As an affine function of a softplus on that latent variable.
      self.register_parameter(
          'latent_scale',
          torch.nn.Parameter(
              torch.zeros((1, self.num_dims)).to(
                  dtype=self.float_dtype, device=self.device),
              requires_grad=True))
      self.scale = lambda: util.affine_softplus(
          self.latent_scale, lo=scale_lo, ref=scale_init)

  def lossfun(self, x, **kwargs):
    """Computes the loss on a matrix.

    Args:
      x: The residual for which the loss is being computed. Must be a rank-2
        tensor, where the innermost dimension is the batch index, and the
        outermost dimension must be equal to self.num_dims. Must be a tensor or
        numpy array of type self.float_dtype.
      **kwargs: Arguments to be passed to the underlying distribution.nllfun().

    Returns:
      A tensor of the same type and shape as input `x`, containing the loss at
      each element of `x`. These "losses" are actually negative log-likelihoods
      (as produced by distribution.nllfun()) and so they are not actually
      bounded from below by zero. You'll probably want to minimize their sum or
      mean.
    """
    x = torch.as_tensor(x)
    assert len(x.shape) == 2
    assert x.shape[1] == self.num_dims
    assert x.dtype == self.float_dtype
    return self.distribution.nllfun(x, self.alpha(), self.scale(), **kwargs)


class StudentsTLossFunction(nn.Module):
  """A variant of AdaptiveLossFunction that uses a Student's t-distribution."""

  def __init__(self,
               num_dims,
               float_dtype,
               device,
               scale_lo=1e-5,
               scale_init=1.0):
    """Sets up the adaptive loss for a matrix of inputs.

    Args:
      num_dims: The number of dimensions of the input to come.
      float_dtype: The floating point precision of the inputs to come.
      device: The device to run on (cpu, cuda, etc).
      scale_lo: The lowest possible value for the loss's scale parameters. Must
        be > 0 and a scalar. This value may have more of an effect than you
        think, as the loss is unbounded as scale approaches zero (say, at a
        delta function).
      scale_init: The initial value used for the loss's scale parameters. This
        also defines the zero-point of the latent representation of scales, so
        SGD may cause optimization to gravitate towards producing scales near
        this value.
    """
    super(StudentsTLossFunction, self).__init__()

    if not np.isscalar(scale_lo):
      raise ValueError('`scale_lo` must be a scalar, but is of type {}'.format(
          type(scale_lo)))
    if not np.isscalar(scale_init):
      raise ValueError(
          '`scale_init` must be a scalar, but is of type {}'.format(
              type(scale_init)))
    if not scale_lo > 0:
      raise ValueError('`scale_lo` must be > 0, but is {}'.format(scale_lo))
    if not scale_init >= scale_lo:
      raise ValueError('`scale_init` = {} must be >= `scale_lo` = {}'.format(
          scale_init, scale_lo))

    self.num_dims = num_dims
    if float_dtype == np.float32:
      float_dtype = torch.float32
    if float_dtype == np.float64:
      float_dtype = torch.float64
    self.float_dtype = float_dtype
    self.device = device
    if isinstance(device, int) or\
       (isinstance(device, str) and 'cuda' in device) or\
       (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.set_device(self.device)

    self.log_df = torch.nn.Parameter(
        torch.zeros(
            (1, self.num_dims)).to(dtype=self.float_dtype, device=self.device),
        requires_grad=True)
    self.register_parameter('log_df', self.log_df)

    if scale_lo == scale_init:
      # If the difference between the minimum and initial scale is zero, then
      # we just fix `scale` to be a constant.
      self.latent_scale = None
      self.scale = torch.tensor(
          scale_init, dtype=self.float_dtype,
          device=self.device)[np.newaxis, np.newaxis].repeat(1, self.num_dims)
    else:
      # Otherwise we construct a "latent" scale variable and define `scale`
      # As an affine function of a softplus on that latent variable.
      self.latent_scale = torch.nn.Parameter(
          torch.zeros(
              (1,
               self.num_dims)).to(dtype=self.float_dtype, device=self.device),
          requires_grad=True)
    self.register_parameter('latent_scale', self.latent_scale)
    self.df = lambda: torch.exp(self.log_df)
    self.scale = lambda: util.affine_softplus(
        self.latent_scale, lo=scale_lo, ref=scale_init)

  def lossfun(self, x):
    """A variant of lossfun() that uses the NLL of a Student's t-distribution.

    Args:
      x: The residual for which the loss is being computed. Must be a rank-2
        tensor, where the innermost dimension is the batch index, and the
        outermost dimension must be equal to self.num_dims. Must be a tensor or
        numpy array of type self.float_dtype.

    Returns:
      A tensor of the same type and shape as input `x`, containing the loss at
      each element of `x`. These "losses" are actually negative log-likelihoods
      (as produced by distribution.nllfun()) and so they are not actually
      bounded from below by zero. You'll probably want to minimize their sum or
      mean.
    """
    x = torch.as_tensor(x)
    assert len(x.shape) == 2
    assert x.shape[1] == self.num_dims
    assert x.dtype == self.float_dtype
    return util.students_t_nll(x, self.df(), self.scale())


class AdaptiveImageLossFunction(nn.Module):
  """A wrapper around AdaptiveLossFunction for handling images."""

  def transform_to_mat(self, x):
    """Transforms a batch of images to a matrix."""
    assert len(x.shape) == 4
    x = torch.as_tensor(x)
    if self.color_space == 'YUV':
      x = util.rgb_to_syuv(x)
    # If `color_space` == 'RGB', do nothing.

    # Reshape `x` from
    #   (num_batches, width, height, num_channels) to
    #   (num_batches * num_channels, width, height)
    _, width, height, num_channels = x.shape
    x_stack = torch.reshape(x.permute(0, 3, 1, 2), (-1, width, height))

    # Turn each channel in `x_stack` into the spatial representation specified
    # by `representation`.
    if self.representation in wavelet.generate_filters():
      x_stack = wavelet.flatten(
          wavelet.rescale(
              wavelet.construct(x_stack, self.wavelet_num_levels,
                                self.representation), self.wavelet_scale_base))
    elif self.representation == 'DCT':
      x_stack = util.image_dct(x_stack)
    # If `representation` == 'PIXEL', do nothing.

    # Reshape `x_stack` from
    #   (num_batches * num_channels, width, height) to
    #   (num_batches, num_channels * width * height)
    x_mat = torch.reshape(
        torch.reshape(x_stack,
                      (-1, num_channels, width, height)).permute(0, 2, 3, 1),
        [-1, width * height * num_channels])
    return x_mat

  def __init__(self,
               image_size,
               float_dtype,
               device,
               color_space='YUV',
               representation='CDF9/7',
               wavelet_num_levels=5,
               wavelet_scale_base=1,
               use_students_t=False,
               **kwargs):
    """Sets up the adaptive form of the robust loss on a set of images.

    This function is a wrapper around AdaptiveLossFunction. It requires inputs
    of a specific shape and size, and constructs internal parameters describing
    each non-batch dimension. By default, this function uses a CDF9/7 wavelet
    decomposition in a YUV color space, which often works well.

    Args:
      image_size: The size (width, height, num_channels) of the input images.
      float_dtype: The dtype of the floats used as input.
      device: The device to use.
      color_space: The color space that `x` will be transformed into before
        computing the loss. Must be 'RGB' (in which case no transformation is
        applied) or 'YUV' (in which case we actually use a volume-preserving
        scaled YUV colorspace so that log-likelihoods still have meaning, see
        util.rgb_to_syuv()). Note that changing this argument does not change
        the assumption that `x` is the set of differences between RGB images, it
        just changes what color space `x` is converted to from RGB when
        computing the loss.
      representation: The spatial image representation that `x` will be
        transformed into after converting the color space and before computing
        the loss. If this is a valid type of wavelet according to
        wavelet.generate_filters() then that is what will be used, but we also
        support setting this to 'DCT' which applies a 2D DCT to the images, and
        to 'PIXEL' which applies no transformation to the image, thereby causing
        the loss to be imposed directly on pixels.
      wavelet_num_levels: If `representation` is a kind of wavelet, this is the
        number of levels used when constructing wavelet representations.
        Otherwise this is ignored. Should probably be set to as large as
        possible a value that is supported by the input resolution, such as that
        produced by wavelet.get_max_num_levels().
      wavelet_scale_base: If `representation` is a kind of wavelet, this is the
        base of the scaling used when constructing wavelet representations.
        Otherwise this is ignored. For image_lossfun() to be volume preserving
        (a useful property when evaluating generative models) this value must be
        == 1. If the goal of this loss isn't proper statistical modeling, then
        modifying this value (say, setting it to 0.5 or 2) may significantly
        improve performance.
      use_students_t: If true, use the NLL of Student's T-distribution instead
        of the adaptive loss. This causes all `alpha_*` inputs to be ignored.
      **kwargs: Arguments to be passed to the underlying lossfun().

    Raises:
      ValueError: if `color_space` of `representation` are unsupported color
        spaces or image representations, respectively.
    """
    super(AdaptiveImageLossFunction, self).__init__()

    color_spaces = ['RGB', 'YUV']
    if color_space not in color_spaces:
      raise ValueError('`color_space` must be in {}, but is {!r}'.format(
          color_spaces, color_space))
    representations = wavelet.generate_filters() + ['DCT', 'PIXEL']
    if representation not in representations:
      raise ValueError('`representation` must be in {}, but is {!r}'.format(
          representations, representation))
    assert len(image_size) == 3

    self.color_space = color_space
    self.representation = representation
    self.wavelet_num_levels = wavelet_num_levels
    self.wavelet_scale_base = wavelet_scale_base
    self.use_students_t = use_students_t
    self.image_size = image_size

    if float_dtype == np.float32:
      float_dtype = torch.float32
    if float_dtype == np.float64:
      float_dtype = torch.float64
    self.float_dtype = float_dtype
    self.device = device
    if isinstance(device, int) or\
       (isinstance(device, str) and 'cuda' in device) or\
       (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.set_device(self.device)

    x_example = torch.zeros([1] + list(self.image_size)).type(self.float_dtype)
    x_example_mat = self.transform_to_mat(x_example)
    self.num_dims = x_example_mat.shape[1]

    if self.use_students_t:
      self.adaptive_lossfun = StudentsTLossFunction(self.num_dims,
                                                    self.float_dtype,
                                                    self.device, **kwargs)
    else:
      self.adaptive_lossfun = AdaptiveLossFunction(self.num_dims,
                                                   self.float_dtype,
                                                   self.device, **kwargs)

  def lossfun(self, x):
    """Computes the adaptive form of the robust loss on a set of images.

    Args:
      x: A set of image residuals for which the loss is being computed. Must be
        a rank-4 tensor of size (num_batches, width, height, color_channels).
        This is assumed to be a set of differences between RGB images.

    Returns:
      A tensor of losses of the same type and shape as input `x`. These "losses"
      are actually negative log-likelihoods (as produced by
      distribution.nllfun())
      and so they are not actually bounded from below by zero.
      You'll probably want to minimize their sum or mean.
    """
    x_mat = self.transform_to_mat(x)

    loss_mat = self.adaptive_lossfun.lossfun(x_mat)

    # Reshape the loss function's outputs to have the shapes as the input.
    loss = torch.reshape(loss_mat, [-1] + list(self.image_size))
    return loss

  def alpha(self):
    """Returns an image of alphas."""
    assert not self.use_students_t
    return torch.reshape(self.adaptive_lossfun.alpha(), self.image_size)

  def df(self):
    """Returns an image of degrees of freedom, for the Student's T model."""
    assert self.use_students_t
    return torch.reshape(self.adaptive_lossfun.df(), self.image_size)

  def scale(self):
    """Returns an image of scales."""
    return torch.reshape(self.adaptive_lossfun.scale(), self.image_size)
