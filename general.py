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
r"""Implements the general form of the loss.

This is the simplest way of using this loss. No parameters will be tuned
automatically, it's just a simple function that takes in parameters (likely
hand-tuned ones) and return a loss. For an adaptive loss, look at adaptive.py
or distribution.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from robust_loss_pytorch import util


def lossfun(x, alpha, scale, approximate=False, epsilon=1e-6):
  r"""Implements the general form of the loss.

  This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.

  Args:
    x: The residual for which the loss is being computed. x can have any shape,
      and alpha and scale will be broadcasted to match x's shape if necessary.
      Must be a tensor of floats.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Must be a tensor of floats with the
      same precision as `x`. Varying alpha allows
      for smooth interpolation between a number of discrete robust losses:
      alpha=-Infinity: Welsch/Leclerc Loss.
      alpha=-2: Geman-McClure loss.
      alpha=0: Cauchy/Lortentzian loss.
      alpha=1: Charbonnier/pseudo-Huber loss.
      alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha. Must be a tensor of single-precision
      floats.
    approximate: a bool, where if True, this function returns an approximate and
      faster form of the loss, as described in the appendix of the paper. This
      approximation holds well everywhere except as x and alpha approach zero.
    epsilon: A float that determines how inaccurate the "approximate" version of
      the loss will be. Larger values are less accurate but more numerically
      stable. Must be great than single-precision machine epsilon.

  Returns:
    The losses for each element of x, in the same shape and precision as x.
  """
  assert torch.is_tensor(x)
  assert torch.is_tensor(scale)
  assert torch.is_tensor(alpha)
  assert alpha.dtype == x.dtype
  assert scale.dtype == x.dtype
  assert (scale > 0).all()
  if approximate:
    # `epsilon` must be greater than single-precision machine epsilon.
    assert epsilon > np.finfo(np.float32).eps
    # Compute an approximate form of the loss which is faster, but innacurate
    # when x and alpha are near zero.
    b = torch.abs(alpha - 2) + epsilon
    d = torch.where(alpha >= 0, alpha + epsilon, alpha - epsilon)
    loss = (b / d) * (torch.pow((x / scale)**2 / b + 1., 0.5 * d) - 1.)
  else:
    # Compute the exact loss.

    # This will be used repeatedly.
    squared_scaled_x = (x / scale)**2

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = util.log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = util.expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(x)
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.max(machine_epsilon, torch.abs(alpha - 2.))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(alpha >= 0, torch.ones_like(alpha),
                             -torch.ones_like(alpha)) * torch.max(
                                 machine_epsilon, torch.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
        torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
        alpha == -float('inf'), loss_neginf,
        torch.where(
            alpha == 0, loss_zero,
            torch.where(
                alpha == 2, loss_two,
                torch.where(alpha == float('inf'), loss_posinf,
                            loss_otherwise))))

  return loss
