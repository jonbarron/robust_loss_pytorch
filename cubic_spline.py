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
"""Implements 1D cubic Hermite spline interpolation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def interpolate1d(x, values, tangents):
  r"""Perform cubic hermite spline interpolation on a 1D spline.

  The x coordinates of the spline knots are at [0 : 1 : len(values)-1].
  Queries outside of the range of the spline are computed using linear
  extrapolation. See https://en.wikipedia.org/wiki/Cubic_Hermite_spline
  for details, where "x" corresponds to `x`, "p" corresponds to `values`, and
  "m" corresponds to `tangents`.

  Args:
    x: A tensor of any size of single or double precision floats containing the
      set of values to be used for interpolation into the spline.
    values: A vector of single or double precision floats containing the value
      of each knot of the spline being interpolated into. Must be the same
      length as `tangents` and the same type as `x`.
    tangents: A vector of single or double precision floats containing the
      tangent (derivative) of each knot of the spline being interpolated into.
      Must be the same length as `values` and the same type as `x`.

  Returns:
    The result of interpolating along the spline defined by `values`, and
    `tangents`, using `x` as the query values. Will be the same length and type
    as `x`.
  """
  # if x.dtype == 'float64' or torch.as_tensor(x).dtype == torch.float64:
  #   float_dtype = torch.float64
  # else:
  #   float_dtype = torch.float32
  # x = torch.as_tensor(x, dtype=float_dtype)
  # values = torch.as_tensor(values, dtype=float_dtype)
  # tangents = torch.as_tensor(tangents, dtype=float_dtype)
  assert torch.is_tensor(x)
  assert torch.is_tensor(values)
  assert torch.is_tensor(tangents)
  float_dtype = x.dtype
  assert values.dtype == float_dtype
  assert tangents.dtype == float_dtype
  assert len(values.shape) == 1
  assert len(tangents.shape) == 1
  assert values.shape[0] == tangents.shape[0]

  x_lo = torch.floor(torch.clamp(x, torch.as_tensor(0),
                                 values.shape[0] - 2)).type(torch.int64)
  x_hi = x_lo + 1

  # Compute the relative distance between each `x` and the knot below it.
  t = x - x_lo.type(float_dtype)

  # Compute the cubic hermite expansion of `t`.
  t_sq = t**2
  t_cu = t * t_sq
  h01 = -2. * t_cu + 3. * t_sq
  h00 = 1. - h01
  h11 = t_cu - t_sq
  h10 = h11 - t_sq + t

  # Linearly extrapolate above and below the extents of the spline for all
  # values.
  value_before = tangents[0] * t + values[0]
  value_after = tangents[-1] * (t - 1.) + values[-1]

  # Cubically interpolate between the knots below and above each query point.
  neighbor_values_lo = values[x_lo]
  neighbor_values_hi = values[x_hi]
  neighbor_tangents_lo = tangents[x_lo]
  neighbor_tangents_hi = tangents[x_hi]
  value_mid = (
      neighbor_values_lo * h00 + neighbor_values_hi * h01 +
      neighbor_tangents_lo * h10 + neighbor_tangents_hi * h11)

  # Return the interpolated or extrapolated values for each query point,
  # depending on whether or not the query lies within the span of the spline.
  return torch.where(t < 0., value_before,
                     torch.where(t > 1., value_after, value_mid))
