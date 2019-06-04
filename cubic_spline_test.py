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
"""Tests for cubic_spline.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from robust_loss_pytorch import cubic_spline
import torch


class TestCubicSpline(parameterized.TestCase):

  def setUp(self):
    super(TestCubicSpline, self).setUp()
    np.random.seed(0)

  def _interpolate1d(self, x, values, tangents, float_dtype, device):
    """Compute interpolate1d(x, values, tangents) and its derivative.

    This is just a helper function around cubic_spline.interpolate1d() that does
    the necessary work to get derivatives.

    Args:
      x: A np.array of values to interpolate with.
      values: A np.array of knot values for the spline.
      tangents: A np.array of knot tangents for the spline.
      float_dtype: The float dtype.
      device: The device to use.

    Returns:
      A tuple containing:
       (An np.array of interpolated values,
        A np.array of derivatives of interpolated values wrt `x`)

    Typical usage example:
      y, dy_dx = self._interpolate1d(x, values, tangents)
    """
    x = torch.tensor(float_dtype(x), device=device)
    values = torch.tensor(float_dtype(values)).to(x)
    tangents = torch.tensor(float_dtype(tangents)).to(x)
    var_x = torch.autograd.Variable(x, requires_grad=True)
    y = cubic_spline.interpolate1d(var_x, values, tangents)
    sum_y = torch.sum(y)
    sum_y.backward()
    dy_dx = var_x.grad.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    return y, dy_dx

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testInterpolationPreservesDtype(self, float_dtype, device):
    """Check that interpolating at a knot produces the value at that knot."""
    n = 16
    x = float_dtype(np.random.normal(size=n))
    values = float_dtype(np.random.normal(size=n))
    tangents = float_dtype(np.random.normal(size=n))
    y = self._interpolate1d(x, values, tangents, float_dtype, device)[0]
    np.testing.assert_equal(y.dtype, float_dtype)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testInterpolationReproducesValuesAtKnots(self, float_dtype, device):
    """Check that interpolating at a knot produces the value at that knot."""
    n = 32768
    x = np.arange(n)
    values = np.random.normal(size=n)
    tangents = np.random.normal(size=n)
    y = self._interpolate1d(x, values, tangents, float_dtype, device)[0]
    np.testing.assert_allclose(y, values)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testInterpolationReproducesTangentsAtKnots(self, float_dtype, device):
    """Check that the derivative at a knot produces the tangent at that knot."""
    n = 32768
    x = np.arange(n)
    values = np.random.normal(size=n)
    tangents = np.random.normal(size=n)
    _, dy_dx = self._interpolate1d(x, values, tangents, float_dtype, device)
    np.testing.assert_allclose(dy_dx, tangents, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testZeroTangentMidpointValuesAndDerivativesAreCorrect(
      self, float_dtype, device):
    """Check that splines with zero tangents behave correctly at midpoints.

    Make a spline whose tangents are all zeros, and then verify that
    midpoints between each pair of knots have the mean value of their adjacent
    knots, and have a derivative that is 1.5x the difference between their
    adjacent knots.

    Args:
      float_dtype: the dtype of the floats to be tested.
      device: The device to use.
    """
    # Make a spline with random values and all-zero tangents.
    n = 32768
    values = float_dtype(np.random.normal(size=n))
    tangents = np.zeros_like(values)

    # Query n-1 points placed exactly in between each pair of knots.
    x = float_dtype(np.arange(n - 1)) + float_dtype(0.5)

    # Get the interpolated values and derivatives.
    y, dy_dx = self._interpolate1d(x, values, tangents, float_dtype, device)

    # Check that the interpolated values of all queries lies at the midpoint of
    # its surrounding knot values.
    y_true = (values[0:-1] + values[1:]) / 2.
    np.testing.assert_allclose(y, y_true)

    # Check that the derivative of all interpolated values is (fun fact!) 1.5x
    # the numerical difference between adjacent knot values.
    dy_dx_true = 1.5 * (values[1:] - values[0:-1])
    np.testing.assert_allclose(dy_dx, dy_dx_true, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testZeroTangentIntermediateValuesAndDerivativesDoNotOvershoot(
      self, float_dtype, device):
    """Check that splines with zero tangents behave correctly between knots.

    Make a spline whose tangents are all zeros, and then verify that points
    between each knot lie in between the knot values, and have derivatives
    are between 0 and 1.5x the numerical difference between knot values
    (mathematically, 1.5x is the max derivative if the tangents are zero).

    Args:
      float_dtype: the dtype of the floats to be tested.
      device: The device to use.
    """

    # Make a spline with all-zero tangents and random values.
    n = 32768
    values = float_dtype(np.random.normal(size=n))
    tangents = np.zeros_like(values)

    # Query n-1 points placed somewhere randomly in between all adjacent knots.
    x = np.arange(
        n - 1, dtype=float_dtype) + float_dtype(np.random.uniform(size=n - 1))

    # Get the interpolated values and derivatives.
    y, dy_dx = self._interpolate1d(x, values, tangents, float_dtype, device)

    # Check that the interpolated values of all queries lies between its
    # surrounding knot values.
    np.testing.assert_(
        np.all(((values[0:-1] <= y) & (y <= values[1:]))
               | ((values[0:-1] >= y) & (y >= values[1:]))))

    # Check that all derivatives of interpolated values are between 0 and 1.5x
    # the numerical difference between adjacent knot values.
    max_dy_dx = (1.5 + 1e-3) * (values[1:] - values[0:-1])
    np.testing.assert_(
        np.all(((0 <= dy_dx) & (dy_dx <= max_dy_dx))
               | ((0 >= dy_dx) & (dy_dx >= max_dy_dx))))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLinearRampsReproduceCorrectly(self, float_dtype, device):
    """Check that interpolating a ramp reproduces a ramp.

    Generate linear ramps, render them into splines, and then interpolate and
    extrapolate the splines and verify that they reproduce the ramp.

    Args:
      float_dtype: the dtype of the floats to be tested.
      device: The device to use.
    """
    n = 256
    # Generate queries inside and outside the support of the spline.
    x = float_dtype((np.random.uniform(size=1024) * 2 - 0.5) * (n - 1))
    idx = np.arange(n, dtype=float_dtype)
    for _ in range(8):
      slope = np.random.normal()
      bias = np.random.normal()
      values = slope * idx + bias
      tangents = np.ones_like(values) * slope
      y = self._interpolate1d(x, values, tangents, float_dtype, device)[0]
      y_true = slope * x + bias
      np.testing.assert_allclose(y, y_true, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testExtrapolationIsLinear(self, float_dtype, device):
    """Check that extrapolation is linear with respect to the endpoint knots.

    Generate random splines and query them outside of the support of the
    spline, and veify that extrapolation is linear with respect to the
    endpoint knots.

    Args:
      float_dtype: the dtype of the floats to be tested.
      device: The device to use.
    """
    n = 256
    # Generate queries above and below the support of the spline.
    x_below = float_dtype(-(np.random.uniform(size=1024)) * (n - 1))
    x_above = float_dtype((np.random.uniform(size=1024) + 1.) * (n - 1))
    for _ in range(8):
      values = float_dtype(np.random.normal(size=n))
      tangents = float_dtype(np.random.normal(size=n))

      # Query the spline below its support and check that it's a linear ramp
      # with the slope and bias of the beginning of the spline.
      y_below = self._interpolate1d(x_below, values, tangents, float_dtype,
                                    device)[0]
      y_below_true = tangents[0] * x_below + values[0]
      np.testing.assert_allclose(y_below, y_below_true)

      # Query the spline above its support and check that it's a linear ramp
      # with the slope and bias of the end of the spline.
      y_above = self._interpolate1d(x_above, values, tangents, float_dtype,
                                    device)[0]
      y_above_true = tangents[-1] * (x_above - (n - 1)) + values[-1]
      np.testing.assert_allclose(y_above, y_above_true)


if __name__ == '__main__':
  np.testing.run_module_suite()
