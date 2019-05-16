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
"""Tests for general.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from robust_loss_pytorch import general


class TestLossfun:

  def setUp(self):
    np.random.seed(0)

  def _assert_all_close_according_to_type(self, a, b):
    """AssertAllClose() with tighter thresholds for float64 than float32."""
    if a.dtype == np.float32:
      np.testing.assert_allclose(a, b, rtol = 1e-6, atol=1e-6)
    elif a.dtype == np.float64:
      np.testing.assert_allclose(a, b, rtol = 1e-15, atol=1e-15)
    else:
      assert False

  def _precompute_lossfun_inputs(self, float_dtype):
    """Precompute a loss and its derivatives for random inputs and parameters.

    Generates a large number of random inputs to the loss, and random
    shape/scale parameters for the loss function at each sample, and
    computes the loss and its derivative with respect to all inputs and
    parameters, returning everything to be used to assert various properties
    in our unit tests.

    Args:
      float_dtype: The float precision to be used (np.float32 or np.float64).

    Returns:
      A tuple containing:
       (the number (int) of samples, and the length of all following arrays,
        A np.array (float_dtype) of losses for each sample,
        A np.array (float_dtype) of residuals of each sample (the loss inputs),
        A np array (float_dtype) of shape parameters of each loss,
        A np.array (float_dtype) of scale parameters of each loss,
        A np.array (float_dtype) of derivatives of each loss wrt each x,
        A np.array (float_dtype) of derivatives of each loss wrt each alpha,
        A np.array (float_dtype) of derivatives of each loss wrt each scale)

    Typical usage example:
    (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)
        = self._precompute_lossfun_inputs(np.float32)
    """
    num_samples = 100000
    # Normally distributed inputs.
    x = float_dtype(np.random.normal(size=num_samples))

    # Uniformly distributed values in (-16, 3), quantized to the nearest 0.1
    # to ensure that we hit the special cases at 0, 2.
    alpha = float_dtype(
        np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.)
    # Push the sampled alphas at the extents of the range to +/- infinity, so
    # that we probe those cases too.
    alpha[alpha == 3.] = float_dtype(float('inf'))
    alpha[alpha == -16.] = -float_dtype(float('inf'))

    # Random log-normally distributed values in approx (1e-5, 100000):
    scale = float_dtype(np.exp(np.random.normal(size=num_samples) * 4.) + 1e-5)

    # Compute the loss and its derivative with respect to all three inputs.
    var_x = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
    var_alpha = torch.autograd.Variable(torch.tensor(alpha), requires_grad=True)
    var_scale = torch.autograd.Variable(torch.tensor(scale), requires_grad=True)
    loss = general.lossfun(var_x, var_alpha, var_scale)
    sum_loss = torch.sum(loss)
    sum_loss.backward()
    d_x = var_x.grad.detach().numpy()
    d_alpha = var_alpha.grad.detach().numpy()
    d_scale = var_scale.grad.detach().numpy()
    loss = loss.detach().numpy()
    return (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)

  def _lossfun_preserves_dtype(self, float_dtype):
    """Check the loss's output has the same precision as its input."""
    n = 16
    x = float_dtype(np.random.normal(size=n))
    alpha = float_dtype(np.random.normal(size=n))
    scale = float_dtype(np.exp(np.random.normal(size=n)))
    y = general.lossfun(x, alpha, scale)
    np.testing.assert_equal(y.detach().numpy().dtype, float_dtype)

  def testLossfunPreservesDtypeSingle(self):
    self._lossfun_preserves_dtype(np.float32)

  def testLossfunPreservesDtypeDouble(self):
    self._lossfun_preserves_dtype(np.float64)

  def _derivative_is_monotonic_wrt_x(self, float_dtype):
    # Check that the loss increases monotonically with |x|.
    _, _, x, alpha, _, d_x, _, _ = self._precompute_lossfun_inputs(float_dtype)
    d_x[~np.isfinite(d_x)] = 0  # This is just to suppress a warning below.
    mask = np.isfinite(alpha) & (
        np.abs(d_x) > (100. * np.finfo(float_dtype).eps))
    np.testing.assert_equal(np.sign(d_x[mask]), np.sign(x[mask]))

  def testDerivativeIsMonotonicWrtXSingle(self):
    self._derivative_is_monotonic_wrt_x(np.float32)

  def testDerivativeIsMonotonicWrtXDouble(self):
    self._derivative_is_monotonic_wrt_x(np.float64)

  def _loss_is_near_zero_at_origin(self, float_dtype):
    # Check that the loss is near-zero when x is near-zero.
    _, loss, x, _, _, _, _, _ = self._precompute_lossfun_inputs(float_dtype)
    np.testing.assert_(np.all(np.abs(loss[np.abs(x) < 1e-5]) < 1e-5))

  def testLossIsNearZeroAtOriginSingle(self):
    self._loss_is_near_zero_at_origin(np.float32)

  def testLossIsNearZeroAtOriginDouble(self):
    self._loss_is_near_zero_at_origin(np.float64)

  def _loss_is_quadratic_near_origin(self, float_dtype):
    # Check that the loss is well-approximated by a quadratic bowl when
    # |x| < scale
    _, loss, x, _, scale, _, _, _ = self._precompute_lossfun_inputs(float_dtype)
    mask = np.abs(x) < (0.5 * scale)
    loss_quad = 0.5 * np.square(x / scale)
    np.testing.assert_allclose(loss_quad[mask], loss[mask], rtol=1e-5, atol=1e-2)

  def testLossIsQuadraticNearOriginSingle(self):
    self._loss_is_quadratic_near_origin(np.float32)

  def testLossIsQuadraticNearOriginDouble(self):
    self._loss_is_quadratic_near_origin(np.float64)

  def _loss_is_bounded_when_alpha_is_negative(self, float_dtype):
    # Assert that loss < (alpha - 2)/alpha when alpha < 0.
    _, loss, _, alpha, _, _, _, _ = self._precompute_lossfun_inputs(float_dtype)
    mask = alpha < 0.
    min_val = np.finfo(float_dtype).min
    alpha_clipped = np.maximum(min_val, alpha[mask])
    np.testing.assert_(
        np.all(loss[mask] <= ((alpha_clipped - 2.) / alpha_clipped)))

  def testLossIsBoundedWhenAlphaIsNegativeSingle(self):
    self._loss_is_bounded_when_alpha_is_negative(np.float32)

  def testLossIsBoundedWhenAlphaIsNegativeDouble(self):
    self._loss_is_bounded_when_alpha_is_negative(np.float64)

  def _derivative_is_bounded_when_alpha_is_below_2(self, float_dtype):
    # Assert that |d_x| < |x|/scale^2 when alpha <= 2.
    _, _, x, alpha, scale, d_x, _, _ = self._precompute_lossfun_inputs(
        float_dtype)
    mask = np.isfinite(alpha) & (alpha <= 2)
    np.testing.assert_(
        np.all((np.abs(d_x[mask]) <=
                ((np.abs(x[mask]) +
                  (100. * np.finfo(float_dtype).eps)) / scale[mask]**2))))

  def testDerivativeIsBoundedWhenAlphaIsBelow2Single(self):
    self._derivative_is_bounded_when_alpha_is_below_2(np.float32)

  def testDerivativeIsBoundedWhenAlphaIsBelow2Double(self):
    self._derivative_is_bounded_when_alpha_is_below_2(np.float64)

  def _derivative_is_bounded_when_alpha_is_below_1(self, float_dtype):
    # Assert that |d_x| < 1/scale when alpha <= 1.
    _, _, _, alpha, scale, d_x, _, _ = self._precompute_lossfun_inputs(
        float_dtype)
    mask = np.isfinite(alpha) & (alpha <= 1)
    np.testing.assert_(
        np.all((np.abs(d_x[mask]) <=
                ((1. + (100. * np.finfo(float_dtype).eps)) / scale[mask]))))

  def testDerivativeIsBoundedWhenAlphaIsBelow1Single(self):
    self._derivative_is_bounded_when_alpha_is_below_1(np.float32)

  def testDerivativeIsBoundedWhenAlphaIsBelow1Double(self):
    self._derivative_is_bounded_when_alpha_is_below_1(np.float64)

  def _alpha_derivative_is_positive(self, float_dtype):
    # Assert that d_loss / d_alpha > 0.
    _, _, _, alpha, _, _, d_alpha, _ = self._precompute_lossfun_inputs(
        float_dtype)
    mask = np.isfinite(alpha)
    np.testing.assert_(np.all(d_alpha[mask] > (-100. * np.finfo(float_dtype).eps)))

  def testAlphaDerivativeIsPositiveSingle(self):
    self._alpha_derivative_is_positive(np.float32)

  def testAlphaDerivativeIsPositiveDouble(self):
    self._alpha_derivative_is_positive(np.float64)

  def _scale_derivative_is_negative(self, float_dtype):
    # Assert that d_loss / d_scale < 0.
    _, _, _, alpha, _, _, _, d_scale = self._precompute_lossfun_inputs(
        float_dtype)
    mask = np.isfinite(alpha)
    np.testing.assert_(np.all(d_scale[mask] < (100. * np.finfo(float_dtype).eps)))

  def testScaleDerivativeIsNegativeSingle(self):
    self._scale_derivative_is_negative(np.float32)

  def testScaleDerivativeIsNegativeDouble(self):
    self._scale_derivative_is_negative(np.float64)

  def _loss_is_scale_invariant(self, float_dtype):
    # Check that loss(mult * x, alpha, mult * scale) == loss(x, alpha, scale)
    (num_samples, loss, x, alpha, scale, _, _,
     _) = self._precompute_lossfun_inputs(float_dtype)
    # Random log-normally distributed scalings in ~(0.2, 20)
    mult = float_dtype(
        np.maximum(0.2, np.exp(np.random.normal(size=num_samples))))

    # Compute the scaled loss.
    loss_scaled = general.lossfun(mult * x, alpha, mult * scale)
    np.testing.assert_allclose(loss, loss_scaled, atol=1e-4, rtol=1e-4)

  def testLossIsScaleInvariantSingle(self):
    self._loss_is_scale_invariant(np.float32)

  def testLossIsScaleInvariantDouble(self):
    self._loss_is_scale_invariant(np.float64)

  def _alpha_equals_negative_infinity(self, float_dtype):
    # Check that alpha == -Infinity reproduces Welsch aka Leclerc loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(-float('inf'))
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale).detach().numpy()

    # Welsch/Leclerc loss.
    loss_true = (1. - np.exp(-0.5 * (x / scale)**2))

    self._assert_all_close_according_to_type(loss, loss_true)

  def testAlphaEqualsNegativeInfinitySingle(self):
    self._alpha_equals_negative_infinity(np.float32)

  def testAlphaEqualsNegativeInfinityDouble(self):
    self._alpha_equals_negative_infinity(np.float64)

  def _alpha_equals_negative_two(self, float_dtype):
    # Check that alpha == -2 reproduces Geman-McClure loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(-2.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale).detach().numpy()

    # Geman-McClure loss.
    loss_true = 2. * (x / scale)**2 / ((x / scale)**2 + 4.)

    self._assert_all_close_according_to_type(loss, loss_true)

  def testAlphaEqualsNegativeTwoSingle(self):
    self._alpha_equals_negative_two(np.float32)

  def testAlphaEqualsNegativeTwoDouble(self):
    self._alpha_equals_negative_two(np.float64)

  def _alpha_equals_zero(self, float_dtype):
    # Check that alpha == 0 reproduces Cauchy aka Lorentzian loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(0.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale).detach().numpy()

    # Cauchy/Lorentzian loss.
    loss_true = (np.log(0.5 * (x / scale)**2 + 1.))

    self._assert_all_close_according_to_type(loss, loss_true)

  def testAlphaEqualsZeroSingle(self):
    self._alpha_equals_zero(np.float32)

  def testAlphaEqualsZeroDouble(self):
    self._alpha_equals_zero(np.float64)

  def _alpha_equals_one(self, float_dtype):
    # Check that alpha == 1 reproduces Charbonnier aka pseudo-Huber loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(1.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale).detach().numpy()

    # Charbonnier loss.
    loss_true = (np.sqrt((x / scale)**2 + 1.) - 1.)

    self._assert_all_close_according_to_type(loss, loss_true)

  def testAlphaEqualsOneSingle(self):
    self._alpha_equals_one(np.float32)

  def testAlphaEqualsOneDouble(self):
    self._alpha_equals_one(np.float64)

  def _alpha_equals_two(self, float_dtype):
    # Check that alpha == 2 reproduces L2 loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(2.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale).detach().numpy()

    # L2 Loss.
    loss_true = 0.5 * (x / scale)**2

    self._assert_all_close_according_to_type(loss, loss_true)

  def testAlphaEqualsTwoSingle(self):
    self._alpha_equals_two(np.float32)

  def testAlphaEqualsTwoDouble(self):
    self._alpha_equals_two(np.float64)

  def _alpha_equals_four(self, float_dtype):
    # Check that alpha == 4 reproduces a quartic.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(4.)
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale).detach().numpy()

    # The true loss.
    loss_true = np.square(np.square(x / scale)) / 8. + np.square(x / scale) / 2.

    self._assert_all_close_according_to_type(loss, loss_true)

  def testAlphaEqualsFourSingle(self):
    self._alpha_equals_four(np.float32)

  def testAlphaEqualsFourDouble(self):
    self._alpha_equals_four(np.float64)

  def _alpha_equals_infinity(self, float_dtype):
    # Check that alpha == Infinity takes the correct form.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(float('inf'))
    scale = float_dtype(1.7)

    # Our loss.
    loss = general.lossfun(x, alpha, scale).detach().numpy()

    # The true loss.
    loss_true = (np.exp(0.5 * np.square(x / scale)) - 1.)

    self._assert_all_close_according_to_type(loss, loss_true)

  def testAlphaEqualsInfinitySingle(self):
    self._alpha_equals_infinity(np.float32)

  def testAlphaEqualsInfinityDouble(self):
    self._alpha_equals_infinity(np.float64)

  def _approximate_loss_is_accurate(self, float_dtype):
    # Check that the approximate loss (lossfun() with epsilon=1e-6) reasonably
    # approximates the true loss (lossfun() with epsilon=0.) for a range of
    # values of alpha (skipping alpha=0, where the approximation is poor).
    x = np.arange(-10, 10, 0.1, float_dtype)
    scale = float_dtype(1.7)
    for alpha in [-4, -2, -0.2, -0.01, 0.01, 0.2, 1, 1.99, 2, 2.01, 4]:
      alpha = float_dtype(alpha)
      loss = general.lossfun(x, alpha, scale).detach().numpy()
      loss_approx = general.lossfun(x, alpha, scale, approximate=True).detach().numpy()
      np.testing.assert_allclose(
          loss, loss_approx, rtol=1e-5, atol=1e-4)

  def testApproximateLossIsAccurateSingle(self):
    self._approximate_loss_is_accurate(np.float32)

  def testApproximateLossIsAccurateDouble(self):
    self._approximate_loss_is_accurate(np.float64)

  def _loss_and_gradients_are_finite(self, float_dtype):
    # Test that the loss and its approximation both give finite losses and
    # derivatives everywhere that they should for a wide range of values.
    for approximate in [False, True]:
      num_samples = 100000

      # Normally distributed inputs.
      x = float_dtype(np.random.normal(size=num_samples))

      # Uniformly distributed values in (-16, 3), quantized to the nearest
      # 0.1 to ensure that we hit the special cases at 0, 2.
      alpha = float_dtype(
          np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.)

      # Random log-normally distributed values in approx (1e-5, 100000):
      scale = float_dtype(
          np.exp(np.random.normal(size=num_samples) * 4.) + 1e-5)

      # Compute the loss and its derivative with respect to all three inputs.
      var_x = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
      var_alpha = torch.autograd.Variable(
          torch.tensor(alpha), requires_grad=True)
      var_scale = torch.autograd.Variable(
          torch.tensor(scale), requires_grad=True)
      loss = general.lossfun(var_x, var_alpha, var_scale)
      sum_loss = torch.sum(loss)
      sum_loss.backward()
      d_x = var_x.grad.detach().numpy()
      d_alpha = var_alpha.grad.detach().numpy()
      d_scale = var_scale.grad.detach().numpy()
      loss = loss.detach().numpy()

      for v in [loss, d_x, d_alpha, d_scale]:
        np.testing.assert_(np.all(np.isfinite(v)))

  def testLossAndGradientsAreFiniteSingle(self):
    self._loss_and_gradients_are_finite(np.float32)

  def testLossAndGradientsAreFiniteDouble(self):
    self._loss_and_gradients_are_finite(np.float64)

  def _gradient_matches_finite_differences(self, float_dtype):
    # Test that the loss and its approximation both return gradients that are
    # close to the numerical gradient from finite differences, with forward
    # differencing. Returning correct gradients is Torch's job, so this is
    # just an aggressive sanity check in case some implementation detail causes
    # gradients to incorrectly go to zero due to quantization or stop_gradients
    # in some op that is used by the loss.
    for approximate in [False, True]:
      num_samples = 100000

      # Normally distributed inputs.
      x = float_dtype(np.random.normal(size=num_samples))

      # Uniformly distributed values in (-16, 3), quantized to the nearest
      # 0.1 and then shifted by 0.05 so that we avoid the special cases at
      # 0 and 2 where the analytical gradient wont match finite differences.
      alpha = float_dtype(
          np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.)

      # Random uniformy distributed values in [0.5, 1.5]
      scale = float_dtype(np.random.uniform(0.5, 1.5, num_samples))

      # Compute the loss and its derivative with respect to all three inputs.
      var_x = torch.autograd.Variable(torch.tensor(x), requires_grad=True)
      var_alpha = torch.autograd.Variable(
          torch.tensor(alpha), requires_grad=True)
      var_scale = torch.autograd.Variable(
          torch.tensor(scale), requires_grad=True)
      loss = general.lossfun(var_x, var_alpha, var_scale)
      sum_loss = torch.sum(loss)
      sum_loss.backward()
      d_x = var_x.grad.detach().numpy()
      d_alpha = var_alpha.grad.detach().numpy()
      d_scale = var_scale.grad.detach().numpy()
      loss = loss.detach().numpy()

      step_size = float_dtype(1e-3)

      # Assert that the 95th percentile of errors is <= 1e-2.
      def assert_percentile_close(v1, v2):
        np.testing.assert_(np.percentile(np.abs(v1 - v2), 95) <= 1e-2)

      n_x = (np.array(general.lossfun(x + step_size, alpha, scale)) -
             loss) / step_size
      assert_percentile_close(n_x, d_x)

      n_alpha = (np.array(general.lossfun(x, alpha + step_size, scale)) -
                 loss) / step_size
      assert_percentile_close(n_alpha, d_alpha)

      n_scale = (np.array(general.lossfun(x, alpha, scale + step_size)) -
                 loss) / step_size
      assert_percentile_close(n_scale, d_scale)

  def testGradientMatchesFiniteDifferencesSingle(self):
    self._gradient_matches_finite_differences(np.float32)

  def testGradientMatchesFiniteDifferencesDouble(self):
    self._gradient_matches_finite_differences(np.float64)


if __name__ == '__main__':
  np.testing.run_module_suite()
