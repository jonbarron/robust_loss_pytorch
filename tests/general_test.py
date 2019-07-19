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

from absl.testing import parameterized
import numpy as np
import torch
from robust_loss_pytorch import general


class TestGeneral(parameterized.TestCase):

  def setUp(self):
    super(TestGeneral, self).setUp()
    np.random.seed(0)

  def _assert_all_close_according_to_type(self, a, b):
    """AssertAllClose() with tighter thresholds for float64 than float32."""
    if a.dtype == np.float32:
      np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    elif a.dtype == np.float64:
      np.testing.assert_allclose(a, b, rtol=1e-15, atol=1e-15)
    else:
      assert False

  def _precompute_lossfun_inputs(self, float_dtype, device):
    """Precompute a loss and its derivatives for random inputs and parameters.

    Generates a large number of random inputs to the loss, and random
    shape/scale parameters for the loss function at each sample, and
    computes the loss and its derivative with respect to all inputs and
    parameters, returning everything to be used to assert various properties
    in our unit tests.

    Args:
      float_dtype: The float precision to be used (np.float32 or np.float64).
      device: The device to run on.

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
        = self._precompute_lossfun_inputs(np.float32, 'cpu')
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
    var_x = torch.autograd.Variable(
        torch.tensor(x, device=device), requires_grad=True)
    var_alpha = torch.autograd.Variable(
        torch.tensor(alpha, device=device), requires_grad=True)
    var_scale = torch.autograd.Variable(
        torch.tensor(scale, device=device), requires_grad=True)
    loss = general.lossfun(var_x, var_alpha, var_scale)
    sum_loss = torch.sum(loss)
    sum_loss.backward()
    d_x = var_x.grad.cpu().detach().numpy()
    d_alpha = var_alpha.grad.cpu().detach().numpy()
    d_scale = var_scale.grad.cpu().detach().numpy()
    loss = loss.cpu().detach().numpy()
    return (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLossfunPreservesDtype(self, float_dtype, device):
    """Check the loss's output has the same precision as its input."""
    n = 16
    x = torch.tensor(float_dtype(np.random.normal(size=n)), device=device)
    alpha = torch.tensor(float_dtype(np.random.normal(size=n)), device=device)
    scale = torch.tensor(
        float_dtype(np.exp(np.random.normal(size=n))), device=device)
    y = general.lossfun(x, alpha, scale)
    np.testing.assert_equal(y.cpu().detach().numpy().dtype, float_dtype)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLossfunPreservesDevice(self, float_dtype, device):
    """Check the loss's output has the same precision as its input."""
    n = 16
    x = torch.tensor(float_dtype(np.random.normal(size=n)), device=device)
    alpha = torch.tensor(float_dtype(np.random.normal(size=n)), device=device)
    scale = torch.tensor(
        float_dtype(np.exp(np.random.normal(size=n))), device=device)
    y = general.lossfun(x, alpha, scale)
    np.testing.assert_equal(y.device.type, device)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testDerivativeIsMonotonicWrtX(self, float_dtype, device):
    # Check that the loss increases monotonically with |x|.
    _, _, x, alpha, _, d_x, _, _ = self._precompute_lossfun_inputs(
        float_dtype, device)
    d_x[~np.isfinite(d_x)] = 0  # This is just to suppress a warning below.
    mask = np.isfinite(alpha) & (
        np.abs(d_x) > (100. * np.finfo(float_dtype).eps))
    np.testing.assert_equal(np.sign(d_x[mask]), np.sign(x[mask]))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLossIsNearZeroAtOrigin(self, float_dtype, device):
    # Check that the loss is near-zero when x is near-zero.
    _, loss, x, _, _, _, _, _ = self._precompute_lossfun_inputs(
        float_dtype, device)
    np.testing.assert_(np.all(np.abs(loss[np.abs(x) < 1e-5]) < 1e-5))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLossIsQuadraticNearOrigin(self, float_dtype, device):
    # Check that the loss is well-approximated by a quadratic when |x| < scale
    _, loss, x, _, scale, _, _, _ = self._precompute_lossfun_inputs(
        float_dtype, device)
    mask = np.abs(x) < (0.5 * scale)
    loss_quad = 0.5 * np.square(x / scale)
    np.testing.assert_allclose(
        loss_quad[mask], loss[mask], rtol=1e-5, atol=1e-2)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLossIsBoundedWhenAlphaIsNegative(self, float_dtype, device):
    # Assert that loss < (alpha - 2)/alpha when alpha < 0.
    _, loss, _, alpha, _, _, _, _ = self._precompute_lossfun_inputs(
        float_dtype, device)
    mask = alpha < 0.
    min_val = np.finfo(float_dtype).min
    alpha_clipped = np.maximum(min_val, alpha[mask])
    np.testing.assert_(
        np.all(loss[mask] <= ((alpha_clipped - 2.) / alpha_clipped)))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testDerivativeIsBoundedWhenAlphaIsBelow2(self, float_dtype, device):
    # Assert that |d_x| < |x|/scale^2 when alpha <= 2.
    _, _, x, alpha, scale, d_x, _, _ = self._precompute_lossfun_inputs(
        float_dtype, device)
    mask = np.isfinite(alpha) & (alpha <= 2)
    np.testing.assert_(
        np.all((np.abs(d_x[mask]) <=
                ((np.abs(x[mask]) +
                  (100. * np.finfo(float_dtype).eps)) / scale[mask]**2))))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testDerivativeIsBoundedWhenAlphaIsBelow1(self, float_dtype, device):
    # Assert that |d_x| < 1/scale when alpha <= 1.
    _, _, _, alpha, scale, d_x, _, _ = self._precompute_lossfun_inputs(
        float_dtype, device)
    mask = np.isfinite(alpha) & (alpha <= 1)
    np.testing.assert_(
        np.all((np.abs(d_x[mask]) <=
                ((1. + (100. * np.finfo(float_dtype).eps)) / scale[mask]))))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testAlphaDerivativeIsPositive(self, float_dtype, device):
    # Assert that d_loss / d_alpha > 0.
    _, _, _, alpha, _, _, d_alpha, _ = self._precompute_lossfun_inputs(
        float_dtype, device)
    mask = np.isfinite(alpha)
    np.testing.assert_(
        np.all(d_alpha[mask] > (-300. * np.finfo(float_dtype).eps)))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testScaleDerivativeIsNegative(self, float_dtype, device):
    # Assert that d_loss / d_scale < 0.
    _, _, _, alpha, _, _, _, d_scale = self._precompute_lossfun_inputs(
        float_dtype, device)
    mask = np.isfinite(alpha)
    np.testing.assert_(
        np.all(d_scale[mask] < (100. * np.finfo(float_dtype).eps)))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLossIsScaleInvariant(self, float_dtype, device):
    # Check that loss(mult * x, alpha, mult * scale) == loss(x, alpha, scale)
    (num_samples, loss, x, alpha, scale, _, _,
     _) = self._precompute_lossfun_inputs(float_dtype, device)
    # Random log-normally distributed scalings in ~(0.2, 20)
    mult = float_dtype(
        np.maximum(0.2, np.exp(np.random.normal(size=num_samples))))

    x = torch.tensor(np.array(mult * x, dtype=float_dtype), device=device)
    alpha = torch.tensor(np.array(alpha, dtype=float_dtype), device=device)
    scale = torch.tensor(
        np.array(mult * scale, dtype=float_dtype), device=device)
    # Compute the scaled loss.
    loss_scaled = general.lossfun(x, alpha, scale).cpu().detach()
    np.testing.assert_allclose(loss, loss_scaled, atol=1e-4, rtol=1e-4)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testAlphaEqualsNegativeInfinity(self, float_dtype, device):
    # Check that alpha == -Infinity reproduces Welsch aka Leclerc loss.
    x = float_dtype(np.arange(-20, 20, 0.1))
    alpha = float_dtype(np.array([-float('inf')]))
    scale = float_dtype(np.array([2.]))

    # Our loss.
    x_t = torch.tensor(x, device=device)
    alpha_t = torch.tensor(alpha).to(x_t)
    scale_t = torch.tensor(scale).to(x_t)
    loss = general.lossfun(x_t, alpha_t, scale_t).cpu().detach().numpy()

    # Welsch/Leclerc loss.
    loss_true = (1. - np.exp(-0.5 * (x / scale)**2))

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testAlphaEqualsNegativeTwo(self, float_dtype, device):
    # Check that alpha == -2 reproduces Geman-McClure loss.
    x = float_dtype(np.arange(-20, 20, 0.1))
    alpha = float_dtype(np.array(-2.))
    scale = float_dtype(np.array(2.))

    # Our loss.
    x_t = torch.tensor(x, device=device)
    alpha_t = torch.tensor(alpha).to(x_t)
    scale_t = torch.tensor(scale).to(x_t)
    loss = general.lossfun(x_t, alpha_t, scale_t).cpu().detach().numpy()

    # Geman-McClure loss.
    loss_true = 2. * (x / scale)**2 / ((x / scale)**2 + 4.)

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testAlphaEqualsZero(self, float_dtype, device):
    # Check that alpha == 0 reproduces Cauchy aka Lorentzian loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(0.)
    scale = float_dtype(2.)

    # Our loss.
    x_t = torch.tensor(x, device=device)
    alpha_t = torch.tensor(alpha).to(x_t)
    scale_t = torch.tensor(scale).to(x_t)
    loss = general.lossfun(x_t, alpha_t, scale_t).cpu().detach().numpy()

    # Cauchy/Lorentzian loss.
    loss_true = (np.log(0.5 * (x / scale)**2 + 1.))

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testAlphaEqualsOne(self, float_dtype, device):
    # Check that alpha == 1 reproduces Charbonnier aka pseudo-Huber loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(1.)
    scale = float_dtype(2.)

    # Our loss.
    x_t = torch.tensor(x, device=device)
    alpha_t = torch.tensor(alpha).to(x_t)
    scale_t = torch.tensor(scale).to(x_t)
    loss = general.lossfun(x_t, alpha_t, scale_t).cpu().detach().numpy()

    # Charbonnier loss.
    loss_true = (np.sqrt((x / scale)**2 + 1.) - 1.)

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testAlphaEqualsTwo(self, float_dtype, device):
    # Check that alpha == 2 reproduces L2 loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(2.)
    scale = float_dtype(2.)

    # Our loss.
    x_t = torch.tensor(x, device=device)
    alpha_t = torch.tensor(alpha).to(x_t)
    scale_t = torch.tensor(scale).to(x_t)
    loss = general.lossfun(x_t, alpha_t, scale_t).cpu().detach().numpy()

    # L2 Loss.
    loss_true = 0.5 * (x / scale)**2

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testAlphaEqualsFour(self, float_dtype, device):
    # Check that alpha == 4 reproduces a quartic.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(4.)
    scale = float_dtype(2.)

    # Our loss.
    x_t = torch.tensor(x, device=device)
    alpha_t = torch.tensor(alpha).to(x_t)
    scale_t = torch.tensor(scale).to(x_t)
    loss = general.lossfun(x_t, alpha_t, scale_t).cpu().detach().numpy()

    # The true loss.
    loss_true = np.square(np.square(x / scale)) / 8. + np.square(x / scale) / 2.

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testAlphaEqualsInfinity(self, float_dtype, device):
    # Check that alpha == Infinity takes the correct form.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(float('inf'))
    scale = float_dtype(2.)

    # Our loss.
    x_t = torch.tensor(x, device=device)
    alpha_t = torch.tensor(alpha).to(x_t)
    scale_t = torch.tensor(scale).to(x_t)
    loss = general.lossfun(x_t, alpha_t, scale_t).cpu().detach().numpy()

    # The true loss.
    loss_true = (np.exp(0.5 * np.square(x / scale)) - 1.)

    self._assert_all_close_according_to_type(loss, loss_true)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testApproximateLossIsAccurate(self, float_dtype, device):
    # Check that the approximate loss (lossfun() with epsilon=1e-6) reasonably
    # approximates the true loss (lossfun() with epsilon=0.) for a range of
    # values of alpha (skipping alpha=0, where the approximation is poor).
    x = np.arange(-10, 10, 0.1, float_dtype)
    scale = float_dtype(1.7)
    for alpha in [-4, -2, -0.2, -0.01, 0.01, 0.2, 1, 1.99, 2, 2.01, 4]:
      alpha = float_dtype(alpha)
      x_t = torch.tensor(x, device=device)
      alpha_t = torch.tensor(alpha).to(x_t)
      scale_t = torch.tensor(scale).to(x_t)
      loss = general.lossfun(x_t, alpha_t, scale_t).cpu().detach().numpy()
      loss_approx = general.lossfun(
          x_t, alpha_t, scale_t, approximate=True).cpu().detach().numpy()
      np.testing.assert_allclose(loss, loss_approx, rtol=1e-5, atol=1e-4)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLossAndGradientsAreFinite(self, float_dtype, device):
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
      var_x = torch.autograd.Variable(
          torch.tensor(x, device=device), requires_grad=True)
      var_alpha = torch.autograd.Variable(
          torch.tensor(alpha, device=device), requires_grad=True)
      var_scale = torch.autograd.Variable(
          torch.tensor(scale, device=device), requires_grad=True)
      loss = general.lossfun(
          var_x, var_alpha, var_scale, approximate=approximate)
      sum_loss = torch.sum(loss)
      sum_loss.backward()
      d_x = var_x.grad.cpu().detach().numpy()
      d_alpha = var_alpha.grad.cpu().detach().numpy()
      d_scale = var_scale.grad.cpu().detach().numpy()
      loss = loss.cpu().detach().numpy()

      for v in [loss, d_x, d_alpha, d_scale]:
        np.testing.assert_(np.all(np.isfinite(v)))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testGradientMatchesFiniteDifferences(self, float_dtype, device):
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
      var_x = torch.autograd.Variable(
          torch.tensor(x, device=device), requires_grad=True)
      var_alpha = torch.autograd.Variable(
          torch.tensor(alpha, device=device), requires_grad=True)
      var_scale = torch.autograd.Variable(
          torch.tensor(scale, device=device), requires_grad=True)
      loss = general.lossfun(
          var_x, var_alpha, var_scale, approximate=approximate)
      sum_loss = torch.sum(loss)
      sum_loss.backward()
      d_x = var_x.grad.cpu().detach().numpy()
      d_alpha = var_alpha.grad.cpu().detach().numpy()
      d_scale = var_scale.grad.cpu().detach().numpy()
      loss = loss.cpu().detach().numpy()

      step_size = float_dtype(1e-3)

      # Assert that the 95th percentile of errors is <= 1e-2.
      def assert_percentile_close(v1, v2):
        np.testing.assert_(np.percentile(np.abs(v1 - v2), 95) <= 1e-2)

      def loss_helper(x, a, c):
        x = torch.tensor(x, device=device)
        a = torch.tensor(a).to(x)
        c = torch.tensor(c).to(x)
        return general.lossfun(x, a, c).cpu().detach().numpy()

      n_x = (loss_helper(x + step_size, alpha, scale) - loss) / step_size
      assert_percentile_close(n_x, d_x)

      n_alpha = (loss_helper(x, alpha + step_size, scale) - loss) / step_size
      assert_percentile_close(n_alpha, d_alpha)

      n_scale = (loss_helper(x, alpha, scale + step_size) - loss) / step_size
      assert_percentile_close(n_scale, d_scale)


if __name__ == '__main__':
  np.testing.run_module_suite()
