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
"""Tests for adaptive.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import scipy.stats
import torch
from torch.autograd import Variable
from robust_loss_pytorch import adaptive
from robust_loss_pytorch import util
from robust_loss_pytorch import wavelet


def _get_device(device_string):
    """Returns a `torch.device`

    Args:
      device_string: 'cpu' or 'cuda'

    Returns:
      `torch.device`
    """
    if device_string.lower() == 'cpu':
      return torch.device('cpu')
    if device_string.lower() == 'cuda':
      if torch.cuda.device_count() == 0:
        print("Warning: There's no GPU available on this machine!")
        return None
      return torch.device('cuda:0')
    raise Exception(
      '{} is not a valid option. Choose `cpu` or `cuda`.'.format(device_string))


def _generate_pixel_toy_image_data(image_width, num_samples, _):
  """Generates pixel data for _test_fitting_toy_image_data_is_correct().

  Constructs a "mean" image in RGB pixel space (parametrized by `image_width`)
  and draws `num_samples` samples from a normal distribution using that mean,
  and returns those samples and their empirical mean as reference.

  Args:
    image_width: The width and height in pixels of the images being produced.
    num_samples: The number of samples to generate.
    _: Dummy argument so that this function's interface matches
      _generate_wavelet_toy_image_data()

  Returns:
    A tuple of (samples, reference, color_space, representation), where
    samples = A set of sampled images of size
      (`num_samples`, `image_width`, `image_width`, 3)
    reference = The empirical mean of `samples` of size
      (`image_width`, `image_width`, 3).
    color_space = 'RGB'
    representation = 'PIXEL'
  """
  color_space = 'RGB'
  representation = 'PIXEL'
  mu = np.random.uniform(size=(image_width, image_width, 3))
  samples = np.random.normal(
      loc=np.tile(mu[np.newaxis], [num_samples, 1, 1, 1]))
  reference = np.mean(samples, 0)
  return samples, reference, color_space, representation


def _generate_wavelet_toy_image_data(image_width, num_samples,
                                     wavelet_num_levels):
  """Generates wavelet data for testFittingImageDataIsCorrect().

  Constructs a "mean" image in the YUV wavelet domain (parametrized by
  `image_width`, and `wavelet_num_levels`) and draws `num_samples` samples
  from a normal distribution using that mean, and returns RGB images
  corresponding to those samples and to the mean (computed in the
  specified latent space) of those samples.

  Args:
    image_width: The width and height in pixels of the images being produced.
    num_samples: The number of samples to generate.
    wavelet_num_levels: The number of levels in the wavelet decompositions of
      the generated images.

  Returns:
    A tuple of (samples, reference, color_space, representation), where
    samples = A set of sampled images of size
      (`num_samples`, `image_width`, `image_width`, 3)
    reference = The empirical mean of `samples` (computed in YUV Wavelet space
      but returned as an RGB image) of size (`image_width`, `image_width`, 3).
    color_space = 'YUV'
    representation = 'CDF9/7'
  """
  color_space = 'YUV'
  representation = 'CDF9/7'
  samples = []
  reference = []
  for level in range(wavelet_num_levels):
    samples.append([])
    reference.append([])
    w = image_width // 2**(level + 1)
    scaling = 2**level
    for _ in range(3):
      # Construct the ground-truth pixel band mean.
      mu = scaling * np.random.uniform(size=(3, w, w))
      # Draw samples from the ground-truth mean.
      band_samples = np.random.normal(
          loc=np.tile(mu[np.newaxis], [num_samples, 1, 1, 1]))
      # Take the empirical mean of the samples as a reference.
      band_reference = np.mean(band_samples, 0)
      samples[-1].append(np.reshape(band_samples, [-1, w, w]))
      reference[-1].append(band_reference)
  # Handle the residual band.
  mu = scaling * np.random.uniform(size=(3, w, w))
  band_samples = np.random.normal(
      loc=np.tile(mu[np.newaxis], [num_samples, 1, 1, 1]))
  band_reference = np.mean(band_samples, 0)
  samples.append(np.reshape(band_samples, [-1, w, w]))
  reference.append(band_reference)
  # Collapse and reshape wavelets to be ({_,} width, height, 3).
  samples = wavelet.collapse(samples, representation)
  reference = wavelet.collapse(reference, representation)
  samples = np.transpose(
      np.reshape(samples, [num_samples, 3, image_width, image_width]),
      [0, 2, 3, 1])
  reference = np.transpose(reference, [1, 2, 0])
  # Convert into RGB space.
  samples = util.syuv_to_rgb(samples)
  reference = util.syuv_to_rgb(reference)
  samples = samples.detach().numpy()
  reference = reference.detach().numpy()
  return samples, reference, color_space, representation


class TestAdaptive(parameterized.TestCase):

  def setUp(self):
    super(TestAdaptive, self).setUp()
    np.random.seed(0)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testInitialAlphaAndScaleAreCorrect(self, float_dtype, device_string):
    """Tests that `alpha` and `scale` are initialized as expected."""
    device = _get_device(device_string)
    for i in range(8):
      # Generate random ranges for alpha and scale.
      alpha_lo = float_dtype(np.random.uniform())
      alpha_hi = float_dtype(np.random.uniform() + 1.)
      # Half of the time pick a random initialization for alpha, the other half
      # use the default value.
      if i % 2 == 0:
        alpha_init = float_dtype(alpha_lo + np.random.uniform() *
                                 (alpha_hi - alpha_lo))
        true_alpha_init = alpha_init
      else:
        alpha_init = None
        true_alpha_init = (alpha_lo + alpha_hi) / 2.
      scale_init = float_dtype(np.random.uniform() + 0.5)
      scale_lo = float_dtype(np.random.uniform() * 0.1)
      adaptive_lossfun = adaptive.AdaptiveLossFunction(
          10,
          float_dtype,
          device,
          alpha_lo=alpha_lo,
          alpha_hi=alpha_hi,
          alpha_init=alpha_init,
          scale_lo=scale_lo,
          scale_init=scale_init)
      alpha = adaptive_lossfun.alpha().cpu().detach().numpy()
      scale = adaptive_lossfun.scale().cpu().detach().numpy()
      np.testing.assert_allclose(alpha, true_alpha_init * np.ones_like(alpha))
      np.testing.assert_allclose(scale, scale_init * np.ones_like(scale))

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testFixedAlphaAndScaleAreCorrect(self, float_dtype, device_string):
    """Tests that fixed alphas and scales do not change during optimization)."""
    device = _get_device(device_string)
    alpha_lo = np.random.uniform() * 2.
    alpha_hi = alpha_lo
    scale_init = float_dtype(np.random.uniform() + 0.5)
    scale_lo = scale_init
    num_dims = 10
    # We must construct some variable for TF to attempt to optimize.
    adaptive_lossfun = adaptive.AdaptiveLossFunction(
        num_dims,
        float_dtype,
        device,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
        scale_lo=scale_lo,
        scale_init=scale_init)

    params = torch.nn.ParameterList(adaptive_lossfun.parameters())
    assert len(params) == 0
    alpha = adaptive_lossfun.alpha().cpu().detach()
    scale = adaptive_lossfun.scale().cpu().detach()
    alpha_init = (alpha_lo + alpha_hi) / 2.
    np.testing.assert_allclose(alpha, alpha_init * np.ones_like(alpha))
    np.testing.assert_allclose(scale, scale_init * np.ones_like(alpha))

  def _sample_cauchy_ppf(self, num_samples):
    """Draws ``num_samples'' samples from a Cauchy distribution.

    Because actual sampling is expensive and requires many samples to converge,
    here we sample by drawing `num_samples` evenly-spaced values in [0, 1]
    and then interpolate into the inverse CDF (aka PPF) of a Cauchy
    distribution. This produces "samples" where maximum-likelihood estimation
    likely recovers the true distribution even if `num_samples` is small.

    Args:
      num_samples: The number of samples to draw.

    Returns:
      A numpy array containing `num_samples` evenly-spaced "samples" from a
      zero-mean Cauchy distribution whose scale matches our distribution/loss
      when our scale = 1.
    """
    spacing = 1. / num_samples
    p = np.arange(0., 1., spacing) + spacing / 2.
    return scipy.stats.cauchy(0., np.sqrt(2.)).ppf(p)

  def _sample_normal_ppf(self, num_samples):
    """Draws ``num_samples'' samples from a Normal distribution.

    Because actual sampling is expensive and requires many samples to converge,
    here we sample by drawing `num_samples` evenly-spaced values in [0, 1]
    and then interpolate into the inverse CDF (aka PPF) of a Normal
    distribution. This produces "samples" where maximum-likelihood estimation
    likely recovers the true distribution even if `num_samples` is small.

    Args:
      num_samples: The number of samples to draw.

    Returns:
      A numpy array containing `num_samples` evenly-spaced "samples" from a
      zero-mean unit-scale Normal distribution.
    """
    spacing = 1. / num_samples
    p = np.arange(0., 1., spacing) + spacing / 2.
    return scipy.stats.norm(0., 1.).ppf(p)

  def _sample_nd_mixed_data(self, n, m, float_dtype):
    """`n` Samples from `m` scaled+shifted Cauchy and Normal distributions."""
    samples0 = self._sample_cauchy_ppf(n)
    samples2 = self._sample_normal_ppf(n)
    mu = np.random.normal(size=m)
    alpha = (np.random.uniform(size=m) > 0.5) * 2
    scale = np.exp(np.clip(np.random.normal(size=m), -3., 3.))
    samples = (
        np.tile(samples0[:, np.newaxis], [1, m]) *
        (alpha[np.newaxis, :] == 0.) +
        np.tile(samples2[:, np.newaxis], [1, m]) *
        (alpha[np.newaxis, :] == 2.)) * scale[np.newaxis, :] + mu[np.newaxis, :]
    return [float_dtype(x) for x in [samples, mu, alpha, scale]]

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testFittingToyNdMixedDataIsCorrect(self, float_dtype, device_string):
    """Tests that minimizing the adaptive loss recovers the true model.

    Here we generate a 2D array of samples drawn from a mix of scaled and
    shifted Cauchy and Normal distributions. We then minimize our loss with
    respect to the mean, scale, and shape of each distribution, and check that
    after minimization the shape parameter is near-zero for the Cauchy data and
    near 2 for the Normal data, and that the estimated means and scales are
    accurate.

    Args:
      float_dtype: The type (np.float32 or np.float64) of data to test.
      device: The device to run on.
    """
    device = _get_device(device_string)
    num_dims = 8
    samples, mu_true, alpha_true, scale_true = self._sample_nd_mixed_data(
        100, num_dims, float_dtype)
    mu = Variable(
        torch.tensor(
            np.zeros(samples.shape[1], dtype=float_dtype), device=device),
        requires_grad=True)

    adaptive_lossfun = adaptive.AdaptiveLossFunction(num_dims, float_dtype,
                                                     device)
    params = torch.nn.ParameterList(adaptive_lossfun.parameters())
    optimizer = torch.optim.Adam([p for p in params] + [mu], lr=0.1)
    for _ in range(1000):
      optimizer.zero_grad()
      x = torch.as_tensor(samples, device=device) - mu[np.newaxis, :]
      loss = torch.sum(adaptive_lossfun.lossfun(x))
      loss.backward(retain_graph=True)
      optimizer.step()

    mu = mu.cpu().detach().numpy()
    alpha = adaptive_lossfun.alpha()[0, :].cpu().detach().numpy()
    scale = adaptive_lossfun.scale()[0, :].cpu().detach().numpy()
    for a, b in [(alpha, alpha_true), (scale, scale_true), (mu, mu_true)]:
      np.testing.assert_allclose(a, b * np.ones_like(a), rtol=0.1, atol=0.1)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testFittingToyNdMixedDataIsCorrectStudentsT(self, float_dtype, device_string):
    """Tests that minimizing the Student's T loss recovers the true model.

    Here we generate a 2D array of samples drawn from a mix of scaled and
    shifted Cauchy and Normal distributions. We then minimize our loss with
    respect to the mean, scale, and shape of each distribution, and check that
    after minimization the log-df parameter is near-zero for the Cauchy data and
    very large for the Normal data, and that the estimated means and scales are
    accurate.

    Args:
      float_dtype: The type (np.float32 or np.float64) of data to test.
      device: The device to run on.
    """
    device = _get_device(device_string)
    num_dims = 8
    samples, mu_true, alpha_true, scale_true = self._sample_nd_mixed_data(
        100, num_dims, float_dtype)
    mu = Variable(
        torch.tensor(
            np.zeros(samples.shape[1], dtype=float_dtype), device=device),
        requires_grad=True)

    students_t_lossfun = adaptive.StudentsTLossFunction(num_dims, float_dtype,
                                                        device)
    params = torch.nn.ParameterList(students_t_lossfun.parameters())
    optimizer = torch.optim.Adam([params[0], params[1], mu], lr=0.1)
    for _ in range(1000):
      optimizer.zero_grad()
      x = torch.as_tensor(samples, device=device) - mu[np.newaxis, :]
      loss = torch.sum(students_t_lossfun.lossfun(x))
      loss.backward(retain_graph=True)
      optimizer.step()

    mu = mu.cpu().detach().numpy()
    log_df = students_t_lossfun.log_df[0, :].cpu().detach().numpy()
    scale = students_t_lossfun.scale()[0, :].cpu().detach().numpy()

    for ldf, a_true in zip(log_df, alpha_true):
      if a_true == 0:
        np.testing.assert_allclose(ldf, 0., rtol=0.1, atol=0.1)
      elif a_true == 2:
        np.testing.assert_(np.all(ldf > 4))
    scale /= np.sqrt(2. - (alpha_true / 2.))
    for a, b in [(scale, scale_true), (mu, mu_true)]:
      np.testing.assert_allclose(a, b * np.ones_like(a), rtol=0.1, atol=0.1)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testLossfunPreservesDtype(self, float_dtype, device_string):
    """Checks the loss's outputs have the same precisions as its input."""
    device = _get_device(device_string)
    num_dims = 8
    samples, _, _, _ = self._sample_nd_mixed_data(100, num_dims, float_dtype)
    adaptive_lossfun = adaptive.AdaptiveLossFunction(num_dims, float_dtype,
                                                     device)
    loss = adaptive_lossfun.lossfun(torch.tensor(
        samples, device=device)).cpu().detach().numpy()
    alpha = adaptive_lossfun.alpha().cpu().detach().numpy()
    scale = adaptive_lossfun.scale().cpu().detach().numpy()
    np.testing.assert_(loss.dtype, float_dtype)
    np.testing.assert_(alpha.dtype, float_dtype)
    np.testing.assert_(scale.dtype, float_dtype)

  @parameterized.named_parameters(
      ('SingleCPU', np.float32, 'cpu'), ('DoubleCPU', np.float64, 'cpu'),
      ('SingleGPU', np.float32, 'cuda'), ('DoubleGPU', np.float64, 'cuda'))
  def testImageLossfunPreservesDtype(self, float_dtype, device_string):
    """Tests that image_lossfun's outputs precisions match its input."""
    device = _get_device(device_string)
    image_size = (64, 64, 3)
    x = float_dtype(np.random.uniform(size=[10] + list(image_size)))
    adaptive_image_lossfun = adaptive.AdaptiveImageLossFunction(
        image_size, float_dtype, device)
    loss = adaptive_image_lossfun.lossfun(torch.tensor(
        x, device=device)).cpu().detach().numpy()
    alpha = adaptive_image_lossfun.alpha().cpu().detach().numpy()
    scale = adaptive_image_lossfun.scale().cpu().detach().numpy()
    np.testing.assert_(loss.dtype, float_dtype)
    np.testing.assert_(alpha.dtype, float_dtype)
    np.testing.assert_(scale.dtype, float_dtype)

  @parameterized.named_parameters(('Adaptive', False), ('StudentsT', True))
  def testImageLossfunPreservesImageSize(self, use_students_t):
    """Tests that image_lossfun's outputs precisions match its input."""
    image_size = (73, 67, 3)
    float_dtype = np.float32
    x = float_dtype(np.random.uniform(size=[10] + list(image_size)))
    adaptive_image_lossfun = adaptive.AdaptiveImageLossFunction(
        image_size, float_dtype, 'cpu', use_students_t=use_students_t)
    loss = adaptive_image_lossfun.lossfun(x).detach().numpy()
    scale = adaptive_image_lossfun.scale().detach().numpy()
    np.testing.assert_(tuple(loss.shape[1:]) == image_size)
    np.testing.assert_(tuple(scale.shape) == image_size)
    if use_students_t:
      df = adaptive_image_lossfun.df().detach().numpy()
      np.testing.assert_(tuple(df.shape) == image_size)
    else:
      alpha = adaptive_image_lossfun.alpha().detach().numpy()
      np.testing.assert_(tuple(alpha.shape) == image_size)

  @parameterized.named_parameters(
      ('WaveletCPU', _generate_wavelet_toy_image_data, 'cpu'),
      ('WaveletGPU', _generate_wavelet_toy_image_data, 'cuda'),
      ('PixelCPU', _generate_pixel_toy_image_data, 'cpu'),
      ('PixelGPU', _generate_pixel_toy_image_data, 'cuda'))
  def testFittingImageDataIsCorrect(self, image_data_callback, device_string):
    """Tests that minimizing the adaptive image loss recovers the true model.

    Here we generate a stack of color images drawn from a normal distribution,
    and then minimize image_lossfun() with respect to the mean and scale of each
    distribution, and check that after minimization the estimated means are
    close to the true means.

    Args:
      image_data_callback: The function used to generate the training data and
        parameters used during optimization.
      device: The device to run on.
    """
    device = _get_device(device_string)
    # Generate toy data.
    image_width = 4
    num_samples = 10
    wavelet_num_levels = 2  # Ignored by _generate_pixel_toy_image_data().
    (samples, reference, color_space,
     representation) = image_data_callback(image_width, num_samples,
                                           wavelet_num_levels)

    float_dtype = np.float64
    prediction = Variable(
        torch.tensor(np.zeros(reference.shape, float_dtype), device=device),
        requires_grad=True)
    adaptive_image_lossfun = adaptive.AdaptiveImageLossFunction(
        (image_width, image_width, 3),
        float_dtype,
        device,
        color_space=color_space,
        representation=representation,
        wavelet_num_levels=wavelet_num_levels,
        alpha_lo=2,
        alpha_hi=2)

    params = torch.nn.ParameterList(adaptive_image_lossfun.parameters())
    optimizer = torch.optim.Adam([p for p in params] + [prediction], lr=0.1)
    for _ in range(1000):
      optimizer.zero_grad()
      x = torch.as_tensor(samples, device=device) - prediction[np.newaxis, :]
      loss = torch.sum(adaptive_image_lossfun.lossfun(x))
      loss.backward(retain_graph=True)
      optimizer.step()

    prediction = prediction.cpu().detach().numpy()
    np.testing.assert_allclose(prediction, reference, rtol=0.01, atol=0.01)


if __name__ == '__main__':
  np.testing.run_module_suite()
