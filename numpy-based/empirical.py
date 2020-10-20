import numpy as np
from functools import partial

NG_EXACT = 'exact'
NG_BD = 'block_diagonal'
NG_BTD = 'block_tri_diagonal'
NG_KFAC = 'kfac'


__all__ = [
  'init_ntk_mlp',
  'forward_and_backward',
  'empirical_kernel',
  'gradient_mse',
  'natural_gradient_mse',
  'exact_natural_gradient_mse',
  'block_wise_natural_gradient_mse',
  'kfac_mse'
]


def activation(inputs, name='relu'):
  if name == 'relu':
    derivatives = inputs > 0
    outputs = inputs * derivatives
    return outputs, derivatives
  else:
    raise ValueError(f'Invalid activation name: {name}.')


def init_ntk_mlp(M3, M2, M1, M0, W_std=1., b_std=0., random_seed=0):
  """Initialize weights and biases of NTK-parameterization."""
  np.random.seed(random_seed)
  W3 = np.random.randn(M3, M2).astype(np.float32) * W_std / np.sqrt(M2)
  W2 = np.random.randn(M2, M1).astype(np.float32) * W_std / np.sqrt(M1)
  W1 = np.random.randn(M1, M0).astype(np.float32) * W_std / np.sqrt(M0)

  b3 = np.random.randn(M3).astype(np.float32) * b_std
  b2 = np.random.randn(M2).astype(np.float32) * b_std
  b1 = np.random.randn(M1).astype(np.float32) * b_std

  return W3, W2, W1, b3, b2, b1


def forward_and_backward(x, W1, W2, W3, backward=True, kfac=False):
  """
  Simple MLP with three layers.

  x -> (W1) -> u1 -> (relu) -> h1 -> (W2) -> u2
  -> (relu) -> h2 -> (W3) -> f
  """
  act_fn = partial(activation, name='relu')
  N = x.shape[0]
  # forward
  u1 = np.einsum('ab,nb->na', W1, x)
  h1, d_h1 = act_fn(u1)
  del u1
  u2 = np.einsum('ab,nb->na', W2, h1)
  h2, d_h2 = act_fn(u2)
  del u2
  f = np.einsum('ab,nb->na', W3, h2)

  if not backward:
    return f

  M3 = W3.shape[0]

  # back-propagate Jacobian of fx
  if kfac:
    J_h2 = W3
    J_u2 = np.einsum('ab,nb->nab', J_h2, d_h2)
    J_h1 = np.einsum('nab,bc->nac', J_u2, W2)
    J_u1 = np.einsum('nab,nb->nab', J_h1, d_h1)
    return f, h1, h2, J_u1, J_u2
  else:
    J_W3 = np.hstack([np.vstack([h2] * M3)] * M3)
    J_h2 = W3
    J_u2 = np.einsum('ab,nb->nab', J_h2, d_h2)
    del J_h2, d_h2
    J_W2 = np.einsum('nab,nc->nabc', J_u2, h1).reshape(N * M3, -1)
    del h1
    J_h1 = np.einsum('nab,bc->nac', J_u2, W2)
    del J_u2
    J_u1 = np.einsum('nab,nb->nab', J_h1, d_h1)
    del d_h1
    J_W1 = np.einsum('nab,nc->nabc', J_u1, x).reshape(N * M3, -1)
    del J_u1

    return f, J_W1, J_W2, J_W3


def empirical_kernel(x1, x2, w_var, W1, W2, W3, ng_type):
  _, J_W1_1, J_W2_1, J_W3_1 = forward_and_backward(x1, W1, W2, W3)
  if x2 is None:
    J_W1_2, J_W2_2, J_W3_2 = J_W1_1, J_W2_1, J_W3_1
    N = x1.shape[0]
  else:
    _, J_W1_2, J_W2_2, J_W3_2 = forward_and_backward(x2, W1, W2, W3)
    N = np.sqrt(x1.shape[0] * x2.shape[0])

  M0 = W1.shape[-1]
  M1 = W2.shape[-1]
  M2 = W3.shape[-1]

  Th1 = w_var * np.dot(J_W1_1, J_W1_2.T) / (N * M0)
  Th2 = w_var * np.dot(J_W2_1, J_W2_2.T) / (N * M1)
  Th3 = w_var * np.dot(J_W3_1, J_W3_2.T) / (N * M2)

  Th1 = Th1.astype(J_W1_1.dtype)
  Th2 = Th2.astype(J_W2_1.dtype)
  Th3 = Th3.astype(J_W3_1.dtype)

  if ng_type == NG_EXACT:
    return Th1 + Th2 + Th3

  return Th1, Th2, Th3


def gradient_mse(x, y, w_var, W1, W2, W3):
  """
  Gradient.
  """
  fx, J_W1, J_W2, J_W3 = forward_and_backward(x, W1, W2, W3)
  gx = (fx - y).reshape(-1, 1)
  N, M0 = x.shape
  M1 = W1.shape[0]
  M2 = W2.shape[0]

  g1 = np.dot(J_W1.T, gx)
  g2 = np.dot(J_W2.T, gx)
  g3 = np.dot(J_W3.T, gx)

  dW1 = w_var * g1.reshape(W1.shape) / (N * M0)
  dW2 = w_var * g2.reshape(W2.shape) / (N * M1)
  dW3 = w_var * g3.reshape(W3.shape) / (N * M2)

  return dW1, dW2, dW3


def natural_gradient_mse(ng_type, *args, **kwargs):
  if ng_type == NG_EXACT:
    return exact_natural_gradient_mse(*args, **kwargs)

  elif ng_type in [NG_BD, NG_BTD]:
    return block_wise_natural_gradient_mse(ng_type, *args, **kwargs)

  elif ng_type == NG_KFAC:
    return kfac_mse(*args, **kwargs)

  else:
    raise ValueError(f'Invalid ng_type: {ng_type}.')


def exact_natural_gradient_mse(x, y, w_var, W1, W2, W3, damping):
  """
  Exact natural-gradient.
  """
  fx, J_W1, J_W2, J_W3 = forward_and_backward(x, W1, W2, W3)
  gx = (fx - y).flatten()
  N, M0 = x.shape
  M1 = W1.shape[0]
  M2 = W2.shape[0]
  M3 = W3.shape[0]

  Th = empirical_kernel(x, None, w_var, W1, W2, W3, NG_EXACT)
  I = np.eye(N * M3)
  Th += damping * I

  Th_inv_dot_gx = np.linalg.solve(Th, gx)

  v1 = np.dot(J_W1.T, Th_inv_dot_gx)
  v2 = np.dot(J_W2.T, Th_inv_dot_gx)
  v3 = np.dot(J_W3.T, Th_inv_dot_gx)

  dW1 = w_var * v1.reshape(W1.shape) / (N * M0)
  dW2 = w_var * v2.reshape(W2.shape) / (N * M1)
  dW3 = w_var * v3.reshape(W3.shape) / (N * M2)

  return dW1, dW2, dW3


def block_wise_natural_gradient_mse(ng_type, x, y, w_var, W1, W2, W3, damping):
  fx, J_W1, J_W2, J_W3 = forward_and_backward(x, W1, W2, W3)
  gx = (fx - y).flatten()

  N, M0 = x.shape
  M1 = W1.shape[0]
  M2 = W2.shape[0]
  M3 = W3.shape[0]

  # layer-wise empirical kernels
  Th1, Th2, Th3 = empirical_kernel(x, None, w_var, W1, W2, W3, ng_type)

  if ng_type == NG_BD:
    """
    Block-diagonal natural-gradient.
    """
    I = np.eye(N * M3)
    Th1 += damping * I
    Th2 += damping * I
    Th3 += damping * I

    v1 = np.dot(J_W1.T, np.linalg.solve(Th1, gx))
    v2 = np.dot(J_W2.T, np.linalg.solve(Th2, gx))
    v3 = np.dot(J_W3.T, np.linalg.solve(Th3, gx))

  else:
    """
    Block-tridiagonal natural-gradient.
    """
    n_layers = 3
    O = np.zeros((N * M3, N * M3)).astype(Th1.dtype)
    I = np.eye(N * M3 * n_layers).astype(Th1.dtype)
    mat = np.block(
      [[Th1, Th2, O],
       [Th1, Th2, Th3],
       [O, Th2, Th3]]
    ).astype(Th1.dtype)
    mat += damping * I
    gx = np.stack([gx, gx, gx]).reshape(N * M3 * n_layers, 1)

    v = np.linalg.solve(mat, gx)
    v = np.split(v, n_layers)
    v1 = np.dot(J_W1.T, v[0])
    v2 = np.dot(J_W2.T, v[1])
    v3 = np.dot(J_W3.T, v[2])

  dW1 = w_var * v1.reshape(W1.shape) / (N * M0)
  dW2 = w_var * v2.reshape(W2.shape) / (N * M1)
  dW3 = w_var * v3.reshape(W3.shape) / (N * M2)

  return dW1, dW2, dW3


def kfac_mse(x, y, w_var, W1, W2, W3, damping):
  """
  K-FAC.
  """
  # only support binary classification
  assert y.shape[-1] == 1, 'Only binary classification is supported for K-FAC.'
  fx, h1, h2, J_u1, J_u2 = forward_and_backward(x, W1, W2, W3, kfac=True)
  gx = fx - y
  N, M0 = x.shape
  M1 = W1.shape[0]
  M2 = W2.shape[0]
  J_u1 = J_u1.reshape(-1, M1)
  J_u2 = J_u2.reshape(-1, M2)

  def get_A_and_B_inv(h, d, M_in, M_out):
    # compute damping for A and B
    A_dual = (w_var ** 2 / N / M_in) * np.dot(h, h.T)
    B_dual = (1 / N) * np.dot(d, d.T)
    A_avg_trace = np.trace(A_dual) / M_in
    B_avg_trace = np.trace(B_dual) / M_out
    pi = np.sqrt(A_avg_trace / B_avg_trace)
    I = np.eye(N, N)
    A_dmp = I * np.sqrt(damping) * pi
    B_dmp = I * np.sqrt(damping) * (1 / pi)

    A_inv = np.dot(h.T, np.linalg.inv(np.dot(h, h.T) * (1/N) + A_dmp))
    B_inv = np.dot(d.T, np.linalg.inv(np.dot(d, d.T) * (1/N) + B_dmp))
    return A_inv, B_inv

  A0_inv, B1_inv = get_A_and_B_inv(x, J_u1, M0, M1)
  A1_inv, B2_inv = get_A_and_B_inv(h1, J_u2, M1, M2)
  dmp = np.eye(N, N) * damping
  A2_inv = np.dot(h2.T, np.linalg.inv(np.dot(h2, h2.T) * (1/N) + dmp))

  v1 = np.einsum('in,jn,nk->ijk', B1_inv, A0_inv, gx)
  v2 = np.einsum('in,jn,nk->ijk', B2_inv, A1_inv, gx)
  v3 = np.einsum('jn,nk->jk', A2_inv, gx) * 1/N

  dW1 = w_var * v1.reshape(W1.shape) / np.sqrt(M0)
  dW2 = w_var * v2.reshape(W2.shape) / np.sqrt(M1)
  dW3 = w_var * v3.reshape(W3.shape) / np.sqrt(M2)

  return dW1, dW2, dW3
