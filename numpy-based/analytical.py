import numpy as np

NG_EXACT = 'exact'
NG_BD = 'block_diagonal'
NG_BTD = 'block_tri_diagonal'
NG_KFAC = 'kfac'


__all__ = [
  'batch_analytic_kernel',
  'analytic_kernel'
]


def batch_analytic_kernel(x1, x2, w_var, ng_type, batch_size):
  N1 = x1.shape[0]
  N2 = x2.shape[0]
  assert N1 % batch_size == 0 and N2 % batch_size == 0
  n_batch1 = int(N1 / batch_size)
  n_batch2 = int(N2 / batch_size)
  x1s = np.split(x1, n_batch1)
  x2s = np.split(x2, n_batch2)
  kernel = []
  for i in range(n_batch1):
    kernel_i = []
    for j in range(n_batch2):
      rst = analytic_kernel(x1s[i], x2s[j], w_var, ng_type)
      kernel_i.append(rst)
    if isinstance(kernel_i[0], list):
      kernel_i = [np.hstack([k[idx] for k in kernel_i]) for idx in range(len(kernel_i[0]))]
    else:
      kernel_i = np.hstack(kernel_i)
    kernel.append(kernel_i)

  if isinstance(kernel[0], list):
    return [np.vstack([k[idx] for k in kernel]) for idx in range(len(kernel[0]))]
  else:
    return np.vstack(kernel)


def analytic_kernel(x1, x2, w_var, ng_type):
  def _sqrt(x):
    return np.sqrt(np.maximum(x, 0))

  b_var = 0  # assume no bias

  if x2 is None:
    x2 = x1

  N1 = x1.shape[0]
  N2 = x2.shape[0]
  ones = np.ones((N1, N2))
  pi = np.pi
  M0 = x1.shape[-1]

  # forward
  q0 = 2 / M0
  A0 = np.matmul(x1, x2.T) / M0

  q1 = w_var / 2 * q0 + b_var
  Q1 = w_var * A0 + b_var * ones
  P1 = Q1 / q1
  P1 = np.clip(P1, -1, 1)
  arcsin_P1 = np.arcsin(P1)

  A1 = (q1 / 2 / pi) * (_sqrt(ones - P1 ** 2) + (pi / 2) * P1 + P1 * arcsin_P1)

  q2 = w_var / 2 * q1 + b_var
  Q2 = w_var * A1 + b_var * ones
  P2 = Q2 / q2
  P2 = np.clip(P2, -1, 1)
  arcsin_P2 = np.arcsin(P2)
  A2 = (q2 / 2 / pi) * (_sqrt(ones - P2 ** 2) + (pi / 2) * P2 + P2 * arcsin_P2)

  Q3 = w_var * A2 + b_var * ones

  # backward
  B3 = ones

  K2 = (1 / 2 / pi) * (arcsin_P2 + (pi / 2) * ones)
  B2 = w_var * B3 * K2

  K1 = (1 / 2 / pi) * (arcsin_P1 + (pi / 2) * ones)
  B1 = w_var * B2 * K1

  if ng_type in [NG_BD, NG_BTD]:
    Th1 = w_var * B1 * A0 + b_var * B1
    Th2 = w_var * B2 * A1 + b_var * B2
    Th3 = w_var * B3 * A2 + b_var * B3
    return [Th1, Th2, Th3]

  if ng_type == NG_KFAC:
    return [Q1, B1, Q2, B2, Q3, B3]

  Th = w_var * B1 * A0 + b_var * B1
  Th += w_var * B2 * A1 + b_var * B2
  Th += w_var * B3 * A2 + b_var * B3
  return Th


