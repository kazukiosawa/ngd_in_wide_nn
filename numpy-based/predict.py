import numpy as np

NG_EXACT = 'exact'
NG_BD = 'block_diagonal'
NG_BTD = 'block_tri_diagonal'
NG_KFAC = 'kfac'


__all__ = [
  'gradient_descent_mse',
  'natural_gradient_descent_mse',
  'exact_natural_gradient_descent_mse',
  'block_wise_natural_gradient_descent_mse',
  'kfac_mse',
  'get_ng_opt_lr',
]


def gradient_descent_mse(fx_train, y_train, g_dd, g_td):
  gx = fx_train - y_train
  eigvals, eigvecs = np.linalg.eigh(g_dd)

  NC = y_train.size

  def expm1_fn(dt):
    # g_dd should be positive semi-definite
    return np.expm1(-np.maximum(eigvals, 0.) * dt / NC)

  # Get analytical predictions
  def predict(fx_test, step, learning_rate):
    dt = step * learning_rate
    expm1_dot_gx = np.einsum(
      'ji,i,ki,k...->j...',
      eigvecs, expm1_fn(dt), eigvecs, gx, optimize=True)


    v_dd = expm1_dot_gx
    g_dd_inv_v_dd = np.linalg.solve(g_dd, v_dd)
    v_td = np.dot(g_td, g_dd_inv_v_dd)

    return fx_train + v_dd, fx_test + v_td

  return predict


def natural_gradient_descent_mse(ng_type, *args, **kwargs):
  if ng_type == NG_EXACT:
    return exact_natural_gradient_descent_mse(*args, **kwargs)

  elif ng_type in [NG_BD, NG_BTD]:
    return block_wise_natural_gradient_descent_mse(ng_type, *args, **kwargs)

  elif ng_type == NG_KFAC:
    return kfac_mse(*args, **kwargs)

  else:
    raise ValueError(f'Invalid ng_type: {ng_type}.')


def exact_natural_gradient_descent_mse(fx_train, y_train, g_dd, g_td):
  gx = fx_train - y_train

  # Get update vector
  g_dd_inv_dot_gx = np.linalg.solve(g_dd, gx)
  v_dd = np.dot(g_dd, g_dd_inv_dot_gx)
  v_td = np.dot(g_td, g_dd_inv_dot_gx)

  # Get analytical predictions
  def predict(fx_test, step, learning_rate):
    dt = 1 - (1 - learning_rate) ** step
    return fx_train - v_dd * dt, fx_test - v_td * dt

  return predict


def block_wise_natural_gradient_descent_mse(ng_type, fx_train, y_train, g_dd, g_td):
  n_layers = 3
  gx = fx_train - y_train

  # Get update vector
  Th1_dd, Th2_dd, Th3_dd = g_dd
  Th1_td, Th2_td, Th3_td = g_td

  Th1_dd_inv_dot_gx = np.linalg.solve(Th1_dd, gx)
  Th2_dd_inv_dot_gx = np.linalg.solve(Th2_dd, gx)
  Th3_dd_inv_dot_gx = np.linalg.solve(Th3_dd, gx)

  v1_dd = np.dot(Th1_dd, Th1_dd_inv_dot_gx)
  v1_td = np.dot(Th1_td, Th1_dd_inv_dot_gx)
  v2_dd = np.dot(Th2_dd, Th2_dd_inv_dot_gx)
  v2_td = np.dot(Th2_td, Th2_dd_inv_dot_gx)
  v3_dd = np.dot(Th3_dd, Th3_dd_inv_dot_gx)
  v3_td = np.dot(Th3_td, Th3_dd_inv_dot_gx)

  if ng_type == NG_BD:

    v_dd = (v1_dd + v2_dd + v3_dd) / n_layers
    v_td = (v1_td + v2_td + v3_td) / n_layers

    # Get analytical predictions
    def predict(fx_test, step, learning_rate):
      dt = 1 - (1 - n_layers * learning_rate) ** step
      return fx_train - v_dd * dt, fx_test - v_td * dt

    return predict

  else:

    T = np.array(
      [[1, 1, 0],
       [1, 1, 1],
       [0, 1, 1]]
    )
    ones = np.ones(n_layers)
    coeff = np.linalg.solve(T, ones)
    alpha = np.sum(coeff)
    v_dd = (v1_dd * coeff[0] + v2_dd * coeff[1] + v3_dd * coeff[2]) / alpha
    v_td = (v1_td * coeff[0] + v2_td * coeff[1] + v3_td * coeff[2]) / alpha

    # Get analytical predictions
    def predict(fx_test, step, learning_rate):
      dt = 1 - (1 - alpha * learning_rate) ** step
      return fx_train - v_dd * dt, fx_test - v_td * dt

    return predict


def kfac_mse(fx_train, y_train, g_dd, g_td, x_train, x_test):
  n_layers = 3
  N, M0 = x_train.shape
  is_low_input_dim = M0 < N

  # Create analytical kernels
  gx = fx_train - y_train
  Q1_dd, B1_dd, Q2_dd, B2_dd, Q3_dd, B3_dd = g_dd
  Q1_td, B1_td, Q2_td, B2_td, Q3_td, B3_td = g_td

  if is_low_input_dim:
    # under the Forster algorithm
    cal_A0_dd = np.matmul(x_train, x_train.T)
    cal_A0_td = np.matmul(x_test, x_train.T)
  else:
    Q1_dd_inv = np.linalg.inv(Q1_dd)
    cal_A0_dd = np.matmul(Q1_dd, Q1_dd_inv)
    cal_A0_td = np.matmul(Q1_td, Q1_dd_inv)

  Q2_dd_inv = np.linalg.inv(Q2_dd)
  cal_A1_dd = np.matmul(Q2_dd, Q2_dd_inv)
  cal_A1_td = np.matmul(Q2_td, Q2_dd_inv)

  Q3_dd_inv = np.linalg.inv(Q3_dd)
  cal_A2_dd = np.matmul(Q3_dd, Q3_dd_inv)
  cal_A2_td = np.matmul(Q3_td, Q3_dd_inv)

  B1_dd_inv = np.linalg.inv(B1_dd)
  cal_B1_dd = np.matmul(B1_dd, B1_dd_inv)
  cal_B1_td = np.matmul(B1_td, B1_dd_inv)

  B2_dd_inv = np.linalg.inv(B2_dd)
  cal_B2_dd = np.matmul(B2_dd, B2_dd_inv)
  cal_B2_td = np.matmul(B2_td, B2_dd_inv)

  cal_B3_dd = np.ones_like(B3_dd)
  cal_B3_td = np.ones_like(B3_td)

  alpha = N * n_layers

  # Get update vector
  G_dd = (cal_B1_dd * cal_A0_dd
          + cal_B2_dd * cal_A1_dd
          + cal_B3_dd * cal_A2_dd) / n_layers
  G_td = (cal_B1_td * cal_A0_td
          + cal_B2_td * cal_A1_td
          + cal_B3_td * cal_A2_td) / n_layers
  v_dd = np.dot(G_dd, gx)
  v_td = np.dot(G_td, gx)

  # Get analytical predictions
  def predict(fx_test, t, learning_rate):
    dt = 1 - (1 - alpha * learning_rate) ** t
    return fx_train - v_dd * dt, fx_test - v_td * dt

  return predict


def get_ng_opt_lr(ng_type, n_layers=None, data_size=None):
  if ng_type == NG_EXACT:
    """
    Exact NGD.
    """
    return 1.

  assert n_layers is not None, 'n_layers needs to be specified.'
  if ng_type == NG_BD:
    """
    NGD with Block-diagonal FIM.
    """
    return 1. / n_layers

  elif ng_type == NG_BTD:
    """
    NGD with Block-tri-diagonal FIM.
    
    This script raises np.linalg.LinAlgError
    when n_layers = 3s+2 (ex. 5, 8, 11, ...)
    because T is singular.
    """
    # build tri-diagonal matrix
    T = np.tri(n_layers, n_layers, 1)
    T = T * T.transpose()
    ones = np.ones(n_layers)
    T_inv_dot_ones = np.linalg.solve(T, ones)
    alpha = np.dot(ones, T_inv_dot_ones)
    return 1. / alpha

  assert data_size is not None, 'data_size needs to be specified.'
  if ng_type == NG_KFAC:
    """
    NGD with K-FAC
    """
    alpha = data_size * n_layers
    return 1. / alpha

  else:
    raise ValueError(f'Invalid ng_type: {ng_type}.')
