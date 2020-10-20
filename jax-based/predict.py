"""Functions to make predictions on the test set using NTK kernel."""


import functools
import operator

import jax.numpy as np
import jax.scipy as sp

from jax.tree_util import tree_map, tree_multimap, tree_reduce
from empirical import flatten_features


NG_EXACT = 'exact'
NG_BD = 'block_diagonal'
NG_BTD = 'block_tri_diagonal'
NG_KFAC = 'kfac'
NG_UNIT = 'unit_wise'


__all__ = [
  'natural_gradient_descent_mse',
  'block_wise_natural_gradient_descent_mse',
  'unit_wise_natural_gradient_descent_mse',
  'get_ng_opt_lr'
]


def natural_gradient_descent_mse(g_dd, y_train, g_td=None, damping=0., diag_reg=0.):
  """Natural gradient with full FIM."""

  assert damping >= 0
  assert diag_reg >= 0
  output_dimension = y_train.shape[-1]

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  g_dd = flatten_features(g_dd)
  inv_op = _inv_operator(g_dd, damping=damping, diag_reg=diag_reg)

  def _dt(eta, t):
    return 1 - (1 - eta) ** t

  def train_predict(eta, t, fx=0., g_dd_inv_dot_gx=None):
    dt = _dt(eta, t)
    if damping == 0:
      gx_train = fx - y_train
      dfx = -gx_train * dt
      return fx + dfx
    else:
      gx_train = fl(fx - y_train)
      if g_dd_inv_dot_gx is None:
        g_dd_inv_dot_gx = inv_op(gx_train)
      dfx = -np.dot(g_dd, g_dd_inv_dot_gx) * dt
      return ufl(dfx) + fx

  if g_td is None:
    return train_predict

  g_td = flatten_features(g_td)

  def train_test_predict(eta, t, fx_train=0., fx_test=0.):
    dt = _dt(eta, t)
    gx_train = fl(fx_train - y_train)
    g_dd_inv_dot_gx = inv_op(gx_train)
    dfx = -np.dot(g_td, g_dd_inv_dot_gx) * dt
    return train_predict(eta, t, fx_train, g_dd_inv_dot_gx), fx_test + ufl(dfx)

  return train_test_predict


def _map_inv_vec_product(kernels, damping, diag_reg=0.):

  def op(vec):
    return tree_map(lambda k: _inv_operator(k, damping=damping, diag_reg=diag_reg)(vec), kernels)

  return op


def _multimap_kernel_vec_product(kernels):

  def op(vecs):
    return tree_reduce(
      operator.add,
      tree_multimap(lambda k, v: np.dot(k, v), kernels, vecs))

  return op


def block_wise_natural_gradient_descent_mse(lw_g_dd, y_train, lw_g_td=None, damping=0., diag_reg=0., tri=False):

  assert damping >= 0
  n_data, output_dimension = y_train.shape
  n_layers = len(lw_g_dd)

  lw_g_dd = tree_map(lambda k: flatten_features(k), lw_g_dd)
  inv_op = _map_inv_vec_product(lw_g_dd, damping, diag_reg)
  kvp_op_dd = _multimap_kernel_vec_product(lw_g_dd)

  if lw_g_td is not None:
    lw_g_td = tree_map(lambda k: flatten_features(k), lw_g_td)
    kvp_op_td = _multimap_kernel_vec_product(lw_g_td)

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  def _dt(eta, t, alpha):
    return 1 - (1 - alpha * eta) ** t

  if not tri:
    """Natural gradient with block-diagonal FIM."""
    _dt = functools.partial(_dt, alpha=n_layers)

    def train_predict(eta, t, fx=0., lw_vec=None):
      dt = _dt(eta, t)
      if damping == 0:
        gx_train = fx - y_train
        dfx = -gx_train * dt
        return dfx + fx
      else:
        if lw_vec is None:
          gx_train = fl(fx - y_train)
          lw_vec = inv_op(gx_train)
        kvp = kvp_op_dd(lw_vec)
        dfx = -kvp / n_layers * dt
        return ufl(dfx) + fx

    if lw_g_td is None:
      return train_predict

    def train_test_predict(eta, t, fx_train=0., fx_test=0.):
      dt = _dt(eta, t)
      gx_train = fl(fx_train - y_train)
      lw_vec = inv_op(gx_train)
      kvp = kvp_op_td(lw_vec)
      dfx = -kvp / n_layers * dt
      return train_predict(eta, t, fx_train, lw_vec), fx_test + ufl(dfx)

    return train_test_predict

  """Natural gradient with block-tri-diagonal FIM."""
  assert n_layers % 3 != 2, f'Cannot get predictions with given n_layers: {n_layers}.'

  # Prepare tri-diagonal matrix
  T = np.tri(n_layers, n_layers, 1)
  T = T * T.transpose()
  ones = np.ones(n_layers)
  T_inv_dot_ones = sp.linalg.solve(T, ones)
  alpha = np.dot(ones, T_inv_dot_ones)
  _dt = functools.partial(_dt, alpha=alpha)

  if damping > 0 or lw_g_td is not None:
    T_kron_I = np.kron(T, np.eye(n_data * output_dimension))
    block_tri_diag_g_dd = T_kron_I @ sp.linalg.block_diag(*lw_g_dd)
    block_tri_diag_g_dd = _add_damping(block_tri_diag_g_dd, damping)
    block_inv_op = lambda vec: sp.linalg.solve(block_tri_diag_g_dd, vec)

  def train_predict(eta, t, fx=0., lw_vec=None):
    dt = _dt(eta, t)
    if damping == 0:
      gx_train = fx - y_train
      dfx = -gx_train * dt
      return dfx + fx
    else:
      if lw_vec is None:
        gx_train = fl(fx - y_train)
        gx_train = np.tile(gx_train, n_layers)
        lw_vec = np.split(block_inv_op(gx_train), n_layers)
      kvp = kvp_op_dd(lw_vec)
      dfx = -kvp / alpha * dt
      return ufl(dfx) + fx

  if lw_g_td is None:
    return train_predict

  def train_test_predict(eta, t, fx_train=0., fx_test=0.):
    dt = _dt(eta, t)
    gx_train = fl(fx_train - y_train)
    if damping == 0:
      lw_vec = [v * val for v, val in zip(inv_op(gx_train), T_inv_dot_ones)]
    else:
      gx_train = np.tile(gx_train, n_layers)
      lw_vec = np.split(block_inv_op(gx_train), n_layers)
    kvp = kvp_op_td(lw_vec)
    dfx = -kvp / alpha * dt
    return train_predict(eta, t, fx_train, lw_vec), fx_test + ufl(dfx)

  return train_test_predict


def unit_wise_natural_gradient_descent_mse(uw_g_dd, y_train, width, uw_g_td=None,
                                           damping=0., activation='relu'):
  """Natural gradient with unit-wise FIM."""
  assert damping >= 0
  n_data, output_dimension = y_train.shape

  uw_g_dd = tree_map(lambda g_dd: flatten_features(g_dd), uw_g_dd)
  inv_op = _map_inv_vec_product(uw_g_dd, damping)
  kvp_op_dd = _multimap_kernel_vec_product(uw_g_dd)
  if uw_g_td is not None:
    uw_g_td = tree_map(lambda g_td: flatten_features(g_td), uw_g_td)
    kvp_op_td = _multimap_kernel_vec_product(uw_g_td)

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  n_layers = len(uw_g_dd)
  gamma = 0.5 if activation == 'relu' else 1.
  alpha = gamma * (n_layers - 1) * width

  def _dt(eta, t):
    return 1 - (1 - alpha * eta) ** t

  def train_predict(eta, t, fx=0., uw_vec=None):
    dt = _dt(eta, t)
    if damping == 0:
      gx_train = fx - y_train
      dfx = -gx_train * dt
      return dfx + fx
    else:
      if uw_vec is None:
        gx_train = fl(fx - y_train)
        uw_vec = inv_op(gx_train)
      kvp = kvp_op_dd(uw_vec)
      dfx = -kvp / alpha * dt
      return ufl(dfx) + fx

  if uw_g_td is None:
    return train_predict

  def train_test_predict(eta, t, fx_train=0., fx_test=0.):
    dt = _dt(eta, t)
    gx_train = fl(fx_train - y_train)
    uw_vec = inv_op(gx_train)
    kvp = kvp_op_td(uw_vec)
    dfx = -kvp / alpha * dt
    return train_predict(eta, t, fx_train, uw_vec), fx_test + ufl(dfx)

  return train_test_predict


def _add_damping(mat, damping=0.):
  dimension = mat.shape[0]
  return mat + damping * np.eye(dimension)


def _add_diagonal_regularizer(covariance, diag_reg=0.):
  dimension = covariance.shape[0]
  reg = np.trace(covariance) / dimension
  return covariance + diag_reg * reg * np.eye(dimension)


def _inv_operator(g_dd, diag_reg=0.0, damping=0.0):
  if diag_reg > 0:
    g_dd = _add_diagonal_regularizer(g_dd, diag_reg)
  if damping > 0:
    g_dd = _add_damping(g_dd, damping)
  return lambda vec: sp.linalg.solve(g_dd, vec, sym_pos=True)


def get_ng_opt_lr(ng_type, n_layers=None, width=None, gamma=None):
  if ng_type not in [NG_EXACT, NG_BD, NG_BTD, NG_UNIT]:
    raise ValueError(f'Invalid ng_type {ng_type}.')

  if ng_type == NG_EXACT:
    """
    Exact NGD.
    opt_lr = 1
    """
    return 1.

  assert n_layers is not None, 'n_layers needs to be specified.'
  if ng_type == NG_BD:
    """
    NGD with Block-diagonal FIM.
    opt_lr = 1/n_layers
    """
    return 1 / n_layers

  elif ng_type == NG_BTD:
    """
    NGD with Block-tri-diagonal FIM.
    opt_lr = 1/{1'inv(T)1}
    
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
    return 1 / alpha

  assert width is not None, 'width needs to be specified.'
  """
  Unit-wise NGD.
  opt_lr = 1/gamma/(n_layers-1)/width
  """
  if gamma is None:
    gamma = 0.5
  return 1 / gamma / (n_layers-1) / width
