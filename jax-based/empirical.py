from functools import partial

from absl import flags
from jax import random
from jax.api import eval_shape
from jax.api import jacobian, hessian
from jax.api import vjp
from jax.experimental.stax import Softmax
import jax.numpy as np
import jax.scipy as sp
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap
from jax.ops import index, index_update
import neural_tangents as nt


FLAGS = flags.FLAGS


__all__ = [
  'natural_gradient_mse_fn',
  'block_wise_natural_gradient_mse_fn',
  'unit_wise_natural_gradient_mse_fn',
  'empirical_direct_layer_wise_ntk_fn',
  'empirical_implicit_layer_wise_ntk_fn',
  'empirical_direct_unit_wise_ntk_fn',
  'natural_gradient_cross_entropy_fn',
  'unflatten_layer_wise_kernel'
]


def flatten_features(kernel):
  """Flatten an empirical kernel."""
  if kernel.ndim == 2:
    return kernel
  assert kernel.ndim % 2 == 0
  half_shape = (kernel.ndim - 1) // 2
  n1, n2 = kernel.shape[:2]
  feature_size = int(np.prod(np.array(kernel.shape[2 + half_shape:])))
  transposition = ((0,) + tuple(i + 2 for i in range(half_shape)) +
                   (1,) + tuple(i + 2 + half_shape for i in range(half_shape)))
  kernel = np.transpose(kernel, transposition)
  return np.reshape(kernel, (feature_size * n1, feature_size * n2))


def _read_keys(keys):
  if keys is None or (isinstance(keys, np.ndarray) and keys.shape == (2,)):
    key1 = key2 = keys
  elif isinstance(keys, tuple):
    # assuming x1 and x2 using key1 and key2, resp.
    key1, key2 = keys
  elif isinstance(keys, np.ndarray) and keys.shape == (2, 2):
    key1, key2 = keys[0], keys[1]
  else:
    raise ValueError('`keys` must be one of the following: `None`, a PRNG '
                     'key, a tuple of PRNG keys or a (2, 2) array and dtype '
                     'unint32')
  return key1, key2


def empirical_implicit_layer_wise_ntk_fn(f, single_input):

  if single_input:

    def ntk_fn(x1, params, keys=None, flatten=True):
      key, _ = _read_keys(keys)
      f_dummy = partial(f, rng=random.PRNGKey(1))
      fx_struct = eval_shape(f_dummy, params, x1)
      fx_dummy = np.ones(fx_struct.shape, fx_struct.dtype)

      def delta_vjjv(delta):
        def delta_vjp(delta):
          return vjp(lambda p: f(p, x1, rng=key), params)[1](delta)
        return tree_map(lambda v: np.sum(v * v), delta_vjp(delta))

      ntk_list = hessian(delta_vjjv)(fx_dummy)[0]
      ntk_list = tree_map(lambda x: np.transpose(x * 0.5, (0, 2, 1, 3)), ntk_list)
      lw_ntk = np.stack([np.sum(ntk, axis=0) for ntk in ntk_list if len(ntk) > 0])

      if flatten:
        lw_ntk = np.transpose(lw_ntk, (1, 2, 3, 4, 0))  # (N, N, C, C, L)
        lw_ntk = np.reshape(lw_ntk, lw_ntk.shape[:3] + (-1,))  # (N, N, C, CL)

      return lw_ntk

    return ntk_fn

  def ntk_fn(x1, x2, params, keys=None, flatten=True):
    key1, key2 = _read_keys(keys)
    f_dummy = partial(f, rng=random.PRNGKey(1))
    fx1_struct = eval_shape(f_dummy, params, x1)
    fx1_dummy = np.ones(fx1_struct.shape, fx1_struct.dtype)
    if x2 is None:
      x2 = x1
      fx2_dummy = fx1_dummy
    else:
      fx2_struct = eval_shape(f_dummy, params, x2)
      fx2_dummy = np.ones(fx2_struct.shape, fx2_struct.dtype)

    def delta_vjjv(delta1, delta2):
      def delta_vjp(delta, x, key):
        return vjp(lambda p: f(p, x, rng=key), params)[1](delta)
      djp1 = delta_vjp(delta1, x1, key1)
      djp2 = delta_vjp(delta2, x2, key2)
      return tree_multimap(lambda v1, v2: np.sum(v1 * v2), djp1, djp2)

    def delta_jjv(delta2, delta1):
      f_jjv = jacobian(partial(delta_vjjv, delta2=delta2))
      return f_jjv(delta1)

    ntk_list = jacobian(partial(delta_jjv, delta1=fx1_dummy))(fx2_dummy)[0]
    ntk_list = tree_map(lambda x: np.transpose(x, (0, 2, 1, 3)), ntk_list)
    lw_ntk = np.stack([np.sum(ntk, axis=0) for ntk in ntk_list if len(ntk) > 0])

    if flatten:
      lw_ntk = np.transpose(lw_ntk, (1, 2, 3, 4, 0))  # (N, N, C, C, L)
      lw_ntk = np.reshape(lw_ntk, lw_ntk.shape[:3] + (-1,))  # (N, N, C, CL)

    return lw_ntk

  return ntk_fn


def empirical_direct_layer_wise_ntk_fn(f):

  def just_contract(j1, j2):
    def contract(x, y):
      param_count = int(np.prod(np.array(x.shape[2:])))
      x = np.reshape(x, x.shape[:2] + (param_count,))
      y = np.reshape(y, y.shape[:2] + (param_count,))
      return np.transpose(np.dot(x, np.transpose(y, (0, 2, 1))), (0, 2, 1, 3))

    return tree_multimap(contract, j1, j2)

  def ntk_fn(x1, x2, params, keys=None, flatten=True):
    key1, key2 = _read_keys(keys)
    f1 = partial(f, rng=key1)
    jac_fn1 = jacobian(f1)
    j1 = jac_fn1(params, x1)
    if x2 is None:
      j2 = j1
    else:
      f2 = partial(f, rng=key2)
      jac_fn2 = jacobian(f2)
      j2 = jac_fn2(params, x2)

    ntk_list = just_contract(j1, j2)
    lw_ntk = np.stack([np.sum(np.array(ntk), axis=0) for ntk in ntk_list if len(ntk) > 0])
    if flatten:
      lw_ntk = np.transpose(lw_ntk, (1, 2, 3, 4, 0))  # (N, N, C, C, L)
      lw_ntk = np.reshape(lw_ntk, lw_ntk.shape[:3] + (-1,))  # (N, N, C, CL)
    return lw_ntk

  return ntk_fn


def unflatten_layer_wise_kernel(kernel, n_layers):
  # (N, N, C, CL) -> L x (N, N, C, C)
  kernel = np.reshape(kernel, kernel.shape[:3] + (-1, n_layers))
  kernel = np.transpose(kernel, (4, 0, 1, 2, 3))
  return list(kernel)


def empirical_direct_unit_wise_ntk_fn(f):

  def just_contract(j1, j2):
    def contract(x, y):
      x = np.transpose(x, (-1,) + tuple(range(x.ndim - 1)))
      y = np.transpose(y, (-1,) + tuple(range(y.ndim - 1)))
      return np.einsum('mic...,mjk...->mijck', x, y)

    return tree_multimap(contract, j1, j2)

  def ntk_fn(x1, x2, params, keys=None):
    key1, key2 = _read_keys(keys)
    f1 = partial(f, rng=key1)
    jac_fn1 = jacobian(f1)
    j1 = jac_fn1(params, x1)
    if x2 is None:
      j2 = j1
    else:
      f2 = partial(f, rng=key2)
      jac_fn2 = jacobian(f2)
      j2 = jac_fn2(params, x2)

    # TODO(kazukiosawa): support neural_tangents.batch
    ntk_list = just_contract(j1, j2)
    uw_ntk = [list(np.sum(np.array(ntk), axis=0))
              for ntk in ntk_list
              if len(ntk) > 0]  # (L, M, N, N, C, C)
    return uw_ntk

  return ntk_fn


def hessian_of_log_likelihood_fn(f):

  softmax_fn = partial(Softmax[1], params=None)

  def hessian_fn(params, x):
    fx = f(params, x)
    probs = softmax_fn(inputs=fx)
    diags = np.array([np.diag(p) for p in probs])
    hessian = diags - np.einsum('bi,bj->bij', probs, probs)
    return hessian

  return hessian_fn


def _add_damping(covariance, damping=0., diag_reg=0.):
  dimension = covariance.shape[0]
  if damping > 0:
    covariance += damping * np.eye(dimension)
  if diag_reg > 0:
    reg = np.trace(covariance) / dimension
    covariance += diag_reg * reg * np.eye(dimension)
  return covariance + damping * np.eye(dimension)


def solve_w_damping(mat, vec, damping, diag_reg, sym_pos=False):
  mat = flatten_features(mat)
  mat = _add_damping(mat, damping, diag_reg)
  return sp.linalg.solve(mat, vec, sym_pos=sym_pos)


def solve(mat, vec, sym_pos=False):
  mat = flatten_features(mat)
  return sp.linalg.solve(mat, vec, sym_pos=sym_pos)


def get_solver_by_damping_value(damping, diag_reg=0.):
  if damping == 0 and diag_reg == 0:
    return solve
  return partial(solve_w_damping, damping=damping, diag_reg=diag_reg)


def _vjp(vec, f, params, inputs):
  fx = partial(f, inputs=inputs)
  _, f_vjp = vjp(fx, params)
  return f_vjp(vec)[0]


def _layer_wise_vjp(lw_vec, f, params, inputs):
  # TODO(kazukiosawa): use tree_map instead of for-loop
  fx = partial(f, inputs=inputs)
  _, f_vjp = vjp(fx, params)
  rst = []
  layer_id = 0
  for i, param in enumerate(params):
    if len(param) == 0:
      rst.append(())
      continue
    rst.append(f_vjp(lw_vec[layer_id])[0][i])
    layer_id += 1

  assert layer_id == len(lw_vec)

  return rst


def _unit_wise_vjp(uw_vec, f, params, inputs):
  fx = partial(f, inputs=inputs)
  _, f_vjp = vjp(fx, params)
  rst = []
  layer_id = 0
  # TODO(kazukiosawa): use tree_map instead of for-loop
  for i, param in enumerate(params):
    if len(param) == 0:
      rst.append(())
      continue
    n_units = param[0].shape[-1]
    if len(param) > 1:
      for j in range(1, len(param)):
        assert n_units == param[j].shape[-1]

    vec = [np.zeros_like(p) for p in param]
    for unit_id in range(n_units):
      _vec = f_vjp(uw_vec[layer_id][unit_id])[0][i]
      for j, (v, _v) in enumerate(zip(vec, _vec)):
        if v.ndim == 2:
            vec[j] = index_update(v, index[:, unit_id], _v[:, unit_id])
        else:
            vec[j] = index_update(v, unit_id, _v[unit_id])
    rst.append(tuple(vec))

    layer_id += 1

  assert layer_id == len(uw_vec)

  return rst


def natural_gradient_cross_entropy_fn(f, output_dimension, damping=1e-5, diag_reg=0,
                                      plus_mse=False, kernel_batch_size=32, device_count=0, store_on_device=True):

  def fl(gx):
    """Flatten loss gradient wrt fx."""
    return np.reshape(gx, (-1,))

  def ufl(gx):
    """Unflatten loss gradient wrt fx."""
    return np.reshape(gx, (-1, output_dimension))

  ntk_fn = nt.batch(nt.empirical_ntk_fn(f, trace_axes=()), kernel_batch_size, device_count, store_on_device)
  hessian_fn = hessian_of_log_likelihood_fn(f)
  softmax_fn = partial(Softmax[1], params=None)
  _solve_w_damping = get_solver_by_damping_value(damping, diag_reg)

  def _get_gx(params, x, y):
    preds = softmax_fn(inputs=f(params, x))  # (N, C)
    gx = fl(preds - y)  # (NC,)
    return gx

  def _get_hessian_fx(params, x):
    hess = hessian_fn(params, x)  # (N, C, C)
    hess = sp.linalg.block_diag(*list(hess))  # (NC, NC)
    return hess

  if not plus_mse:
    """Natural gradient with full FIM."""
    def ng_fn(params, x, y):
      g_dd = ntk_fn(x, None, params)  # (N, N, C, C)
      g_dd = flatten_features(g_dd)  # (NC, NC)
      gx = _get_gx(params, x, y)  # (NC,)
      hess = _get_hessian_fx(params, x)  # (NC, NC)
      vec = ufl(_solve_w_damping(hess @ g_dd, gx))  # (N, C)
      return _vjp(vec, f, params, x)  # params.shape

    return ng_fn

  def ng_plus_mse_fn(params, x, y):
    """Natural gradient with full FIM plus MSE correction."""
    g_dd = ntk_fn(x, None, params)  # (N, N, C, C)
    gx = _get_gx(params, x, y)  # (NC,)
    hess = _get_hessian_fx(params, x)  # (NC, NC)
    hess_inv_dot_gx = _solve_w_damping(hess, gx, sym_pos=True)  # (NC,)
    vec = ufl(solve(g_dd, hess_inv_dot_gx, sym_pos=True))  # (N, C)
    return _vjp(vec, f, params, x)  # params.shape

  return ng_plus_mse_fn


def natural_gradient_mse_fn(f, output_dimension, damping=0., diag_reg=0.,
                            kernel_batch_size=32, device_count=0, store_on_device=True):
  """Natural gradient with full FIM."""

  def fl(gx):
    """Flatten loss gradient wrt fx."""
    return np.reshape(gx, (-1,))

  def ufl(gx):
    """Unflatten loss gradient wrt fx."""
    return np.reshape(gx, (-1, output_dimension))

  ntk_fn = nt.batch(nt.empirical_ntk_fn(f, trace_axes=()), kernel_batch_size, device_count, store_on_device)
  _solve_w_damping = get_solver_by_damping_value(damping, diag_reg)

  def ng_fn(params, x, y):
    g_dd = ntk_fn(x, None, params)  # (N, N, C, C)
    gx = fl(f(params, x) - y)  # (NC,)
    vec = ufl(_solve_w_damping(g_dd, gx, sym_pos=True))  # (N, C)
    return _vjp(vec, f, params, x)  # params.shape

  return ng_fn


def block_wise_natural_gradient_mse_fn(f, output_dimension, damping=0., diag_reg=0., tri=False,
                                       kernel_batch_size=32, device_count=0, store_on_device=True):

  def fl(gx):
    """Flatten loss gradient wrt fx."""
    return np.reshape(gx, (-1,))

  def ufl(gx):
    """Unflatten loss gradient wrt fx."""
    return np.reshape(gx, (-1, output_dimension))

  #TODO(kazukiosawa): support neural_tangents.batch
#  _lw_ntk_fn = nt.batch(empirical_direct_layer_wise_ntk_fn(f),
#                        kernel_batch_size, device_count, store_on_device)
  _lw_ntk_fn = empirical_direct_layer_wise_ntk_fn(f)

  def lw_ntk_fn(x1, x2, params):
    n_layers = sum([len(param) > 0 for param in params])
    kernel = _lw_ntk_fn(x1, x2, params)
    return unflatten_layer_wise_kernel(kernel, n_layers)

  _solve_w_damping = get_solver_by_damping_value(damping, diag_reg)

  if not tri:
    """Natural gradient with block-diagonal FIM."""
    def diag_ng_fn(params, x, y):
      lw_g_dd = lw_ntk_fn(x, None, params)  # L x (N, N, C, C)
      gx = fl(f(params, x) - y)  # (NC,)
      lw_vec = tree_map(
        lambda g_dd: ufl(_solve_w_damping(g_dd, gx, sym_pos=True)), lw_g_dd)  # (L, N, C)

      del lw_g_dd, gx

      return _layer_wise_vjp(lw_vec, f, params, x)  # params.shape

    return diag_ng_fn

  def _prepare_tri_diagonal(params):
    n_layers = sum([len(param) > 0 for param in params])  # L
    T = np.tri(n_layers, n_layers, 1)  # (L, L)
    T = T * T.transpose()  # (L, L)
    return T, n_layers

  if damping == 0 and diag_reg == 0:
    def tri_diag_ng_wo_damping_fn(params, x, y):
      """Natural gradient with block-tri-diagonal FIM without damping."""
      lw_g_dd = lw_ntk_fn(x, None, params)  # (L, N, N, C, C)
      gx = fl(f(params, x) - y)  # (NC,)
      T, n_layers = _prepare_tri_diagonal(params)

      ones = np.ones(n_layers)  # (L,)
      T_inv_dot_ones = solve(T, ones)  # (L,)

      lw_vec = []
      for g_dd, val in zip(lw_g_dd, T_inv_dot_ones):
        if val == 0:
          lw_vec.append(ufl(np.zeros_like(gx)))
        else:
          lw_vec.append(ufl(solve(g_dd, gx)) * val)

      del lw_g_dd, gx, T, ones, T_inv_dot_ones

      return _layer_wise_vjp(lw_vec, f, params, x)  # params.shape

    return tri_diag_ng_wo_damping_fn

  def tri_diag_ng_fn(params, x, y):
    """Natural gradient with block-tri-diagonal FIM."""
    lw_g_dd = lw_ntk_fn(x, None, params)  # (L, N, N, C, C)
    gx = fl(f(params, x) - y)  # (NC,)
    T, n_layers = _prepare_tri_diagonal(params)  # (L, L)

    n_data = y.shape[0]  # N
    I = np.eye(n_data * output_dimension)  # (NC, NC)
    T_kron_I = np.kron(T, I)  # (LNC, LNC)
    lw_g_dd = tree_map(lambda g_dd: flatten_features(g_dd), lw_g_dd)  # (L, NC, NC)
    tri_diag_gdd = T_kron_I @ sp.linalg.block_diag(*lw_g_dd)  # (LNC, LNC)

    gx = np.tile(gx, n_layers)  # (LNC,)
    vec = ufl(_solve_w_damping(tri_diag_gdd, gx))  # (LN, C)
    lw_vec = np.split(vec, n_layers)  #  (L, N, C)

    del lw_g_dd, gx, T, I, T_kron_I, tri_diag_gdd, vec

    return _layer_wise_vjp(lw_vec, f, params, x)  # params.shape

  return tri_diag_ng_fn


def unit_wise_natural_gradient_mse_fn(f, output_dimension, damping=0.):

  # TODO(kazukiosawa): support neural_tangents.batch

  def fl(gx):
    """Flatten loss gradient wrt fx."""
    return np.reshape(gx, (-1,))

  def ufl(gx):
    """Unflatten loss gradient wrt fx."""
    return np.reshape(gx, (-1, output_dimension))

  uw_ntk = empirical_direct_unit_wise_ntk_fn(f)
  _solve_w_damping = get_solver_by_damping_value(damping)

  def ng_fn(params, x, y):
    """Natural gradient with unit-wise FIM."""
    uw_g_dd = uw_ntk(x, None, params)  # L x M x (N, N, C, C)
    gx = fl(f(params, x) - y)  # (NC,)
    uw_vec = tree_map(
      lambda g_dd: ufl(_solve_w_damping(g_dd, gx, sym_pos=True)), 
      uw_g_dd)  # L x M x (N, C)

    del uw_g_dd, gx

    return _unit_wise_vjp(uw_vec, f, params, x)  # params.shape

  return ng_fn
