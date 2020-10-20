import os

import numpy as onp
from absl import app
from absl import flags
import jax
from jax import random
from jax.api import grad
from jax.api import jit
from jax.experimental import optimizers
from jax.experimental.stax import LogSoftmax
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from utils import create_dataset, load_dataset
from functools import partial

from empirical import *
from predict import *

DATASET_MNIST = 'mnist'
DATASET_CIFAR10 = 'cifar10'

LOSS_MSE = 'mse'
LOSS_CROSS_ENTROPY = 'cross_entropy'

NG_EXACT = 'exact'
NG_BD = 'block_diagonal'
NG_BTD = 'block_tri_diagonal'
NG_UNIT = 'unit_wise'
NG_EXACT_PLUS_MSE = 'exact_plus_mse'

flags.DEFINE_string('dataset', DATASET_MNIST,
                    'Dataset name.')
flags.DEFINE_integer('n_classes', 2,
                     'Number of target classes.')
flags.DEFINE_list('target_classes', None,
                  'Target classes.')
flags.DEFINE_string('loss_type', LOSS_MSE,
                    'Loss type. mse or cross_entropy.')
flags.DEFINE_integer('train_size', 100,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 50,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('kernel_batch_size', 10,
                     'Batch size for kernel computation by Neural Tangents.')
flags.DEFINE_integer('device_count', None,
                     'Number of devices.')
flags.DEFINE_integer('n_layers', 3,
                     'Number of layers.')
flags.DEFINE_integer('width', 4096,
                     'Width of each layer.')
flags.DEFINE_float('weight_variance', 2.0,
                   'Init weight variance.')
flags.DEFINE_float('bias_variance', 0,
                   'Init bias variance.')
flags.DEFINE_string('natural_gradient_type', None,
                    'Type of natural-gradient.')
flags.DEFINE_integer('n_steps', 2,
                     'Number of steps during training.')
flags.DEFINE_float('learning_rate', None,
                   'Learning rate to use during training.')
flags.DEFINE_float('lr_factor', None,
                   'If not None, replace FLAGS.learning_rate'
                   'with FLAGS.learning_rate * FLAGS.lr_factor.')
flags.DEFINE_float('damping', 0.,
                   'Damping term for NGD.')
flags.DEFINE_boolean('store_on_device', False,
                     'If False, move kernel computed in device to cpu.')
flags.DEFINE_string('dataset_path', None,
                    'If not None, dataset at specified path will be used.')
flags.DEFINE_boolean('save_dataset', False,
                     'If True, dataset for this run will be saved.'
                     'If dataset_path is not None, this will be ignored.')
flags.DEFINE_string('save_dataset_dir', './datasets',
                    'Path to save dataset.')
flags.DEFINE_boolean('save_init_weights', False,
                     'If True, initial weights be saved.')
flags.DEFINE_string('save_init_weights_dir', './weights',
                    'Directory to save initial weights.')
flags.DEFINE_integer('random_seed', 1,
                     'Random seed.')
FLAGS = flags.FLAGS


def _accuracy(y, y_hat):
  """Compute the accuracy of the predictions with respect to one-hot labels."""
  if y_hat.shape[-1] == 1:
    # binary classification
    return np.mean(np.sign(y) == np.sign(y_hat))

  return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))


def _cross_entropy_loss():
  log_softmax_fn = partial(LogSoftmax[1], params=None)

  def loss_fn(y, y_hat):
    preds = log_softmax_fn(inputs=y)
    return -np.mean(np.sum(preds * y_hat, axis=1))

  return loss_fn


def main(unused_argv):
  # Setup device
  if FLAGS.device_count is None:
    device_count = jax.device_count()
    FLAGS.device_count = device_count
    FLAGS.kernel_batch_size = int(FLAGS.kernel_batch_size / device_count)

  # Build data pipelines.
  print('Loading data.')
  assert FLAGS.n_classes >= 2
  if FLAGS.target_classes is None:
    FLAGS.target_classes = list(range(FLAGS.n_classes))
  else:
    FLAGS.target_classes = [int(cls) for cls in FLAGS.target_classes]

  target_classes = FLAGS.target_classes
  assert len(target_classes) == FLAGS.n_classes

  dataset = FLAGS.dataset
  dataset_path = FLAGS.dataset_path
  train_size = FLAGS.train_size
  test_size = FLAGS.test_size
  n_classes = FLAGS.n_classes
  if dataset_path is not None:
    data = load_dataset(dataset_path, train_size, test_size, n_classes)
  else:
    data = create_dataset(dataset, train_size, test_size,
                          do_flatten_and_normalize=True,
                          target_classes=target_classes,
                          save_dataset=FLAGS.save_dataset,
                          save_dataset_dir=FLAGS.save_dataset_dir
                          )

  x_train, y_train, x_test, y_test, target_classes = data

  FLAGS.target_classes = target_classes
  if FLAGS.n_classes == 2:
    n_outputs = 1
  else:
    n_outputs = FLAGS.n_classes

  # Build the network
  layers = []
  assert FLAGS.n_layers > 1
  w_std = onp.sqrt(FLAGS.weight_variance)
  b_std = onp.sqrt(FLAGS.bias_variance)
  for i in range(FLAGS.n_layers - 1):
    layers.append(stax.Dense(FLAGS.width, W_std=w_std, b_std=b_std, parameterization='ntk'))
    layers.append(stax.Relu())
  layers.append(stax.Dense(n_outputs, W_std=w_std, b_std=b_std, parameterization='ntk'))

  init_fn, apply_fn, _ = stax.serial(*layers)

  key = random.PRNGKey(FLAGS.random_seed)
  _, params = init_fn(key, (-1, x_train.shape[-1]))

  if FLAGS.save_init_weights:
    save_dir = FLAGS.save_init_weights_dir
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    layer_id = 1
    for param in params:
      if len(param) > 0:
        path = os.path.join(save_dir, f'W{layer_id}.npy')
        np.save(path, param[0])
        layer_id += 1

  ng_type = FLAGS.natural_gradient_type
  natural_gradient = ng_type is not None  # GD or NGD
  if FLAGS.learning_rate is None and natural_gradient:
    n_layers = 3
    FLAGS.learning_rate = get_ng_opt_lr(ng_type, n_layers, train_size)
  assert FLAGS.learning_rate is not None, 'Learning rate is not specified.'

  if FLAGS.lr_factor is not None:
    FLAGS.learning_rate *= FLAGS.lr_factor

  learning_rate = FLAGS.learning_rate

  # Create and initialize an optimizer.
  opt_init, opt_apply, get_params = optimizers.sgd(learning_rate)
  state = opt_init(params)

  # Create a loss function.
  loss_type = FLAGS.loss_type
  if loss_type == LOSS_MSE:
    # MSE
    loss = lambda _f, _y: 0.5 * np.mean(np.sum((_f - _y) ** 2, axis=1))
  elif loss_type == LOSS_CROSS_ENTROPY:
    # Cross-entropy
    loss = _cross_entropy_loss()
  else:
    raise ValueError(f"Invalid loss type {loss_type}.")

  # Create a gradient function.
  if not natural_gradient:
    # gradient descent
    grad_fn = jit(grad(lambda params, x, y: loss(apply_fn(params, x), y)))
  else:
    # natural gradient descent
    kwargs = dict(f=apply_fn, output_dimension=n_outputs,
                  damping=FLAGS.damping,
                  kernel_batch_size=FLAGS.kernel_batch_size,
                  device_count=FLAGS.device_count, store_on_device=FLAGS.store_on_device)
    if loss_type == LOSS_CROSS_ENTROPY:
      if ng_type == NG_EXACT:
        grad_fn = natural_gradient_cross_entropy_fn(**kwargs)
      elif ng_type == NG_EXACT_PLUS_MSE:
        grad_fn = natural_gradient_cross_entropy_fn(**kwargs, plus_mse=True)
      else:
        raise ValueError(f"Invalid natural-gradient type {ng_type}.")
    else:
      if ng_type == NG_EXACT:
        grad_fn = natural_gradient_mse_fn(**kwargs)
      elif ng_type == NG_BD:
        grad_fn = block_wise_natural_gradient_mse_fn(**kwargs)
      elif ng_type == NG_BTD:
        grad_fn = block_wise_natural_gradient_mse_fn(**kwargs, tri=True)
      elif ng_type == NG_UNIT:
        grad_fn = unit_wise_natural_gradient_mse_fn(
            apply_fn, n_outputs, FLAGS.damping)
      else:
        raise ValueError(f"Invalid natural-gradient type {ng_type}.")

  # Setup config
  key_flags = FLAGS.get_key_flags_for_module(os.path.basename(__file__))
  config = {flag.name: flag.value for flag in key_flags}

  config.pop('lr_factor')

  if not natural_gradient:
    config.pop('damping')
    config.pop('natural_gradient_type')

  if ng_type in [NG_BD, NG_BTD, NG_UNIT]:
    # bd, btd, unit-wise ngd doesn't support nt.batch
    config.pop('kernel_batch_size')
    config.pop('store_on_device')
    config.pop('device_count')

  print('=====================')
  for key, val in config.items():
    print(f'{key}: {val}')
  print('=====================')

  n_steps = FLAGS.n_steps

  # Get initial values of the network in function space.
  fx0_train = apply_fn(params, x_train)
  fx0_test = apply_fn(params, x_test)

  if loss_type == LOSS_MSE:
    # Get predictions
    print('Creating empirical predictor for MSE...')
    if not natural_gradient or ng_type == NG_EXACT:
      """GD or exact NGD"""
      ntk_fn = nt.batch(nt.empirical_ntk_fn(apply_fn, trace_axes=()),
                     FLAGS.kernel_batch_size, FLAGS.device_count, FLAGS.store_on_device)
      g_dd = ntk_fn(x_train, None, params)
      g_td = ntk_fn(x_test, x_train, params)
      if not natural_gradient:
        predictor = nt.predict.gradient_descent_mse(g_dd, y_train)
      else:
        predictor = natural_gradient_descent_mse(g_dd, y_train, g_td, FLAGS.damping)
    elif ng_type in [NG_BD, NG_BTD]:
      """BD NGD or BTD NGD"""
      # TODO(kazukiosawa): support neural_tangents.batch
#      lw_ntk = nt.batch(empirical_direct_layer_wise_ntk_fn(apply_fn),
#                        FLAGS.kernel_batch_size, FLAGS.device_count, FLAGS.store_on_device)
      lw_ntk = empirical_direct_layer_wise_ntk_fn(apply_fn)
      g_dd = lw_ntk(x_train, None, params)
      g_dd = unflatten_layer_wise_kernel(g_dd, FLAGS.n_layers)
      g_td = lw_ntk(x_test, x_train, params)
      g_td = unflatten_layer_wise_kernel(g_td, FLAGS.n_layers)
      if ng_type == NG_BD:
        predictor = block_wise_natural_gradient_descent_mse(
          g_dd, y_train, g_td, damping=FLAGS.damping)
      else:
          predictor = block_wise_natural_gradient_descent_mse(
          g_dd, y_train, g_td, damping=FLAGS.damping, tri=True)
    else:
      """Unit-wise NGD"""
      uw_ntk = empirical_direct_unit_wise_ntk_fn(apply_fn)
      g_dd = uw_ntk(x_train, None, params)
      g_td = uw_ntk(x_test, x_train, params)
      predictor = unit_wise_natural_gradient_descent_mse(
        g_dd, y_train, FLAGS.width, g_td, FLAGS.damping
      )

    predictions = []
    print(f'Computing empirical prediction for {n_steps} steps...')
    for i in range(n_steps+1):
      if i == 0:
        fx_train, fx_test = fx0_train, fx0_test
      else:
        if not natural_gradient:
          train_time = learning_rate * i
          fx_train, fx_test = predictor(train_time, fx0_train, fx0_test, g_td)
        else:
          fx_train, fx_test = predictor(learning_rate, i, fx0_train, fx0_test)

      pred = {
        'train': {'accuracy': _accuracy(fx_train, y_train).tolist(),
                  'loss': loss(fx_train, y_train).tolist()},
        'test': {'accuracy': _accuracy(fx_test, y_test).tolist(),
                 'loss': loss(fx_test, y_test).tolist()},
      }
      predictions.append(pred)

    del g_dd, g_td, predictor

  else:
    predictions = None

  # Train the network.
  print(f'Training for {n_steps} steps')

  # Prepare print report format used in Chainer
  # https://github.com/chainer/chainer/blob/v7.1.0/chainer/training/extensions/print_report.py
  if predictions is None:
    entries = ['step', 'train_accuracy', 'train_loss', 'test_accuracy', 'test_loss']
  else:
    entries = ['step',
               'train_accuracy', 'predict',
               'train_loss', 'predict',
               'test_accuracy', 'predict',
               'test_loss', 'predict']
  entry_widths = [max(11, len(s)) for s in entries]
  templates = []
  for entry, w in zip(entries, entry_widths):
    templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))

  header = '  '.join(('{:%d}' % w for w in entry_widths)).format(*entries)
  print(header)

  for i in range(n_steps+1):
    if i == 0:
      # use init values
      fx_train = fx0_train
      fx_test = fx0_test
    else:
      # update params
      args = [params, x_train, y_train]
      grads = grad_fn(*args)
      state = opt_apply(i, grads, state)
      params = get_params(state)
      fx_train = apply_fn(params, x_train)
      fx_test = apply_fn(params, x_test)

    train_loss = loss(fx_train, y_train).tolist()
    train_accuracy = _accuracy(fx_train, y_train).tolist()
    test_loss = loss(fx_test, y_test).tolist()
    test_accuracy = _accuracy(fx_test, y_test).tolist()

    log = {'step': i,
           'train_accuracy': train_accuracy,
           'train_loss': train_loss,
           'test_accuracy': test_accuracy,
           'test_loss': test_loss,
           }

    if predictions is not None:
      # predict train/test
      pred = predictions[i]
      log['predict'] = {
        'train_accuracy': pred['train']['accuracy'],
        'train_loss': pred['train']['loss'],
        'test_accuracy': pred['test']['accuracy'],
        'test_loss': pred['test']['loss'],
      }

    # print report
    report = ''
    last_entry = ''
    for entry, template, empty in templates:
      if entry == 'predict':
        report += template.format(log['predict'][last_entry])
      elif entry in log:
        report += template.format(log[entry])
      else:
        report += empty
      last_entry = entry
    print(report)


if __name__ == '__main__':
  app.run(main)
