import os

import numpy as np

from absl import app
from absl import flags

from predict import *
from empirical import *
from analytical import *
from utils import create_dataset, load_dataset, forster_transform

DATASET_MNIST = 'mnist'
DATASET_CIFAR10 = 'cifar10'
NG_EXACT = 'exact'
NG_BD = 'block_diagonal'
NG_BTD = 'block_tri_diagonal'
NG_KFAC = 'kfac'


flags.DEFINE_string('dataset', DATASET_MNIST,
                    'Dataset name.')
flags.DEFINE_integer('n_classes', 2,
                     'Number of classes.')
flags.DEFINE_list('target_classes', None,
                  'target classes.')
flags.DEFINE_integer('train_size', 100,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 50,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('kernel_batch_size', 10,
                     'batch size for computing kernel.')
flags.DEFINE_integer('width', 4096,
                     'Width of each layer.')
flags.DEFINE_float('weight_variance', 2,
                   'Init weight variance.')
flags.DEFINE_string('natural_gradient_type', None,
                    'Type of natural-gradient. If None, gradient descent is performed.')
flags.DEFINE_integer('n_steps', 2,
                     'Number of steps during training.')
flags.DEFINE_float('learning_rate', None,
                   'Learning rate to use during training.')
flags.DEFINE_float('lr_factor', None,
                   'Update lr with lr * lr_factor.')
flags.DEFINE_float('damping', 0,
                   'Damping term for NGD.')
flags.DEFINE_string('dataset_path', None,
                    'If not None, dataset at specified path will be used.')
flags.DEFINE_boolean('save_dataset', False,
                     'If True, dataset for this run will be saved.'
                     'If dataset_path is not None, this will be ignored.')
flags.DEFINE_string('save_dataset_dir', './datasets',
                     'Path to save dataset.')
flags.DEFINE_string('kernel_path', None,
                    'If not None, kernels at specified path will be used.')
flags.DEFINE_boolean('save_kernel', False,
                     'If True, kernels for this run will be saved.')
flags.DEFINE_string('save_kernel_dir', './kernels',
                    'Path to save dataset.')
flags.DEFINE_integer('random_seed', 1,
                     'Random seed.')
flags.DEFINE_string('init_weights_dir', None,
                    'If not None, use W1,W2 and W3 generated'
                    'by neural-tangents as initial weights.')
flags.DEFINE_boolean('forster_transform', False,
                     'If True, apply Forster transform to dataset.')
flags.DEFINE_integer('forster_transform_n_iters', 100,
                     'Number of iterations for Forster transform.')
flags.DEFINE_float('forster_transform_tol', 1e-3,
                   'Error tolerance for Forster transform.')
FLAGS = flags.FLAGS


def main(unused_argv):
  dataset = FLAGS.dataset
  N_train = FLAGS.train_size
  N_test = FLAGS.test_size
  n_classes = FLAGS.n_classes
  target_classes = FLAGS.target_classes
  if target_classes is not None:
    target_classes = [int(cls) for cls in FLAGS.target_classes]
  else:
    target_classes = list(range(n_classes))
  ng_type = FLAGS.natural_gradient_type
  natural_gradient = ng_type is not None  # GD or NGD
  n_steps = FLAGS.n_steps
  damping = FLAGS.damping
  random_seed = FLAGS.random_seed

  dataset_path = FLAGS.dataset_path
  if dataset_path is not None:
    data = load_dataset(dataset_path, N_train, N_test, n_classes)
  else:
    data = create_dataset(dataset, N_train, N_test,
                          do_flatten_and_normalize=True,
                          target_classes=target_classes,
                          save_dataset=FLAGS.save_dataset,
                          save_dataset_dir=FLAGS.save_dataset_dir
                          )

  x_train, y_train, x_test, y_test, target_classes = data

  M = FLAGS.width
  w_var = FLAGS.weight_variance
  if n_classes == 2:
    M3 = 1  # binary classification
  else:
    M3 = n_classes
  M2 = M
  M1 = M
  M0 = x_train.shape[-1]

  learning_rate = FLAGS.learning_rate
  if learning_rate is None and natural_gradient:
    n_layers = 3
    learning_rate = get_ng_opt_lr(ng_type, n_layers, N_train)
  assert learning_rate is not None, 'Learning rate is not specified.'

  if FLAGS.lr_factor is not None:
    learning_rate *= FLAGS.lr_factor

  if ng_type == NG_KFAC and FLAGS.forster_transform:
    # Forster algorithm
    x_train = forster_transform(x_train, FLAGS.forster_transform_n_iters, FLAGS.forster_transform_tol)
    x_test = forster_transform(x_test, FLAGS.forster_transform_n_iters, FLAGS.forster_transform_tol)
    if FLAGS.save_dataset:
      if not os.path.isdir(FLAGS.save_dataset_dir):
        os.makedirs(FLAGS.save_dataset_dir)
      filename = f'{dataset}_train{N_train}_test{N_test}_{n_classes}classes_forster'
      path = os.path.join(FLAGS.save_dataset_dir, filename)
      np.savez(path, x_train, y_train, x_test, y_test, target_classes)

  init_weights_dir = FLAGS.init_weights_dir
  if init_weights_dir is None:
    W3, W2, W1, _, _, _ = \
      init_ntk_mlp(M3, M2, M1, M0,
                   W_std=np.sqrt(w_var), b_std=0, random_seed=random_seed)
  else:
    W3 = np.load(os.path.join(init_weights_dir, 'W3.npy'))
    W2 = np.load(os.path.join(init_weights_dir, 'W2.npy'))
    W1 = np.load(os.path.join(init_weights_dir, 'W1.npy'))
    w_std = np.sqrt(w_var)
    W3 = W3.transpose() * w_std / np.sqrt(M2)
    W2 = W2.transpose() * w_std / np.sqrt(M1)
    W1 = W1.transpose() * w_std / np.sqrt(M0)

  print('==================')
  print(f'dataset: {dataset}')
  print(f'target_classes {target_classes}')
  print(f'N_train: {N_train}')
  print(f'N_test: {N_test}')
  print(f'batch_size for kernel: {FLAGS.kernel_batch_size}')
  print(f'M0: {M0}')
  print(f'M1: {M1}')
  print(f'M2: {M2}')
  print(f'M3: {M3}')
  print(f'weight_variance: {w_var}')
  print(f'ng_type: {ng_type}')
  print(f'n_steps: {n_steps}')
  print(f'learning_rate: {learning_rate}')
  if natural_gradient:
    print(f'damping: {damping}')
  print(f'random_seed: {random_seed}')
  if dataset_path is not None:
    print(f'dataset_path: {dataset_path}')
  if init_weights_dir is not None:
    print(f'init_weights_dir: {init_weights_dir}')
  if ng_type == NG_KFAC and FLAGS.forster_transform:
    print(f'Forster transform iteration: {FLAGS.forster_transform_n_iters}')
    print(f'Forster transform tol: {FLAGS.forster_transform_tol}')
  if FLAGS.kernel_path is not None:
    print(f'kernel_path: {FLAGS.kernel_path}')
  print('==================')

  # MSE
  loss_fn = lambda f, y: 0.5 * np.mean(np.sum((f - y) ** 2, axis=1))
  if n_classes == 2:
    accuracy_fn = lambda f, y: np.mean(np.sign(f) == np.sign(y))  # binary
  else:
    accuracy_fn = lambda f, y: np.mean(f.argmax(axis=1) == y.argmax(axis=1))

  # Initial state
  fx0_train = forward_and_backward(x_train, W1, W2, W3, backward=False)
  fx0_test = forward_and_backward(x_test, W1, W2, W3, backward=False)

  print('Creating analytical predictor for MSE...')
  if FLAGS.kernel_path:
    kernels = np.load(FLAGS.kernel_path, allow_pickle=True).item()
    g_dd, g_td = kernels['g_dd'], kernels['g_td']
  else:
    # Calculate analytical kernels
    batch_size = FLAGS.kernel_batch_size
    if batch_size:
      g_dd = batch_analytic_kernel(x_train, x_train, w_var, ng_type, batch_size)
      g_td = batch_analytic_kernel(x_test, x_train, w_var, ng_type, batch_size)
    else:
      g_dd = analytic_kernel(x_train, x_train, w_var, ng_type)
      g_td = analytic_kernel(x_test, x_train, w_var, ng_type)

    if FLAGS.save_kernel:
      if not os.path.isdir(FLAGS.save_kernel_dir):
        os.makedirs(FLAGS.save_kernel_dir)
      filename = f'{ng_type}_{dataset}_train{N_train}_test{N_test}_{n_classes}classes'
      path = os.path.join(FLAGS.save_kernel_dir, filename)
      kernels = {'g_dd': g_dd, 'g_td': g_td}
      np.save(path, kernels)

  args = [fx0_train, y_train, g_dd, g_td]
  if ng_type == NG_KFAC:
    args.extend([x_train, x_test])
  if natural_gradient:
    predictor = natural_gradient_descent_mse(ng_type, *args)
  else:
    predictor = gradient_descent_mse(*args)

  # Get analytical predictions
  predictions = []
  print(f'Calculating analytical prediction for {n_steps} steps...')
  for t in range(n_steps + 1):
    if t == 0:
      fx_train, fx_test = fx0_train, fx0_test
    else:
      fx_train, fx_test = predictor(fx0_test, t, learning_rate)

    pred = {
      'train': {'accuracy': accuracy_fn(fx_train, y_train).tolist(),
                'loss': loss_fn(fx_train, y_train).tolist()},
      'test': {'accuracy': accuracy_fn(fx_test, y_test).tolist(),
               'loss': loss_fn(fx_test, y_test).tolist()},
    }
    predictions.append(pred)

  del g_dd, g_td, predictor

  # Prepare print report format used in Chainer
  # https://github.com/chainer/chainer/blob/v7.1.0/chainer/training/extensions/print_report.py
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

  for t in range(n_steps + 1):
    if t == 0:
      # use init values
      fx_train = fx0_train
      fx_test = fx0_test
    else:
      args = [x_train, y_train, w_var, W1, W2, W3]
      if natural_gradient:
        dW1, dW2, dW3 = natural_gradient_mse(ng_type, *args, damping)
      else:
        dW1, dW2, dW3 = gradient_mse(*args)
      dW1 = dW1.astype(np.float32)
      dW2 = dW2.astype(np.float32)
      dW3 = dW3.astype(np.float32)
      # update in weight space
      W1 = W1 - learning_rate * dW1
      W2 = W2 - learning_rate * dW2
      W3 = W3 - learning_rate * dW3

      fx_train = forward_and_backward(x_train, W1, W2, W3, backward=False)
      fx_test = forward_and_backward(x_test, W1, W2, W3, backward=False)

    train_loss = loss_fn(fx_train, y_train).tolist()
    train_accuracy = accuracy_fn(fx_train, y_train).tolist()
    test_loss = loss_fn(fx_test, y_test).tolist()
    test_accuracy = accuracy_fn(fx_test, y_test).tolist()

    pred = predictions[t]
    log = {'step': t,
           'train_accuracy': train_accuracy,
           'train_loss': train_loss,
           'test_accuracy': test_accuracy,
           'test_loss': test_loss,
           'predict': {
             'train_accuracy': pred['train']['accuracy'],
             'train_loss': pred['train']['loss'],
             'test_accuracy': pred['test']['accuracy'],
             'test_loss': pred['test']['loss']
           }}

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
