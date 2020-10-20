import os
import numpy as np
import tensorflow_datasets as tfds


def create_dataset(name, n_train=None, n_test=None, do_flatten_and_normalize=True,
                   target_classes=None, save_dataset=False, save_dataset_dir='./datasets'):
  x_train, y_train, x_test, y_test = \
    _load_tf_dataset(name, n_train, n_test,
                     do_flatten_and_normalize=do_flatten_and_normalize,
                     target_classes=target_classes)
  n_classes = y_train.shape[-1]
  if n_classes == 2:
    # [1, 0] -> [0.5], [0, 1] -> [-0.5]
    y_train = np.array([[0.5 * (c1 - c2)] for c1, c2 in y_train])
    y_test = np.array([[0.5 * (c1 - c2)] for c1, c2 in y_test])

  if target_classes is None:
    target_classes = list(range(n_classes))

  # normalize ||x|| = 1
  for i in range(len(x_train)):
    x_train[i] /= np.linalg.norm(x_train[i])
  for i in range(len(x_test)):
    x_test[i] /= np.linalg.norm(x_test[i])

  if save_dataset:
    if not os.path.isdir(save_dataset_dir):
      os.makedirs(save_dataset_dir)
    filename = f'{name}_train{n_train}_test{n_test}_{n_classes}classes'
    path = os.path.join(save_dataset_dir, filename)
    np.savez(path, x_train, y_train, x_test, y_test, target_classes)

  return x_train, y_train, x_test, y_test, target_classes


def _load_tf_dataset(name, n_train=None, n_test=None, do_flatten_and_normalize=True,
                     target_classes=None, data_dir='./tensorflow_datasets'):
  """Download, parse and process a dataset to unit scale and one-hot labels."""
  ds_builder = tfds.builder(name)

  ds_train, ds_test = tfds.as_numpy(
    tfds.load(
      name,
      split=["train", "test"],
      data_dir=data_dir,
      batch_size=-1,
      as_dataset_kwargs={"shuffle_files": False}))

  train_images, train_labels, test_images, test_labels = (ds_train['image'],
                                                          ds_train['label'],
                                                          ds_test['image'],
                                                          ds_test['label'])

  num_classes = ds_builder.info.features['label'].num_classes
  if target_classes is not None:
    for target in target_classes:
      assert 0 <= target < num_classes
    num_classes = len(target_classes)

    def extract_subset(images, labels):
      new_images = []
      new_labels = []
      for image, label in zip(images, labels):
        if label in target_classes:
          idx = target_classes.index(label)
          new_images.append(image)
          new_labels.append(idx)
      return np.array(new_images), np.array(new_labels)

    train_images, train_labels = extract_subset(train_images, train_labels)
    test_images, test_labels = extract_subset(test_images, test_labels)

  if n_train is not None:
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
  if n_test is not None:
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

  if do_flatten_and_normalize:
    train_images = _partial_flatten_and_normalize(train_images)
    test_images = _partial_flatten_and_normalize(test_images)

  train_labels = _one_hot(train_labels, num_classes)
  test_labels = _one_hot(test_labels, num_classes)

  return train_images, train_labels, test_images, test_labels


def _partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def load_dataset(dataset_path, n_train, n_test, n_classes):
  npzfile = np.load(dataset_path)
  x_train, y_train = npzfile['arr_0'], npzfile['arr_1']
  x_test, y_test = npzfile['arr_2'], npzfile['arr_3']
  target_classes = npzfile['arr_4']
  assert x_train.shape[0] == y_train.shape[0] == n_train
  assert x_test.shape[0] == y_test.shape[0] == n_test

  if n_classes == 2:
    assert y_train.shape[-1] == y_test.shape[-1] == 1
  else:
    assert y_train.shape[-1] == y_test.shape[-1] == n_classes

  return x_train, y_train, x_test, y_test, target_classes
