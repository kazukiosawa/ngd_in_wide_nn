# JAX-based implementation
The implementation uses [Neural Tangents](https://github.com/google/neural-tangents) for computing the empirical NTK. 

*NOTE: The real benefit of Neural Tangents is that it can calculate analytical NTKs efficiently, but our current implementation cannot utilize it for NGD. Our [numpy-based](../numpy-based) implementation, which has been used for the experiments in our paper, can calculate analytical NTKs of three-layered MLPs.*

## Scripts
- `predict.py`: defines exact/approximate NGD in function-space. 
- `empirical.py`: defines exact/approximate NG in parameter-space.  
- `train.py`: main script for training finite-width MLP with GD or exact/approximate NGD. 
    - `--dataset`: Dataset name ("mnist" or "cifar10").
    - `--n_classes`: Number of classes ("2","3",,,"10").
    - `--target_classes`: Target classes (e.g. "[0,7]" if `--n_classes`=2).
    - `--loss_type`: Type of loss ("mse" or "cross_entropy").
    - `--train_size`: Train set size.
    - `--test_size`: Test set size.
    - `--n_layers`: Number of layers of the MLP.
    - `--width`: Width of each layer of the MLP.
    - `--weight_variance`: Variance of init weights.
    - `--bias_variance`: Variance of init biases.
    - `--natural_gradient_type`: Type of NGD ("exact", "block_diagonal", "block_tri_diagonal", "unit_wise" or "exact_plus_mse") If not specified, GD will be performed. "exact_plus_mse" is only for "cross_entropy" loss (See Appendix D of the paper for detail).
    - `--learning_rate`: Learning rate. If not specifed, only for NGD, the optimal LR will be used.
    - `--n_steps`: Number of update steps.
    - `--damping`: Damping value to be added to the diagonal of FIM for NGD.
    - (See the source code for the default values of each argument and other arguments.)

## Example (NGD with exact Fisher information)
```console
$ python train.py --dataset mnist --n_classes 2 --n_steps 2 --natural_gradient_type exact
```
Output
```console
=====================
dataset: mnist
n_classes: 2
target_classes: [0, 1]
loss_type: mse
train_size: 100
test_size: 50
kernel_batch_size: 10
device_count: 1
n_layers: 3
width: 4096
weight_variance: 2.0
bias_variance: 0.0
natural_gradient_type: exact
n_steps: 2
learning_rate: None
damping: 0.0
store_on_device: False
dataset_path: None
save_dataset: False
save_dataset_dir: ./datasets
save_init_weights: False
save_init_weights_dir: ./weights
random_seed: 1
=====================
Creating empirical predictor for MSE...
Computing empirical prediction for 2 steps...
Training for 2 steps
step         train_accuracy  predict      train_loss   predict      test_accuracy  predict      test_loss    predict
0            0.6             0.6          0.123769     0.123769     0.48           0.48         0.130732     0.130732
1            1               1            0.000197204  2.66454e-17  1              1            0.00461683   0.00467353
2            1               1            6.32923e-08  2.66454e-17  1              1            0.0042116    0.00467353
```

- `train_accuracy`, `train_loss`, `test_accuracy`, and `test_loss` are evaluated by the finite-width MLP at each step, while the `predict` values are calculated by the empirical NTK of the initial *finite-width* MLP.
- The values at step 0 are evaluated by the initial MLP.
