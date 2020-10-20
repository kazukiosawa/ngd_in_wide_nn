# NumPy-based implementation
Supports only three-layered MLPs (three fully-connected layers).

## Scripts
- `predict.py`: defines GD and exact/approximate NGD in function-space. 
- `analytical.py`: defines analytical NTK computation.
- `empirical.py`: defines empirical NTK computation, forward/backward of MLP, and exact/approximate NG in parameter-space.  
- `train.py`: main script for training finite/infinite-width MLP with GD or exact/approximate NGD. 
    - `--dataset`: Dataset name ("mnist" or "cifar10").
    - `--n_classes`: Number of classes ("2","3",,,"10").
    - `--target_classes`: Target classes (e.g. "[0,7]" if `--n_classes`=2).
    - `--train_size`: Train set size.
    - `--test_size`: Test set size.
    - `--width`: Width of each layer of the MLP.
    - `--weight_variance`: Variance of init weights. NOTE: Biases are 0 for this script.
    - `--natural_gradient_type`: Type of NGD ("exact", "block_diagonal", "block_tri_diagonal", or "kfac"). If not specified, GD will be performed.
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
==================
dataset: mnist
target_classes [0, 1]
N_train: 100
N_test: 50
batch_size for kernel: 10
M0: 784
M1: 4096
M2: 4096
M3: 1
weight_variance: 2.0
ng_type: exact
n_steps: 2
learning_rate: 1.0
damping: 0.0
random_seed: 1
==================
Creating analytical predictor for MSE...
Calculating analytical prediction for 2 steps...
step         train_accuracy  predict      train_loss   predict      test_accuracy  predict      test_loss    predict
0            0.36            0.36         0.134341     0.134341     0.48           0.48         0.128435     0.128435
1            1               1            0.000203484  1.30195e-16  1              1            0.00422796   0.00436911
2            1               1            1.35276e-07  1.30195e-16  1              1            0.00401106   0.00436911
```

- `M0` is the input dimension, `M1` and `M2` are hidden layer dimensions, and `M3` is the output dimension (=1 for binary classification) of the MLP.  
- `train_accuracy`, `train_loss`, `test_accuracy`, and `test_loss` are evaluated by the finite-width MLP at each step, while the `predict` values are calculated by the analytical NTK of the initial *infinite-width* MLP.
- The values at step 0 are evaluated by the initial MLP.
