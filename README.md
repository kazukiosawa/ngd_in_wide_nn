# Natural Gradient Descent in Wide Neural Networks

Python code used in [Understanding Approximate Fisher Information for Fast Convergence of Natural Gradient Descent in Wide Neural Networks](https://arxiv.org/abs/2010.00879) (To appear in NeurIPS 2020 as an oral presentation).
This repository provides simple [JAX](https://github.com/google/jax)-/[NumPy](https://numpy.org/)-based implementations of NGD with exact/approximate Fisher Information Matrix (FIM) both in parameter-space and function-space (by empirical/analytical NTK).

![image](https://user-images.githubusercontent.com/7961228/96432037-4da79500-123f-11eb-9d13-d3666d06fa85.png)


| Code | NTK | Loss | Exact | BD | BTD | K-FAC | Unit-wise |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [jax-based](./jax-based) | empirical | MSE, cross-entropy | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\checkmark"/> | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\checkmark"/> | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\checkmark"/> | - | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\checkmark"/> |
| [numpy-based](./numpy-based) | empirical, analytical | MSE | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\checkmark"/> | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\checkmark"/> | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\checkmark"/> | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\checkmark"/> | - |

*NOTE: The NumPy-based code supports only three-layered MLP. The JAX-based code implements NGD with empirical NTK (for finite-width DNNs) on top of [Neural Tangents](https://github.com/google/neural-tangents). It supports more general DNN architectures and (multi) GPU acceleration, but it does not support NGD with analytical NTK (for infinite-width DNNs).*

## Setup
```console
$ git clone git@github.com:kazukiosawa/ngd_in_wide_nn.git
$ cd ngd_in_wide_nn
$ pip install -r requirements.txt
```
To use GPU, follow [JAX's installation guide](https://github.com/google/jax#installation).

## How to run
Visit [jax-based](./jax-based) or [numpy-based](./numpy-based) for information.

## Citation
```
@misc{karakida2020understanding,
      title={Understanding Approximate Fisher Information for Fast Convergence of Natural Gradient Descent in Wide Neural Networks}, 
      author={Ryo Karakida and Kazuki Osawa},
      year={2020},
      eprint={2010.00879},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
(To appear in NeurIPS 2020 as an oral presentation)
