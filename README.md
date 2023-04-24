# CVRBSNet
This is the PyTorch implementation of the spatial and angular SR method in our paper "Cross-View Recurrence-based Self-Supervised Super-Resolution of Light Field". Please refer to our paper
## Preparation:
#### 1. Requirement:
* PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.6, cuda=9.0.
#### 2. Train:
* Set the hyper-parameters in `parse_args()` if needed. We have provided our default settings in the realeased codes.
* Run `train.py` to perform network training.
## Test on the datasets:
* Run `test.py` to perform test on each dataset.
* The original result files and the metric scores will be saved to `./Results/`.
