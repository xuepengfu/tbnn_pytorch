# tbnn_pytorch

The fluid data can be obtained from kinds of various sources ($.dat$, $.cas$ or $.csv$). The prepocessing codes need to be written specifically for the data format.

The codes here are shown after prepocessing. The prepocessing procudure can be found in Ling et al.(2016) https://github.com/tbnn/tbnn.

traindata_all.mat is the data with anisotropyRS_all, invarants_5 and Tensorbasis (after preprocess)

/model: different MLP model for training.

option.py: change hyperparemeters.

exe.py: training Neural Networks with different hyperparameters.

Code is develped based on：

- Python 3.8.13
- Pytorch 1.12.1
- numpy 1.23.3
