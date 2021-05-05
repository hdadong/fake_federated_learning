# Federated Learning
<br>聚合方式实现了fed-avg[1],逆距离[5]，avg-adan优化器聚合[4]
<br>优化器的参数保留方法实现了[3]的保存本地adam优化器的方法save-adam
<br>添加噪音的方法实现了对梯度添加噪音，对模型参数添加噪音[2]。

## Requirements
- PyTorch
- NumPy

## Files
- FLModel.py: definition of the FL client and FL server class.
- MLModel.py: DNN model 
- DPMechanisms.py: generate gaussian noise


## Usag
1. Set parameters in test.py
2. Run ```python test.py``` or Execute test.ipynb to train model 

### FL model parameters
```python
client_num = 50  #设置客户端数量
aggregate_round = 200 #设置聚合次数T
noniid_label = 2    #设置每个客户端有多少个类别

    fl_param = {
    'noise':False,#是否在参数上添加噪音
    'noise2':False,#是否在梯度上添加噪音
    'ifclip':False,#是否进行参数剪切，如果要加噪音则要剪切
    'ifclip2':False,#是否进行梯度剪切，如果要加噪音则要剪切
    'whichoptimizer':2,#4:avg-adam，3:save-SGD，2:savd-adam，1:nonsave-adam，0:nonsave-SGD
    'output_size': 5,#预测类被为5个类
    'client_num':client_num,#客户端数量
    'model': MLP,#DNN神经网络
    'data': d,
    'lr': 0.0001,
    'E': 1,#本地客户端的训练epoch

    'C': 0.3,#客户端每轮的参与率
    'method':0,#0表示fedavg,1表示逆距离

    'epsilon':0,#参数的隐私预算，不添加噪音默认为0
    'delta':0,#参数的松弛项，不添加噪音默认为0
    'epsilon2': 0,#梯度的隐私预算，不添加噪音默认为0
    'delta2':0,#梯度的松弛项，不添加噪音默认为0
    'clip':  0,#参数的剪切阈值，不添加噪音默认为0
    'clip2':0,#梯度的剪切阈值，不添加噪音默认为0

    'batch_size': 256,
    'device': device#设置为GPU或CPU
    }
```

## Reference
<br>[1] McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In *Proc. Artificial Intelligence and Statistics (AISTATS)*, 2017.
<br>[2] K. Wei, J. Li, M. Ding, C. Ma, H. H. Yang, F. Farokhi, S. Jin, T. Q. S. Quek, H. V. Poor, Federated Learning with Differential Privacy: Algorithms and Performance Analysis. In *IEEE Transactions on Information Forensics & Security*, 15, pp. 3454-3469, 2020.
<br>[3] Li W. et al. (2019) Privacy-Preserving Federated Brain Tumour Segmentation. In: Suk HI., Liu M., Yan P., Lian C. (eds) Machine Learning in Medical Imaging. MLMI 2019. Lecture Notes in Computer Science, vol 11861. Springer, Cham. https://doi.org/10.1007/978-3-030-32692-0_16
<br>[4] Yu, Hao et al. “On the Linear Speedup Analysis of Communication Efficient Momentum SGD for Distributed Non-Convex Optimization.” ICML (2019).
<br>[5] Yeganeh, Yousef et al. “Inverse Distance Aggregation for Federated Learning with Non-IID Data.” DART/DCL@MICCAI (2020).
