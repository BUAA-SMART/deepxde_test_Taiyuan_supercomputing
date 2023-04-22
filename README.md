# How to use DeepXDE on DCU Z100


1. 进入首页，使用vscode环境，创建如下环境
![](images/%E7%95%8C%E9%9D%A2.png)

2. 使用``roc-smi``检查DCU环境

```bash
rocm-smi

==========================System Management Interface ==========================
================================================================================
DCU  Temp   AvgPwr  Fan   Perf    PwrCap  VRAM%  DCU%
0    47.0c  58.0W   0.0%  manual  450.0W    0%   0%
================================================================================
=================================End of SMI Log=================================

```


3. 在terminal中进入workspace,根据requirements安装wheel文件
```bash
cd workspace
pip3 install -r requirements.txt --no-index --find-links=file:///public/home/ghfund3_c3/deepxde_dependencies
```

4. 跑一个简单的deepxde example
```bash
python3 test_deepxde.py
```

在DCU环境下:
```
Using backend: pytorch
Other available backends: tensorflow.compat.v1, tensorflow, jax, paddle.
paddle supports more examples now and is recommended.
 
Compiling model...
'compile' took 0.000217 s

Training model...

Step      Train loss    Test loss     Test metric   
0         [2.60e-01]    [2.22e-01]    [1.03e+00]    
1000      [1.89e-04]    [2.01e-04]    [3.10e-02]    
2000      [6.77e-05]    [9.32e-05]    [2.11e-02]    
3000      [2.36e-05]    [5.11e-05]    [1.56e-02]    
4000      [7.66e-06]    [3.25e-05]    [1.25e-02]    
5000      [1.96e-05]    [4.07e-05]    [1.40e-02]    
6000      [6.76e-07]    [2.00e-05]    [9.79e-03]    
7000      [3.06e-07]    [1.94e-05]    [9.63e-03]    
8000      [1.24e-07]    [1.84e-05]    [9.38e-03]    
9000      [6.23e-08]    [1.78e-05]    [9.23e-03]    
10000     [3.49e-06]    [2.27e-05]    [1.04e-02]    

Best model at step 9000:
  train loss: 6.23e-08
  test loss: 1.78e-05
  test metric: [9.23e-03]

'train' took 23.917568 s
```

## TODO

[ ] see how to only use cpu