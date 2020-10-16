# A PyTorch implementation for [Understanding and Exploring the Network with Stochastic Architectures](http://ml.cs.tsinghua.edu.cn/~zhijie/nsa/NSA_NIPS_camera_ready.pdf), [Zhijie Deng](http://ml.cs.tsinghua.edu.cn/~zhijie/), [Yinpeng Dong](http://ml.cs.tsinghua.edu.cn/~yinpeng/), Shifeng Zhang, and [Jun Zhu](http://ml.cs.tsinghua.edu.cn/~jun/) (NeurIPS 2020)

## Usage
### Dependencies
+ python 3
+ torch 1.4.0
+ torchvision

### Scripts for training and evaluating NSA as well as baselines:

Vanilla NSA:
```python
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long-bn-noaggr-ba --use_bn --batch_arch
```

NSA-i:
```python
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long-bn-noaggr --use_bn
```

NSA-id (only aggr)
```python
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long-bn --use_bn
```

NSA-id:
```python
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long
```

Wide resnet 28-10 in our implementation:
```python
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --dropout_rate 0. --num_ensemble 1 --decay 5e-4 --learn_aggr --arch_type residual --aux --aux_weight 0.1 --job-id resnet-r-aux0.1-long
```

Wide resnet 28-10 with MC dropout:
```python
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --dropout_rate 0.2 --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_type residual --aux --aux_weight 0.1 --job-id resnet-r-aux0.1-dp0.2-long
```

Individual training:
```python
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 1 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_1-r-p0.7-aux0.1-long
```

## Contact and cooperate
If you have any problem about this library or want to contribute to it, please send us an Email at:
- dzj17@mails.tsinghua.edu.cn

## Cite
Please cite our paper if you use this code in your own work:
```
PlaceHolder
```


