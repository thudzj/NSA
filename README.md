env
```
python3, torch1.4.0, torchvision
```

NSA-id with 5 architectures:
```
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long
```

Wide resnet 28-10 in our implementation:
```
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --dropout_rate 0. --num_ensemble 1 --decay 5e-4 --learn_aggr --arch_type residual --aux --aux_weight 0.1 --job-id resnet-r-aux0.1-long
```


Wide resnet 28-10 with MC dropout:
```
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --dropout_rate 0.2 --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_type residual --aux --aux_weight 0.1 --job-id resnet-r-aux0.1-dp0.2-long
```

Individual training:
```
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 1 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_1-r-p0.7-aux0.1-long
```


Vanilla NSA:
```
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long-bn-noaggr-ba --use_bn --batch_arch
```

NSA-i:
```
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long-bn-noaggr --use_bn
```

NSA-id (only aggr)
```
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long-bn --use_bn
```

Semi-supervised learning:
```
python main_semi.py --num_ensemble 100 --dataset cifar10 --arch wrn_r --dropout_rate 0.  --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --job-id snet-semi-1-ul20-decay5e-4 --epochs 100 --schedule 40 80 --decay 5e-4 --un_weight 20  --gammas 0.2 0.2 
```
