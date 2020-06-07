'''
python main_s.py --arch wrn_r --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --job-id snet-1_10-r-p0.5 --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.5
Prec@1 96.950 Prec@5 99.940 Error@1 3.050 Loss 0.10400 ECE 0.00876

python main_s.py --arch wrn_r --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --job-id snet-1_10-r-p0.5-aux --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.5 --aux
Prec@1 96.960 Prec@5 99.940 Error@1 3.040 Loss 0.09887 ECE 0.00719

python main_s.py --arch wrn_r --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --job-id snet-1_10-r-p0.7-aux --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.7 --aux
Prec@1 96.920 Prec@5 99.930 Error@1 3.080 Loss 0.10406 ECE 0.00464


python main_s.py --arch wrn_r --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_10-r-p0.7-aux0.1
Prec@1 97.030 Prec@5 99.920 Error@1 2.970 Loss 0.09594 ECE 0.00334


python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.7 --aux --aux_weight 0 --job-id snet-1_10-r-p0.7-aux0-long
Prec@1 97.110 Prec@5 99.950 Error@1 2.890 Loss 0.09588 ECE 0.00493

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.7 --aux --aux_weight 0.4 --job-id snet-1_10-r-p0.7-aux0.4-long
Prec@1 97.110 Prec@5 99.950 Error@1 2.890 Loss 0.09703 ECE 0.00448

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.7 --aux --aux_weight 0.1 --label_smooth_rate 0 --job-id snet-1_10-r-p0.7-aux0.1-long
Prec@1 97.030 Prec@5 99.920 Error@1 2.970 Loss 0.09501 ECE 0.00505

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 7 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_7-r-p0.7-aux0.1-long
Prec@1 97.140 Prec@5 99.960 Error@1 2.860 Loss 0.09617 ECE 0.00462
snet-1_7-r-p0.7-aux0.1-long_2: Prec@1 97.090 Prec@5 99.940 Error@1 2.910 Loss 0.09331 ECE 0.00326

    snet-1_7-r-p0.7-aux0.1-noaggr-long: Prec@1 96.930 Prec@5 99.960 Error@1 3.070 Loss 0.09973 ECE 0.00601

    python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 7 --arch_p 0.7 --aux --aux_weight 0.1 --se --job-id snet-1_7-r-p0.7-aux0.1-se-long
    Prec@1 96.980 Prec@5 99.960 Error@1 3.020 Loss 0.09915 ECE 0.00318

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long
Prec@1 97.250 Prec@5 99.970 Error@1 2.750 Loss 0.09261 ECE 0.00250
snet-1_5-r-p0.7-aux0.1-long_2: Prec@1 97.240 Prec@5 99.940 Error@1 2.760 Loss 0.09622 ECE 0.00340

    python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1  --job-id snet-1_5-r-p0.7-aux0.1-noaggr-long
    Prec@1 97.130 Prec@5 99.970 Error@1 2.870 Loss 0.09290 ECE 0.00396

    python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --se --job-id snet-1_5-r-p0.7-aux0.1-se-long
    Prec@1 97.020 Prec@5 99.950 Error@1 2.980 Loss 0.09728 ECE 0.00432

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 3 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_3-r-p0.7-aux0.1-long
Prec@5 99.940 Error@1 3.040 Loss 0.10078 ECE 0.00687

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.4 --job-id snet-1_5-r-p0.7-aux0.4-long
Prec@1 96.900 Prec@5 99.940 Error@1 3.100 Loss 0.10313 ECE 0.00553

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.3  --job-id snet-1_5-r-p0.7-aux0.3-long 
Prec@1 96.920 Prec@5 99.930 Error@1 3.080 Loss 0.09749 ECE 0.00563

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.2  --job-id snet-1_5-r-p0.7-aux0.2-long
Prec@1 97.070 Prec@5 99.940 Error@1 2.930 Loss 0.09258 ECE 0.00415

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.2  --job-id snet-1_5-r-p0.7-aux0.2-noaggr-long
Prec@1 97.070 Prec@5 99.960 Error@1 2.930 Loss 0.09534 ECE 0.00540 

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0 --job-id snet-1_5-r-p0.7-aux0-long
Prec@1 96.980 Prec@5 99.930 Error@1 3.020 Loss 0.09817 ECE 0.00343


python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --dropout_rate 0. --num_ensemble 1 --decay 5e-4 --learn_aggr --arch_type residual --aux --aux_weight 0.1 --job-id resnet-r-aux0.1-long
aux0: Prec@1 97.120 Prec@5 99.940 Error@1 2.880 Loss 0.11414 ECE 0.01322
aux0.1: Prec@1 97.070 Prec@5 99.910 Error@1 2.930 Loss 0.11806 ECE 0.01396
    resnet-r-aux0.1-long_2: Prec@1 96.970 Prec@5 99.910 Error@1 3.030 Loss 0.11651 ECE 0.01474
    resnet-r-aux0.1-long_3: Prec@1 96.960 Prec@5 99.930 Error@1 3.040 Loss 0.11922 ECE 0.01656
    resnet-r-aux0.1-long_4: Prec@1 97.130 Prec@5 99.960 Error@1 2.870 Loss 0.11186 ECE 0.01411
    resnet-r-aux0.1-long_5: Prec@1 97.160 Prec@5 99.940 Error@1 2.840 Loss 0.11396 ECE 0.01424
aux0.2: Prec@1 96.980 Prec@5 99.960 Error@1 3.020 Loss 0.11608 ECE 0.01505
aux0.3: Prec@1 97.040 Prec@5 99.900 Error@1 2.960 Loss 0.11971 ECE 0.01512
aux0.4: Prec@1 96.880 Prec@5 99.910 Error@1 3.120 Loss 0.12347 ECE 0.01611

python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --dropout_rate 0.2 --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_type residual --aux --aux_weight 0.1 --job-id resnet-r-aux0.1-dp0.2-long
aux0: Prec@1 96.820 Prec@5 99.940 Error@1 3.180 Loss 0.10463 ECE 0.00930
aux0.1: Prec@1 96.810 Prec@5 99.950 Error@1 3.190 Loss 0.10929 ECE 0.01073
aux0.2: Prec@1 96.960 Prec@5 99.930 Error@1 3.040 Loss 0.10838 ECE 0.00969
aux0.3: Prec@1 96.900 Prec@5 99.900 Error@1 3.100 Loss 0.10059 ECE 0.00886
aux0.4: Prec@1 96.970 Prec@5 99.960 Error@1 3.030 Loss 0.10073 ECE 0.00787


___________________________
python main_s.py --arch wrn_r --dataset cifar100 --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-cifar100-r-p0.7-aux0.1-long
Prec@1 83.560 Prec@5 96.200 Error@1 16.440 Loss 0.64582 ECE 0.02125
    
    snet-1_5-cifar100-r-p0.7-aux0-long  Prec@1 82.100 Prec@5 96.100 Error@1 17.900 Loss 0.69392 ECE 0.01779
    snet-1_5-cifar100-r-p0.8-aux0.1-long  Prec@1 83.250 Prec@5 96.250 Error@1 16.750 Loss 0.61372 ECE 0.01661
    snet-1_5-cifar100-r-p0.7-aux0.1-se-long  Prec@1 83.030 Prec@5 96.380 Error@1 16.970 Loss 0.64559 ECE 0.01550
    snet-1_5-cifar100-r-p0.7-aux0.4-long  Prec@1 83.490 Prec@5 96.280 Error@1 16.510 Loss 0.62980 ECE 0.01986

python main_s.py --arch wrn_r --dataset cifar100 --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 7 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_7-cifar100-r-p0.7-aux0.1-long
Prec@1 83.200 Prec@5 96.280 Error@1 16.800 Loss 0.64165 ECE 0.01580

python main_s.py --arch wrn_r --dataset cifar100 --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_10-cifar100-r-p0.7-aux0.1-long
Prec@1 83.140 Prec@5 96.430 Error@1 16.860 Loss 0.66258 ECE 0.02528


python main_s.py --dataset cifar100 --arch wrn_r --epochs 300 --schedule 90 180 240 --dropout_rate 0. --num_ensemble 1 --decay 5e-4 --learn_aggr --arch_type residual --aux --aux_weight 0 --job-id resnet-cifar100-r-aux0-long
aux0: Prec@1 82.200 Prec@5 95.750 Error@1 17.800 Loss 0.73996 ECE 0.06709
aux0.1: Prec@1 83.410 Prec@5 96.200 Error@1 16.590 Loss 0.70953 ECE 0.06993
    resnet-cifar100-r-aux0.1-long_2: Prec@1 83.250 Prec@5 96.090 Error@1 16.750 Loss 0.71454 ECE 0.06724
    resnet-cifar100-r-aux0.1-long_3: Prec@1 83.260 Prec@5 96.060 Error@1 16.740 Loss 0.72409 ECE 0.07022
    resnet-cifar100-r-aux0.1-long_4: Prec@1 83.300 Prec@5 96.060 Error@1 16.700 Loss 0.70501 ECE 0.06792
    resnet-cifar100-r-aux0.1-long_5: Prec@1 83.720 Prec@5 96.070 Error@1 16.280 Loss 0.69466 ECE 0.06574
aux0.4: Prec@1 82.880 Prec@5 96.060 Error@1 17.120 Loss 0.73086 ECE 0.07369


python main_s.py --dataset cifar100 --arch wrn_r --epochs 300 --schedule 90 180 240 --dropout_rate 0.2 --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_type residual --aux --aux_weight 0 --job-id resnet-cifar100-r-aux0-dp0.2-long
aux0: Prec@1 82.150 Prec@5 96.040 Error@1 17.850 Loss 0.69415 ECE 0.04611
aux0.1: Prec@1 82.750 Prec@5 96.100 Error@1 17.250 Loss 0.65676 ECE 0.04602
aux0.4: Prec@1 83.210 Prec@5 96.240 Error@1 16.790 Loss 0.65963 ECE 0.04635


-----------------------
individual training:
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 1 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_1-r-p0.7-aux0.1-long
1_1: Prec@1 97.020 Prec@5 99.900 Error@1 2.980 Loss 0.11332 ECE 0.01514
2_2: Prec@1 97.010 Prec@5 99.930 Error@1 2.990 Loss 0.11415 ECE 0.01490
3_3: Prec@1 97.090 Prec@5 99.930 Error@1 2.910 Loss 0.11693 ECE 0.01530 
4_4: Prec@1 97.210 Prec@5 99.920 Error@1 2.790 Loss 0.10795 ECE 0.01556
5_5: Prec@1 96.810 Prec@5 99.940 Error@1 3.190 Loss 0.12665 ECE 0.01570

python main_s.py --arch wrn_r --dataset cifar100 --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 1 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_1-cifar100-r-p0.7-aux0.1-long
1_1: Prec@1 83.070 Prec@5 96.510 Error@1 16.930 Loss 0.67912 ECE 0.03539
2_2: Prec@1 83.390 Prec@5 96.330 Error@1 16.610 Loss 0.66182 ECE 0.03028
3_3: Prec@1 82.470 Prec@5 95.810 Error@1 17.530 Loss 0.75377 ECE 0.08710
4_4: Prec@1 82.880 Prec@5 96.090 Error@1 17.120 Loss 0.66784 ECE 0.02814
5_5: Prec@1 83.070 Prec@5 96.020 Error@1 16.930 Loss 0.70251 ECE 0.04211



--------------------
learn arch predictor

entropy of every arch's id:
    python main_s.py --arch wrn_r_d --dataset cifar100 --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-cifar100-long-sm --gradient_est sm --arch_reg 1.
        Prec@1 84.020 Prec@5 96.550 Error@1 15.980 Loss 0.65805 ECE 0.02875

    python main_s.py --arch wrn_r_d --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long-st --gradient_est st --arch_reg 1.
        Prec@1 97.330 Prec@5 99.950 Error@1 2.670 Loss 0.09172 ECE 0.00280

    # python main_s.py --dataset cifar100 --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-cifar100-long-st-cha --gradient_est st --arch_reg 1.
    #     Prec@1 83.620 Prec@5 96.450 Error@1 16.380 Loss 0.68221 ECE 0.03767

    # python main_s.py --epochs 300 --schedule 90 180 240 --num_ensemble 100 --learn_aggr --aux --aux_weight 0.1 --arch wrn_r_d --dataset cifar100 --job-id snet-1_5-rd-p0.7-aux0.1-long-ent-cha-cifar100-gsm --gradient_est gsm
    #     Prec@1 83.950 Prec@5 96.500 Error@1 16.050 Loss 0.64283 ECE 0.04298

    # python main_s.py --epochs 300 --schedule 90 180 240 --num_ensemble 100 --learn_aggr --aux --aux_weight 0.1 --arch wrn_r_d --job-id snet-1_5-rd-p0.7-aux0.1-long-ent-cha-gsm --gradient_est gsm
    #     Prec@1 97.160 Prec@5 99.960 Error@1 2.840 Loss 0.09975 ECE 0.00632

entropy of arch logits:
python main_s.py --epochs 300 --schedule 90 180 240 --num_ensemble 100 --learn_aggr --aux --aux_weight 0.1 --arch wrn_r_d  --gradient_est st --arch_reg 0.3 --job-id snet-final-1512-st0.3
    Prec@1 97.320 Prec@5 99.950 Error@1 2.680 Loss 0.09649 ECE 0.00438


----------------------------------
div:
no cbn:
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_10-r-p0.7-aux0.1-long-bn --use_bn
    snet-1_5-r-p0.7-aux0.1-long-bn: Prec@1 96.850 Prec@5 99.940 Error@1 3.150 Loss 0.11022 ECE 0.00997
    snet-1_10-r-p0.7-aux0.1-long-bn: Prec@1 96.810 Prec@5 99.940 Error@1 3.190 Loss 0.10593 ECE 0.00959
    snet-1_20-r-p0.7-aux0.1-long-bn: Prec@1 96.650 Prec@5 99.960 Error@1 3.350 Loss 0.11461 ECE 0.01055
    snet-1_50-r-p0.7-aux0.1-long-bn: Prec@1 96.740 Prec@5 99.930 Error@1 3.260 Loss 0.10796 ECE 0.00715
    snet-1_100-r-p0.7-aux0.1-long-bn: Prec@1 96.470 Prec@5 99.940 Error@1 3.530 Loss 0.11755 ECE 0.00826
    snet-1_500-r-p0.7-aux0.1-long-bn: Prec@1 96.650 Prec@5 99.960 Error@1 3.350 Loss 0.11347 ECE 0.00442
    snet-1_5000-r-p0.7-aux0.1-long-bn: Prec@1 96.350 Prec@5 99.910 Error@1 3.650 Loss 0.11540 ECE 0.00676
    snet-1_50000-r-p0.7-aux0.1-long-bn: Prec@1 96.460 Prec@5 99.960 Error@1 3.540 Loss 0.11450 ECE 0.00728

no cbn no aggr:
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 10 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_10-r-p0.7-aux0.1-long-bn-noaggr --use_bn
    snet-1_5-r-p0.7-aux0.1-long-bn-noaggr: Prec@1 96.850 Prec@5 99.960 Error@1 3.150 Loss 0.10826 ECE 0.01113
    snet-1_10-r-p0.7-aux0.1-long-bn-noaggr: Prec@1 96.740 Prec@5 99.930 Error@1 3.260 Loss 0.10855 ECE 0.01147
    snet-1_20-r-p0.7-aux0.1-long-bn-noaggr: Prec@1 96.810 Prec@5 99.950 Error@1 3.190 Loss 0.10840 ECE 0.00889
    snet-1_50-r-p0.7-aux0.1-long-bn-noaggr: Prec@1 96.880 Prec@5 99.950 Error@1 3.120 Loss 0.10604 ECE 0.00781
    snet-1_100-r-p0.7-aux0.1-long-bn-noaggr: Prec@1 96.420 Prec@5 99.910 Error@1 3.580 Loss 0.11908 ECE 0.01098
    snet-1_500-r-p0.7-aux0.1-long-bn-noaggr: Prec@1 96.490 Prec@5 99.960 Error@1 3.510 Loss 0.11293 ECE 0.01028
    snet-1_5000-r-p0.7-aux0.1-long-bn-noaggr: Prec@1 96.500 Prec@5 99.980 Error@1 3.500 Loss 0.11583 ECE 0.00925
    snet-1_50000-r-p0.7-aux0.1-long-bn-noaggr: Prec@1 96.110 Prec@5 99.910 Error@1 3.890 Loss 0.15388 ECE 0.01774


ours:
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_seed_start 1 --arch_seed_end 20 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_20-r-p0.7-aux0.1-long
    snet-1_20-r-p0.7-aux0.1-long: Prec@1 96.860 Prec@5 99.940 Error@1 3.140 Loss 0.10215 ECE 0.00655
    snet-1_50-r-p0.7-aux0.1-long: Prec@1 96.560 Prec@5 99.950 Error@1 3.440 Loss 0.12123 ECE 0.02059
    snet-1_100-r-p0.7-aux0.1-long: Prec@1 95.730 Prec@5 99.900 Error@1 4.270 Loss 0.15482 ECE 0.03530
    snet-1_500-r-p0.7-aux0.1-long: Prec@1 76.500 Prec@5 99.030 Error@1 23.500 Loss 0.85378 ECE 0.22221


batch arch, no cbn, no aggr
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long-bn-noaggr-ba --use_bn --batch_arch
    snet-1_5-r-p0.7-aux0.1-long-bn-noaggr-ba: Prec@1 96.690 Prec@5 99.900 Error@1 3.310 Loss 0.21391 ECE 0.11525
    snet-1_50-r-p0.7-aux0.1-long-bn-noaggr-ba:  Prec@1 93.660 Prec@5 99.930 Error@1 6.340 Loss 0.25240 ECE 0.07954
    snet-1_500-r-p0.7-aux0.1-long-bn-noaggr-ba: Prec@1 95.870 Prec@5 99.940 Error@1 4.130 Loss 0.17421 ECE 0.06005
    snet-1_5000-r-p0.7-aux0.1-long-bn-noaggr-ba: Prec@1 95.720 Prec@5 99.910 Error@1 4.280 Loss 0.17990 ECE 0.06477
    snet-1_50000-r-p0.7-aux0.1-long-bn-noaggr-ba: Prec@1 95.760 Prec@5 99.940 Error@1 4.240 Loss 0.17670 ECE 0.06432


train/val mode
CUDA_VISIBLE_DEVICES=0 python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 50000 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_50000-r-p0.7-aux0.1-long-bn-noaggr-ba --use_bn --batch_arch --evaluate_arch train --resume /data/zhijie/snapshots_fbnn/snet-1_50000-r-p0.7-aux0.1-long-bn-noaggr-ba/checkpoint.pth.tar --num_ensemble 100 --evaluate_arch eval


unseen 

CUDA_VISIBLE_DEVICES=1 python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --num_ensemble 100 --decay 5e-4 --arch_seed_start 1 --arch_seed_end 51000 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_10-r-p0.7-aux0.1-long-bn-noaggr --use_bn  --resume /data/zhijie/snapshots_fbnn2/snet-1_10-r-p0.7-aux0.1-long-bn-noaggr/checkpoint.pth.tar --num_ensemble 100 --evaluate_unseen

ens
python main_s.py --arch wrn_r --epochs 300 --schedule 90 180 240 --droppath_rate 0. --decay 5e-4 --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --aux --aux_weight 0.1 --job-id snet-1_5-r-p0.7-aux0.1-long --learn_aggr  --resume /data/zhijie/snapshots_fbnn/snet-1_5-r-p0.7-aux0.1-long/checkpoint.pth.tar --num_ensemble 5 --evaluate_ens

'''


from __future__ import division
import os, sys, shutil, time, random, math, copy
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, Cutout
import models
import numpy as np
import random
from utils import _ECELoss

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for Networks with Soft Sharing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str, default='data', help='Path to dataset')
parser.add_argument('--dataset', metavar='DSET', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between CIFAR/ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='wrn_r', help='model architecture: ' + ' | '.join(model_names) + ' (default: wide resnet)')
parser.add_argument('--width', type=int, metavar='N', default=10)
parser.add_argument('--num_nodes', type=int, metavar='N', default=8)
parser.add_argument('--use_bn', action='store_true', default=False)
parser.add_argument('--affine', dest='affine', action='store_true', help='Enable learnable affine in bn')
parser.add_argument('--track_running_stats', dest='track_running_stats', action='store_true', help='Enable track_running_stats in bn')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=200, help='Number of epochs to train.') #300
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--lr_scheduler', type=str, default='step', help='Lr_scheduler.')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160], help='Decrease learning rate at these epochs.') #[90, 180, 240]
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2], help='LR is multiplied by gamma on schedule')

#Regularization
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--cutout', dest='cutout', action='store_true', help='Enable cutout augmentation')
parser.add_argument('--dropout_rate', type=float, default=0., help='dropout_rate.')
parser.add_argument('--droppath_rate', type=float, default=0.)
parser.add_argument('--learn_aggr', action='store_true', default=False)
parser.add_argument('--aux', action='store_true', default=False)
parser.add_argument('--aux_weight', type=float, default=0.)
parser.add_argument('--label_smooth_rate', type=float, default=0.)

# Checkpoints
parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_fbnn/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on test set')
parser.add_argument('--evaluate_arch', type=str, default=None)
parser.add_argument('--evaluate_unseen', action='store_true', default=False)
parser.add_argument('--evaluate_ens', action='store_true', default=False)

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 8)')

# Random seed
parser.add_argument('--manualSeed', type=int, default='1', help='manual seed')
parser.add_argument('--job-id', type=str, default='')

# arch_type
parser.add_argument('--arch_type', default='random', type=str)
parser.add_argument('--arch_seed_start', type=int, default=1)
parser.add_argument('--arch_seed_end', type=int, default=5)
parser.add_argument('--arch_p', type=float, default=0.7)
parser.add_argument('--gradient_est', default=None, type=str)
parser.add_argument('--arch_reg', type=float, default=1.)
parser.add_argument('--batch_arch', action='store_true', default=False)

# test settings
parser.add_argument('--num_ensemble', type=int, default=1)

args = parser.parse_args()
args.cutout = True
args.affine = True
args.track_running_stats = True
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
job_id = args.job_id
args.save_path = args.save_path + job_id
result_png_path = os.path.join(args.save_path, 'curve.png')
result_cal_path = os.path.join(args.save_path, 'cal.pdf')
    
out_str = str(args)
print(out_str)

if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

best_acc = 0
num_stages = 3

def load_dataset():
    if args.dataset == 'cifar10':
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100
        num_classes = 100
    elif args.dataset != 'imagenet': assert False, "Unknown dataset : {}".format(args.dataset)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
        if args.cutout: train_transform.transforms.append(Cutout(n_holes=1, length=16))
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        train_data = dataset(args.data_path, train=True, transform=train_transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

        test_data = dataset(args.data_path, train=False, transform=test_transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'imagenet':
        import imagenet_seq
        train_loader = imagenet_seq.data.Loader('train', batch_size=args.batch_size, num_workers=args.workers)
        test_loader = imagenet_seq.data.Loader('val', batch_size=args.batch_size, num_workers=args.workers)
        num_classes = 1000
    else: assert False, 'Do not support dataset : {}'.format(args.dataset)

    return num_classes, train_loader, test_loader


def load_model(num_classes, log):
    net = models.__dict__[args.arch](args, num_classes)
    net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(args.ngpu)))
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([p.numel() for p in trainable_params])
    print_log("Number of parameters: {}".format(params), log)
    return net

def main():
    global best_acc

    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}{}.txt'.format(args.manualSeed, '_eva' if args.evaluate or args.evaluate_arch or args.evaluate_unseen or args.evaluate_ens else '')), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)

    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)

    num_classes, train_loader, test_loader = load_dataset()

    net = load_model(num_classes, log)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.label_smooth_rate > 0.:
        criterion_train = LabelSmoothingLoss(num_classes, args.label_smooth_rate)
    else:
        criterion_train = criterion
    params = group_weight_decay(net, state['decay'])
    optimizer = torch.optim.SGD(params, state['learning_rate'], momentum=state['momentum'], nesterov=(state['momentum'] > 0.0))

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == 'auto': args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            if args.evaluate_unseen:
                net.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if 'adj' not in k}, strict=False)
            else:
                net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint", log)

    if args.evaluate:
        validate(test_loader, net, criterion, log, args.num_ensemble)
        return

    if args.evaluate_arch:
        validate_archs(test_loader, net, criterion, log, args.num_ensemble, args.evaluate_arch)
        return

    if args.evaluate_unseen:
        validate_unseen(test_loader, net, criterion, log, args.num_ensemble)
        return

    if args.evaluate_ens:
        validate_ens(test_loader, net, criterion, log, args.num_ensemble)

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, train_los)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        train_acc, train_los = train(train_loader, net, criterion_train, optimizer, epoch, log)

        val_acc, val_los   = validate(test_loader, net, criterion, log, 1 if epoch < args.epochs -1 else args.num_ensemble)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc    

        save_dict = {
          'epoch': epoch + 1,
          'state_dict': net.state_dict(),
          'recorder': recorder,
          'optimizer' : optimizer.state_dict(),
        }
        save_checkpoint(save_dict, False, args.save_path, 'checkpoint.pth.tar')
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)
    log.close()


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    arch_prob = []

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        if args.aux:
            if not args.gradient_est is None:
                output, output_aux, arch_logits = model(input, True)
            else:
                output, output_aux = model(input)
        else:
            if not args.gradient_est is None:
                output, arch_logits = model(input, True)
            else:
                output = model(input)

        loss = criterion(output, target)
        if args.aux:
            loss += args.aux_weight * criterion(output_aux, target)
        if not args.gradient_est is None:
            loss += args.arch_reg * (arch_logits.softmax(-1) * arch_logits.log_softmax(-1)).sum(1).mean(0)
            arch_prob.append(arch_logits.softmax(-1).sum(0).data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
    
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i == len(train_loader)-1:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.avg:.3f}   '
                        'Data {data_time.avg:.3f}   '
                        'Loss {loss.avg:.4f}   '
                        'Prec@1 {top1.avg:.3f}   '
                        'Prec@5 {top5.avg:.3f}   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, 
                        loss=losses, top1=top1, top5=top5) + time_string(), log)
            if len(arch_prob) > 0: print_log(np.array_repr(np.stack(arch_prob).sum(0)/50000.), log)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log, num_ensemble):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    if args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
              m.train()

    entropies = []
    logits = []
    labels = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            output = 0
            for j in range(num_ensemble):
                output += model(input).softmax(-1)
            output = output.div(num_ensemble).log()
            # output = model(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            logits.append(output)
            labels.append(target)
            entropies.append((- output * output.exp()).sum(1))

        entropies = torch.cat(entropies, 0).data.cpu().numpy()
        ece = _ECELoss()(torch.cat(logits, 0), torch.cat(labels, 0), result_cal_path).item()

    np.save(args.save_path + "/logits.npy", torch.cat(logits, 0).data.cpu().numpy())
    np.save(args.save_path + "/entropies.npy", entropies)

    print_log('  **Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss {losses.avg:.5f} ECE {ece:.5f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses, ece=ece), log)
    return top1.avg, losses.avg

def validate_archs(val_loader, model, criterion, log, num_archs, typ):

    if typ == 'eval':
        model.eval()
        if args.dropout_rate > 0.:
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                  m.train()
    else:
        model.train()

    loses = []
    accs = []
    with torch.no_grad():
        for arch in range(num_archs):
            losses = AverageMeter()
            top1 = AverageMeter()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                output = model(input, arch)
                if typ == 'train': output=output[0]
                loss = criterion(output, target)
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
            loses.append(losses.avg)
            accs.append(top1.avg)
            print(loses[-1], accs[-1])
    np.savez(os.path.join(args.save_path, 'random_predicts_{}'.format(typ)), loss=loses, acc=accs)

def validate_unseen(val_loader, model, criterion, log, num_archs):

    model.eval()
    if args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
              m.train()

    rng = np.random.RandomState(0)

    num_seens = int(args.resume.split('_')[-1].split('-')[0])
    print(num_seens)
    if num_seens > 100:
        archs = rng.permutation(500)[:100]
    else:
        archs = rng.choice(np.arange(num_seens), 100)
    print(archs.min(), archs.max())
    loses = []
    accs = []
    with torch.no_grad():
        for arch in archs:
            losses = AverageMeter()
            top1 = AverageMeter()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                output = model(input, arch)
                loss = criterion(output, target)
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
            loses.append(losses.avg)
            accs.append(top1.avg)
            print(arch, loses[-1], accs[-1])
    np.savez(os.path.join(args.save_path, 'seen_predicts'), loss=loses, acc=accs)

    archs = rng.permutation(np.arange(50500, 51000))[:100]
    print(archs.min(), archs.max())
    loses = []
    accs = []
    with torch.no_grad():
        for arch in archs:
            losses = AverageMeter()
            top1 = AverageMeter()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                output = model(input, arch)
                loss = criterion(output, target)
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
            loses.append(losses.avg)
            accs.append(top1.avg)
            print(arch, loses[-1], accs[-1])
    np.savez(os.path.join(args.save_path, 'unseen_predicts'), loss=loses, acc=accs)

def validate_ens(val_loader, model, criterion, log, num_archs):
    model.eval()
    if args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
              m.train()

    loses, accs = [], []
    enss = []
    with torch.no_grad():
        ens = torch.zeros(10000, 10).cuda()
        targets = torch.zeros(10000).cuda()
        for arch in range(num_archs):
            losses = AverageMeter()
            top1 = AverageMeter()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                output = model(input, arch)
                ens[i*args.batch_size:i*args.batch_size+input.shape[0]] += output.softmax(-1)
                targets[i*args.batch_size:i*args.batch_size+input.shape[0]] = target
                loss = criterion(output, target)
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
            loses.append(losses.avg)
            accs.append(top1.avg)
            enss.append((ens.max(1)[1] == targets).float().mean().item())
            print(loses[-1], accs[-1], enss[-1])
    print(np.mean(loses), np.mean(accs))
    print((ens.max(1)[1] == targets).float().mean())
    np.save(args.save_path + "/ens_acc_list.npy", np.array(enss))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule, loss):
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step): lr = lr * gamma
        else: break
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    return lr

def group_weight_decay(net, weight_decay, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue
        if sum([pattern in name for pattern in skip_list]) > 0: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__': main()
