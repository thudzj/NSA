import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

dirs = ['snet-1_5-r-p0.7-aux0.1-long-st'] #['resnet-r-aux0.1-long-one', 'resnet-r-aux0.1-dp0.2-long', 'resnet-r-aux0.1-long', 'snet-r-p0.7-aux0.1-long-ens5', 'snet-1_5-r-p0.7-aux0.1-long', 'snet-1_5-r-p0.7-aux0.1-long-st']#

attack_method = 'robust_eps2_ns3_ss1' #'robust_eps3_ns4_ss1', robust_eps8_ns1_ss8'  # 'robust_eps4_ns5_ss1', 'ood'

for item in dirs:
    typ, xlabel = ('mis', 'Mutual information') #('entropies', 'Entropy') if 'wrnt' in item and 'drop' not in item else ('mis', 'Mutual information')
    ent_nat = np.load('/data/zhijie/snapshots_fbnn/' + item + '/{}_natural.npy'.format(typ))
    ent_adv = np.load('/data/zhijie/snapshots_fbnn/' + item + '/{}_{}.npy'.format(typ, attack_method))
    fig = plt.figure()

    # Draw the density plot
    sns.distplot(ent_nat, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'clip': (-0.01, 3)},
                 label = 'natural')
    sns.distplot(ent_adv, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'clip': (-0.01, 3)},
                 label = 'adversarial' if 'robust' in attack_method else 'OOD')

    x = np.concatenate((ent_nat, ent_adv), 0)
    y = np.zeros(x.shape[0])
    y[ent_nat.shape[0]:] = 1

    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    auc = metrics.auc(fpr, tpr)
    print(item, auc)
    
    # Plot formatting
    plt.legend(prop={'size': 16})
    # plt.title('')
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.tight_layout()
    plt.savefig('pdfs/' + item + '_' + attack_method + ".pdf")
