import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import seaborn as sns
import scipy
from scipy.signal import savgol_filter

def main(filenames):

    g05 = []
    g075 = []
    g08 = []
    vanillas = []

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    exp_names = ['0.50', '0.75', '0.80', 'none']

    for filename in filenames:
        if 'anneal-gamma' in filename:
            if '0.5' in filename:
                g05.append(filename)
            elif '0.75' in filename:
                g075.append(filename)
            elif '0.8' in filename:
                g08.append(filename)
        else:
            vanillas.append(filename)

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.set_title('Annealing gamma')
    for j, exp in enumerate([g05, g075, g08, vanillas]):
        for i, filename in enumerate(exp):
            data = np.genfromtxt(filename, delimiter=',', skip_header=0, skip_footer=0, names=['episode', 'last_reward', 'average_reward'])
            eps = data['episode'][:600]
            rews = data['average_reward'][:600]
            seedlabel = filename[filename.find('seed-')+5:filename.find('seed-')+8]
            plt.plot(eps, savgol_filter(rews, 59, 5), color[j], label="gamma_0={};seed-{}".format(exp_names[j], seedlabel))
            plt.plot(eps, rews, color[j], alpha=0.175)

    plt.legend(loc='lower center', ncol=3)
    axes = plt.gca()
    axes.set_ylim([-700, 5000])
    plot_filename = "plots/anneal-gamma.png"
    plt.savefig(plot_filename)

if __name__ == '__main__':
    filenames = [
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0811_seed-123_lr-0.05_hid-dim-64_anneal-gamma-0.5.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0811_seed-123_lr-0.05_hid-dim-64_anneal-gamma-0.75.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0811_seed-124_lr-0.05_hid-dim-64_anneal-gamma-0.5.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0811_seed-124_lr-0.05_hid-dim-64_anneal-gamma-0.75.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0811_seed-125_lr-0.05_hid-dim-64_anneal-gamma-0.5.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0811_seed-125_lr-0.05_hid-dim-64_anneal-gamma-0.75.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0812_seed-123_lr-0.05_hid-dim-64_anneal-gamma-0.8.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0812_seed-124_lr-0.05_hid-dim-64_anneal-gamma-0.8.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0812_seed-125_lr-0.05_hid-dim-64_anneal-gamma-0.8.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0822_seed-123_lr-0.05_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0822_seed-124_lr-0.05_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180110-0822_seed-125_lr-0.05_hid-dim-64.csv",
    ]

    main(filenames)
