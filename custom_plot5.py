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

    fqes = []
    vanillas = []

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for filename in filenames:
        if 'fqe' in filename:
            fqes.append(filename)
        else:
            vanillas.append(filename)

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.set_title('Factorized vs TRPO')
    for i, filename in enumerate(fqes):
        data = np.genfromtxt(filename, delimiter=',', skip_header=0, skip_footer=0, names=['episode', 'last_reward', 'average_reward'])
        eps = data['episode']
        rews = data['average_reward']
        seedlabel = filename[filename.find('_seed')+6:filename.rfind('_lr')]
        plt.plot(eps, savgol_filter(rews, 59, 5), 'b', label="seed{}".format(seedlabel))
        plt.plot(eps, rews, 'b', alpha=0.175)
    for i, filename in enumerate(vanillas):
        data = np.genfromtxt(filename, delimiter=',', skip_header=0, skip_footer=0, names=['episode', 'last_reward', 'average_reward'])
        eps = data['episode'][:600]
        rews = data['average_reward'][:600]
        seedlabel = filename[filename.find('seed-')+5:filename.find('seed-')+8]
        plt.plot(eps, savgol_filter(rews, 59, 5), 'g', label="seed{}".format(seedlabel))
        plt.plot(eps, rews, 'g', alpha=0.175)

    plt.legend(loc='lower center', ncol=5)
    axes = plt.gca()
    axes.set_ylim([-700, 5000])
    plot_filename = "plots/fqe-vs-vanilla.png"
    plt.savefig(plot_filename)

if __name__ == '__main__':
    filenames = [
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-101_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-51_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-501_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-41_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-401_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-31_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-301_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-21_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-201_lr-0.001_hid-dim-64.csv",
      "logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180109-2345_seed-11_lr-0.001_hid-dim-64.csv",
      "logs/HalfCheetah-v1_20180109-2338_seed-150.csv",
      "logs/HalfCheetah-v1_20180109-2338_seed-250.csv",
      "logs/HalfCheetah-v1_20180109-2338_seed-350.csv",
      "logs/HalfCheetah-v1_20180109-2338_seed-450.csv",
      "logs/HalfCheetah-v1_20180109-2338_seed-550.csv",
      "logs/HalfCheetah-v1_20180109-2200_['seed-100'].csv",
      "logs/HalfCheetah-v1_20180109-2200_['seed-200'].csv",
      "logs/HalfCheetah-v1_20180109-2200_['seed-300'].csv",
      "logs/HalfCheetah-v1_20180109-2200_['seed-400'].csv",
      "logs/HalfCheetah-v1_20180109-2200_['seed-500'].csv",
    ]

    main(filenames)
