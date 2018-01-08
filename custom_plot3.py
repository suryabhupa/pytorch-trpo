import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from scipy.signal import savgol_filter

def main(filenames, env):

    plt.figure(figsize=(24, 6))

    seed100s = []
    seed200s = []
    seed300s = []
    seed400s = []
    rseed100s = []
    rseed200s = []
    rseed300s = []
    rseed400s = []
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for filename in filenames:
        if "HalfCheetah" in filename:
            if 'seed-100' in filename:
                seed100s.append(filename)
            elif 'seed-200' in filename:
                seed200s.append(filename)
            elif 'seed-300' in filename:
                seed300s.append(filename)
            elif 'seed-400' in filename:
                seed400s.append(filename)
        elif "Reacher" in filename:
            if 'seed-100' in filename:
                rseed100s.append(filename)
            elif 'seed-200' in filename:
                rseed200s.append(filename)
            elif 'seed-300' in filename:
                rseed300s.append(filename)
            elif 'seed-400' in filename:
                rseed400s.append(filename)

    if env == 'Reacher':
        seed100s = rseed100s
        seed200s = rseed200s
        seed300s = rseed300s
        seed400s = rseed400s
        fname_env = 'reacher'
    if env == 'HalfCheetah':
        seed100s = seed100s
        seed200s = seed200s
        seed300s = seed300s
        seed400s = seed400s
        fname_env = 'halfcheetah'

    ax = plt.subplot(141)
    ax.set_title('100')
    for i, filename in enumerate(seed100s):
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward'])
        eps = data['episode']
        rews = data['average_reward']

        if 'gae' in filename:
            label = 'gae'
            plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
            plt.plot(eps, rews, color[i], alpha=0.175)
        else:
          label = filename[filename.find('q-l2-reg')+9:filename.find('csv')-1]
          plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
          plt.plot(eps, rews, color[i], alpha=0.175)
    plt.legend(loc='lower center', ncol=4)

    ax = plt.subplot(142)
    ax.set_title('200')
    for i, filename in enumerate(seed200s):
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward'])
        eps = data['episode']
        rews = data['average_reward']

        if 'gae' in filename:
            label = 'gae'
            plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
            plt.plot(eps, rews, color[i], alpha=0.175)
        else:
            label = filename[filename.find('q-l2-reg')+9:filename.find('csv')-1]
            plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
            plt.plot(eps, rews, color[i], alpha=0.175)
    plt.legend(loc='lower center', ncol=4)

    ax = plt.subplot(143)
    ax.set_title('300')
    for i, filename in enumerate(seed300s):
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward'])
        eps = data['episode']
        rews = data['average_reward']

        if 'gae' in filename:
            label = 'gae'
            plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
            plt.plot(eps, rews, color[i], alpha=0.175)
        else:
            label = filename[filename.find('q-l2-reg')+9:filename.find('csv')-1]
            plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
            plt.plot(eps, rews, color[i], alpha=0.175)
    plt.legend(loc='lower center', ncol=4)

    ax = plt.subplot(144)
    ax.set_title('400')
    for i, filename in enumerate(seed400s):
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward'])
        eps = data['episode']
        rews = data['average_reward']

        if 'gae' in filename:
            label = 'gae'
            plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
            plt.plot(eps, rews, color[i], alpha=0.175)
        else:
            label = filename[filename.find('q-l2-reg')+9:filename.find('csv')-1]
            plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
            plt.plot(eps, rews, color[i], alpha=0.175)
    plt.legend(loc='lower center', ncol=4)

    plot_filename = "plots/{}-q-l2-reg.png".format(fname_env)
    plt.savefig(plot_filename)


if __name__ == '__main__':
    filenames = [
      "logs/qe_oracle_gae_HalfCheetah-v1_eg-nan_20180103-0654_seed-100.csv",
      "logs/qe_oracle_gae_HalfCheetah-v1_eg-nan_20180103-0654_seed-200.csv",
      "logs/qe_oracle_gae_HalfCheetah-v1_eg-nan_20180103-0654_seed-300.csv",
      "logs/qe_oracle_gae_HalfCheetah-v1_eg-nan_20180103-0654_seed-400.csv",
      "logs/qe_oracle_gae_Reacher-v1_eg-nan_20180103-0700_seed-100.csv",
      "logs/qe_oracle_gae_Reacher-v1_eg-nan_20180103-0700_seed-200.csv",
      "logs/qe_oracle_gae_Reacher-v1_eg-nan_20180103-0700_seed-300.csv",
      "logs/qe_oracle_gae_Reacher-v1_eg-nan_20180103-0700_seed-400.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-100_q-l2-reg-0.1.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-100_q-l2-reg-1.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-100_q-l2-reg-10.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-200_q-l2-reg-0.1.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-200_q-l2-reg-1.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-200_q-l2-reg-10.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-300_q-l2-reg-0.1.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-300_q-l2-reg-1.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-300_q-l2-reg-10.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-400_q-l2-reg-0.1.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-400_q-l2-reg-1.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0653_seed-400_q-l2-reg-10.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0654_seed-100_q-l2-reg-0.01.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0654_seed-100_q-l2-reg-100.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0654_seed-200_q-l2-reg-0.01.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0654_seed-200_q-l2-reg-100.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0654_seed-300_q-l2-reg-0.01.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0654_seed-300_q-l2-reg-100.0.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0654_seed-400_q-l2-reg-0.01.csv",
      "logs/qe_oracle_qe_HalfCheetah-v1_eg-nan_20180103-0654_seed-400_q-l2-reg-100.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0655_seed-100_q-l2-reg-0.1.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0655_seed-100_q-l2-reg-1.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0655_seed-200_q-l2-reg-0.1.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0655_seed-200_q-l2-reg-1.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0655_seed-300_q-l2-reg-0.1.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0655_seed-300_q-l2-reg-1.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0655_seed-400_q-l2-reg-0.1.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0655_seed-400_q-l2-reg-1.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0658_seed-100_q-l2-reg-100.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0658_seed-200_q-l2-reg-100.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0658_seed-300_q-l2-reg-100.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0658_seed-400_q-l2-reg-100.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0659_seed-100_q-l2-reg-0.01.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0659_seed-200_q-l2-reg-0.01.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0659_seed-300_q-l2-reg-0.01.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0659_seed-400_q-l2-reg-0.01.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0700_seed-100_q-l2-reg-10.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0700_seed-200_q-l2-reg-10.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0700_seed-300_q-l2-reg-10.0.csv",
      "logs/qe_oracle_qe_Reacher-v1_eg-nan_20180103-0700_seed-400_q-l2-reg-10.0.csv",
    ]

    main(filenames, 'Reacher')
    main(filenames, 'HalfCheetah')
