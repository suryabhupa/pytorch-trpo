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

def main(filenames, env, lr, alll=False):

    seed100s = []
    seed200s = []
    seed300s = []
    seed400s = []
    seed500s = []
    rseed100s = []
    rseed200s = []
    rseed300s = []
    rseed400s = []
    rseed500s = []
    list_of_lrs = [0.1, 0.05, 0.01, 0.005, 0.001]
    lr1, lr2, lr3, lr4, lr5 = [], [], [], [], []
    alr1, alr2, alr3, alr4, alr5 = [], [], [], [], []

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    all_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in all_colors.items())
    sorted_names = [name for hsv, name in by_hsv][::-1]


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
            elif 'seed-500' in filename:
                seed500s.append(filename)
            if 'lr-0.1' in filename:
                lr1.append(filename)
            elif 'lr-0.05' in filename:
                lr2.append(filename)
            elif 'lr-0.01' in filename:
                lr3.append(filename)
            elif 'lr-0.005' in filename:
                lr4.append(filename)
            elif 'lr-0.001' in filename:
                lr5.append(filename)
        elif "Ant" in filename:
            if 'seed-100' in filename:
                rseed100s.append(filename)
            elif 'seed-200' in filename:
                rseed200s.append(filename)
            elif 'seed-300' in filename:
                rseed300s.append(filename)
            elif 'seed-400' in filename:
                rseed400s.append(filename)
            elif 'seed-500' in filename:
                rseed500s.append(filename)
            if 'lr-0.1' in filename:
                alr1.append(filename)
            elif 'lr-0.05' in filename:
                alr2.append(filename)
            elif 'lr-0.01' in filename:
                alr3.append(filename)
            elif 'lr-0.005' in filename:
                alr4.append(filename)
            elif 'lr-0.001' in filename:
                alr5.append(filename)

    if env == 'Ant':
        seed100s = rseed100s
        seed200s = rseed200s
        seed300s = rseed300s
        seed400s = rseed400s
        seed500s = rseed500s
        lr1 = alr1
        lr2 = alr2
        lr3 = alr3
        lr4 = alr4
        lr5 = alr4
        fname_env = 'ant'
    if env == 'HalfCheetah':
        seed100s = seed100s
        seed200s = seed200s
        seed300s = seed300s
        seed400s = seed400s
        seed500s = seed500s
        lr1 = lr1
        lr2 = lr2
        lr3 = lr3
        lr4 = lr4
        lr5 = lr5
        fname_env = 'halfcheetah'

    if alll:
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111)
        ax.set_title(env)
        for j, seeds in enumerate([seed100s, seed200s, seed300s, seed400s, seed500s]):
            for i, filename in enumerate(seeds):
                data = np.genfromtxt(filename, delimiter=',', skip_header=0, skip_footer=0, names=['episode', 'last_reward', 'average_reward'])
                eps = data['episode']
                rews = data['average_reward']
                lrlabel = filename[filename.find('_lr')+4:filename.rfind('csv')-1]
                seedlabel = filename[filename.find('_seed')+6:filename.rfind('_lr')]
                plt.plot(eps, savgol_filter(rews, 59, 5), all_colors[sorted_names[i+5*j]], label="lr{}seed{}".format(lrlabel, seedlabel))
                plt.plot(eps, rews, color[i%len(color)], alpha=0.175)
                plt.legend(loc='lower center', ncol=5)
            axes = plt.gca()
            axes.set_ylim([-700, 5000])
        plot_filename = "plots/{}-all-seed-lr.png".format(fname_env)
        plt.savefig(plot_filename)
        return

    if not lr and not alll:
        plt.figure(figsize=(24, 6))
        for j, seeds in enumerate([seed100s, seed200s, seed300s, seed400s, seed500s]):
            subplot_num = 100 + 50 + j + 1
            title = 'seed {}00'.format(j+1)
            ax = plt.subplot(subplot_num)
            ax.set_title(title)
            for i, filename in enumerate(seeds):
                data = np.genfromtxt(filename, delimiter=',', skip_header=0, skip_footer=0, names=['episode', 'last_reward', 'average_reward'])
                eps = data['episode']
                rews = data['average_reward']

                label = filename[filename.find('_lr')+4:filename.rfind('csv')-1]
                plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
                plt.plot(eps, rews, color[i], alpha=0.175)
                plt.legend(loc='lower center', ncol=3)
            axes = plt.gca()
            axes.set_ylim([-700, 5000])
        plot_filename = "plots/{}-seed-fixed-lr.png".format(fname_env)
        plt.savefig(plot_filename)

    elif lr and not alll:
        plt.figure(figsize=(24, 6))
        for j, lrs in enumerate([lr1, lr2, lr3, lr4, lr5]):
            subplot_num = 100 + 50 + j + 1
            title = 'lr {}'.format(list_of_lrs[j])
            ax = plt.subplot(subplot_num)
            ax.set_title(title)
            for i, filename in enumerate(lrs):
                data = np.genfromtxt(filename, delimiter=',', skip_header=0, skip_footer=0, names=['episode', 'last_reward', 'average_reward'])
                eps = data['episode']
                rews = data['average_reward']

                label = filename[filename.find('_seed')+6:filename.rfind('_lr')]
                plt.plot(eps, savgol_filter(rews, 59, 5), color[i], label=label)
                plt.plot(eps, rews, color[i], alpha=0.175)
                plt.legend(loc='lower center', ncol=3)
            axes = plt.gca()
            axes.set_ylim([-700, 5000])

        plot_filename = "plots/{}-seed-lr-fixed.png".format(fname_env)
        plt.savefig(plot_filename)

if __name__ == '__main__':
    filenames = [
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2033_seed-100_lr-0.1.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2033_seed-200_lr-0.1.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2033_seed-300_lr-0.1.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2033_seed-400_lr-0.1.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2033_seed-500_lr-0.1.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-100_lr-0.005.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-100_lr-0.01.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-100_lr-0.05.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-200_lr-0.005.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-200_lr-0.01.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-200_lr-0.05.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-300_lr-0.005.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-300_lr-0.01.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-300_lr-0.05.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-400_lr-0.005.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-400_lr-0.01.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-400_lr-0.05.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-500_lr-0.005.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-500_lr-0.01.csv",
			"logs/qe_oracle_fqe_Ant-v1_eg-nan_20180108-2034_seed-500_lr-0.05.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-100_lr-0.001.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-100_lr-0.005.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-100_lr-0.01.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-100_lr-0.05.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-100_lr-0.1.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-200_lr-0.001.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-200_lr-0.005.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-200_lr-0.01.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-200_lr-0.05.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-200_lr-0.1.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-300_lr-0.001.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-300_lr-0.005.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-300_lr-0.01.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-300_lr-0.05.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-300_lr-0.1.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-400_lr-0.001.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-400_lr-0.005.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-400_lr-0.01.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-400_lr-0.05.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-400_lr-0.1.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-500_lr-0.001.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-500_lr-0.005.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-500_lr-0.01.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-500_lr-0.05.csv",
			"logs/qe_oracle_fqe_HalfCheetah-v1_eg-nan_20180108-2033_seed-500_lr-0.1.csv",
    ]

    print('ant, lr=true')
    main(filenames, 'Ant', lr=True)
    print('ant, lr=false')
    main(filenames, 'Ant', lr=False)
    print('ant, all')
    main(filenames, 'Ant', lr=False, alll=True)
    print('half, lr=true')
    main(filenames, 'HalfCheetah', lr=True)
    print('half, lr=false')
    main(filenames, 'HalfCheetah', lr=False)
    print('half, all')
    main(filenames, 'HalfCheetah', lr=False, alll=True)
