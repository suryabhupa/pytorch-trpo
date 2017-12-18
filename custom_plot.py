import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from scipy.signal import savgol_filter

def main(filename1, filename2, gae, qe):
    if gae:
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '0_step', '1_step', '2_step', '3_step', '10_step'])
        var = np.vstack((data['0_step'], data['1_step'], data['2_step'], data['3_step'], data['10_step'])).T
    elif qe:
        data = np.genfromtxt(filename1, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '1_step', '2_step', '3_step', '10_step', 'qvalue', 'qfirst', 'qsecond'])
        var = np.vstack((data['1_step'], data['2_step'], data['3_step'], data['10_step'], data['qvalue'], data['qfirst'], data['qsecond'])).T
        data2 = np.genfromtxt(filename2, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '1_step', '2_step', '3_step', '10_step', 'qvalue', 'qfirst', 'qsecond'])
        var2 = np.vstack((data2['1_step'], data2['2_step'], data2['3_step'], data2['10_step'], data2['qvalue'], data2['qfirst'], data2['qsecond'])).T
    else:
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '0_step', 'half_step', '1_step', '2_step', '3_step', '10_step'])
        var = np.vstack((data['0_step'], data['half_step'], data['1_step'], data['2_step'], data['3_step'], data['10_step'])).T

    eps = data['episode']
    rews = data['average_reward']
    rews2 = data2['average_reward']

    if gae:
        plt.figure(figsize=(12, 4))
        ax = plt.subplot(121)
        ax.set_title('variance')
        plt.plot(eps,savgol_filter(var[:,0], 51, 5), label='0-step')
        plt.plot(eps,savgol_filter(var[:,1], 51, 5),label='1-step')
        plt.plot(eps,savgol_filter(var[:,2], 51, 5),label='2-step')
        plt.plot(eps,savgol_filter(var[:,3], 51, 5),label='3-step')
        plt.plot(eps,savgol_filter(var[:,4], 51, 5),label='10-step')
        plt.plot(eps,var[:,0], alpha=0.15)
        plt.plot(eps,var[:,1], alpha=0.15)
        plt.plot(eps,var[:,2], alpha=0.15)
        plt.plot(eps,var[:,3], alpha=0.15)
        plt.plot(eps,var[:,4], alpha=0.15)
    elif qe:
        plt.figure(figsize=(16, 6))
        ax = plt.subplot(121)
        ax.set_title('variance')
        plt.plot(eps[:500],savgol_filter(var[:500,0], 51, 5), label='1-step')
        plt.plot(eps[:500],savgol_filter(var[:500,1], 51, 5),label='2-step')
        plt.plot(eps[:500],savgol_filter(var[:500,2], 51, 5),label='3-step')
        plt.plot(eps[:500],savgol_filter(var[:500,3], 51, 5),label='10-step')
        plt.plot(eps[:500],savgol_filter(var[:500,4], 51, 5),label='qvalue')
        # plt.plot(eps,savgol_filter(var[:,5], 101, 5),label='qfirst')
        # plt.plot(eps,savgol_filter(var[:,6], 101, 5),label='qsecond')
        plt.plot(eps[:500],var[:500,0], alpha=0.15)
        plt.plot(eps[:500],var[:500,1], alpha=0.15)
        plt.plot(eps[:500],var[:500,2], alpha=0.15)
        plt.plot(eps[:500],var[:500,3], alpha=0.15)
        plt.plot(eps[:500],var[:500,4], alpha=0.15)
        # plt.plot(eps,var[:,5], alpha=0.15)
        # plt.plot(eps,var[:,6], alpha=0.15)
    else:
        plt.figure(figsize=(16, 6))
        ax = plt.subplot(121)
        ax.set_title('variance')
        plt.plot(eps,savgol_filter(var[:,0], 51, 5), label='0-step')
        plt.plot(eps,savgol_filter(var[:,1], 51, 5), label='half-step')
        plt.plot(eps,savgol_filter(var[:,2], 51, 5),label='1-step')
        plt.plot(eps,savgol_filter(var[:,3], 51, 5),label='2-step')
        plt.plot(eps,savgol_filter(var[:,4], 51, 5),label='3-step')
        plt.plot(eps,savgol_filter(var[:,5], 51, 5),label='10-step')
        plt.plot(eps,var[:,0], alpha=0.15)
        plt.plot(eps,var[:,1], alpha=0.15)
        plt.plot(eps,var[:,2], alpha=0.15)
        plt.plot(eps,var[:,3], alpha=0.15)
        plt.plot(eps,var[:,4], alpha=0.15)
        plt.plot(eps,var[:,5], alpha=0.15)

    ax = plt.subplot(122)
    ax.set_title('average rewards')
    plt.plot(eps[:500],rews[:500], label='new_rewards')
    plt.plot(eps[:500],rews2[:500], label='old_rewards')

    plt.subplot(121)
    if qe:
        plt.legend(loc='upper center', ncol=4)
    else:
        plt.legend(loc='upper center', ncol=3)
    plt.subplot(122)
    lgd = plt.legend(loc='lower center')

    plot_filename = filename.replace('logs', 'plots')
    plot_filename = plot_filename.replace('.csv', '_new.png')
    plt.savefig(plot_filename, bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError('Not enough arguments provided: need [filename] and [gae flag]')

    filename = sys.argv[1]
    filename2 = sys.argv[2]

    if sys.argv[3] == '--gae':
        gae = True
        qe = False
    elif sys.argv[3] == '--qe':
        gae = False
        qe = True
    elif sys.argv[3] == '-no-flag':
        qe = False
        gae = False
    else:
        raise ValueError('Flag must be [--gae] or [--qe] or [--no-flag]')

    main(filename, filename2, gae, qe)


