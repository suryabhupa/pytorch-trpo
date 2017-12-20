import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from scipy.signal import savgol_filter

def main(filename, gae, qe, freq, limit):
    if gae:
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '0_step', '1_step', '2_step', '3_step', '10_step'])
        var = np.vstack((data['0_step'], data['1_step'], data['2_step'], data['3_step'], data['10_step'])).T
    elif qe:
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '0_step', '1_step', '2_step', '3_step', '10_step', 'qvalue', 'qfirst', 'qsecond'])
        var = np.vstack((data['0_step'], data['1_step'], data['2_step'], data['3_step'], data['10_step'], data['qvalue'], data['qfirst'], data['qsecond'])).T
    elif qae:
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '0_step', '1_step', '2_step', '3_step', '10_step', 'qvalue', 'qevalue'])
        var = np.vstack((data['0_step'], data['1_step'], data['2_step'], data['3_step'], data['10_step'], data['qvalue'], data['qevalue'])).T
    else:
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '0_step', 'half_step', '1_step', '2_step', '3_step', '10_step'])
        var = np.vstack((data['0_step'], data['half_step'], data['1_step'], data['2_step'], data['3_step'], data['10_step'])).T

    eps = data['episode']
    rews = data['average_reward']

    if gae:
        plt.figure(figsize=(12, 4))
        ax = plt.subplot(121)
        ax.set_title('variance')
        plt.plot(eps,savgol_filter(var[:,0], freq, 5), label='0-step')
        plt.plot(eps,savgol_filter(var[:,1], freq, 5), label='1-step')
        plt.plot(eps,savgol_filter(var[:,2], freq, 5), label='2-step')
        plt.plot(eps,savgol_filter(var[:,3], freq, 5), label='3-step')
        plt.plot(eps,savgol_filter(var[:,4], freq, 5), label='10-step')
        plt.plot(eps,var[:,0], alpha=0.15)
        plt.plot(eps,var[:,1], alpha=0.15)
        plt.plot(eps,var[:,2], alpha=0.15)
        plt.plot(eps,var[:,3], alpha=0.15)
        plt.plot(eps,var[:,4], alpha=0.15)
    elif qe:
        plt.figure(figsize=(16, 6))
        ax = plt.subplot(121)
        ax.set_title('variance')
        plt.plot(eps[:limit],savgol_filter(var[:limit,0], freq, 5), label='0-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,1], freq, 5), label='1-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,2], freq, 5),label='2-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,3], freq, 5),label='3-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,4], freq, 5),label='10-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,5], freq, 5),label='qvalue')
        # plt.plot(eps,savgol_filter(var[:,5], 101, 5),label='qfirst')
        # plt.plot(eps,savgol_filter(var[:,6], 101, 5),label='qsecond')
        plt.plot(eps[:limit],var[:limit,0], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,1], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,2], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,3], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,4], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,5], alpha=0.15)
        # plt.plot(eps,var[:,5], alpha=0.15)
        # plt.plot(eps,var[:,6], alpha=0.15)
    elif qae:
        plt.figure(figsize=(16, 6))
        ax = plt.subplot(121)
        ax.set_title('variance')
        plt.plot(eps[:limit],savgol_filter(var[:limit,0], freq, 5), label='0-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,1], freq, 5), label='1-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,2], freq, 5),label='2-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,3], freq, 5),label='3-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,4], freq, 5),label='10-step')
        plt.plot(eps[:limit],savgol_filter(var[:limit,5], freq, 5),label='qvalue')
        plt.plot(eps[:limit],savgol_filter(var[:limit,6], freq, 5),label='qevalue')
        plt.plot(eps[:limit],var[:limit,0], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,1], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,2], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,3], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,4], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,5], alpha=0.15)
        plt.plot(eps[:limit],var[:limit,6], alpha=0.15)
    else:
        plt.figure(figsize=(16, 6))
        ax = plt.subplot(121)
        ax.set_title('variance')
        plt.plot(eps,savgol_filter(var[:,0], freq, 5), label='0-step')
        plt.plot(eps,savgol_filter(var[:,1], freq, 5), label='half-step')
        plt.plot(eps,savgol_filter(var[:,2], freq, 5),label='1-step')
        plt.plot(eps,savgol_filter(var[:,3], freq, 5),label='2-step')
        plt.plot(eps,savgol_filter(var[:,4], freq, 5),label='3-step')
        plt.plot(eps,savgol_filter(var[:,5], freq, 5),label='10-step')
        plt.plot(eps,var[:,0], alpha=0.15)
        plt.plot(eps,var[:,1], alpha=0.15)
        plt.plot(eps,var[:,2], alpha=0.15)
        plt.plot(eps,var[:,3], alpha=0.15)
        plt.plot(eps,var[:,4], alpha=0.15)
        plt.plot(eps,var[:,5], alpha=0.15)

    ax = plt.subplot(122)
    ax.set_title('average rewards')
    plt.plot(eps,rews, '.',label='rewards')

    plt.subplot(121)
    if qe:
        plt.legend(loc='upper center', ncol=4)
    else:
        plt.legend(loc='upper center', ncol=3)
    plt.subplot(122)
    lgd = plt.legend(loc='lower center')

    plot_filename = filename.replace('logs', 'plots')
    if limit == -1:
        plot_filename = plot_filename.replace('.csv', '_new.png')
    elif limit != -1:
        plot_filename = plot_filename.replace('.csv', '_l' + str(limit) + '_new.png')
    plt.savefig(plot_filename, bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError('Not enough arguments provided: need [filename] and [gae flag]')

    filename = sys.argv[1]

    if sys.argv[2] == '--gae':
        gae = True
        qe = False
        qae = False
    elif sys.argv[2] == '--qe':
        gae = False
        qe = True
        qae = False
    elif sys.argv[2] == '--qae':
        gae = False
        qe = False
        qae = True
    elif sys.argv[2] == '-no-flag':
        qe = False
        gae = False
        qae = False
    else:
        raise ValueError('Flag must be [--gae] or [--qe] or [--qae] or [--no-flag]')

    try:
        if sys.argv[3] != "":
            freq = int(sys.argv[3][2:])
    except:
        freq = 51

    try:
        if sys.argv[4] != "":
            limit = int(sys.argv[4][2:])
    except:
        limit = -1

    main(filename, gae, qe, freq, limit)


