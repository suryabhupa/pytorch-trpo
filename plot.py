import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from scipy.signal import savgol_filter

def main(filename):

    data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '0_step', 'half_step', '1_step', '2_step', '3_step', '10_step'])

    eps = data['episode']
    rews = data['average_reward']
    var = np.vstack((data['0_step'], data['1_step'], data['2_step'], data['3_step'])).T

    # Interpolation code
    eps_new = np.linspace(eps.min(), eps.max(), 300)

    plt.figure(figsize=(12, 4))

    ax = plt.subplot(121)
    ax.set_title('variance')
    plt.plot(eps,savgol_filter(var[:,0], 101, 5), label='0-step')
    plt.plot(eps,savgol_filter(var[:,1], 101, 5), label='half-step')
    plt.plot(eps,savgol_filter(var[:,2], 101, 5),label='1-step')
    plt.plot(eps,savgol_filter(var[:,3], 101, 5),label='2-step')
    plt.plot(eps,savgol_filter(var[:,4], 101, 3),label='3-step')
    plt.plot(eps,savgol_filter(var[:,5], 101, 3),label='10-step')
    plt.plot(eps,var[:,0], alpha=0.15)
    plt.plot(eps,var[:,1], alpha=0.15)
    plt.plot(eps,var[:,2], alpha=0.15)
    plt.plot(eps,var[:,3], alpha=0.15)
    plt.plot(eps,var[:,4], alpha=0.15)
    plt.plot(eps,var[:,5], alpha=0.15)

    ax = plt.subplot(122)
    ax.set_title('average rewards')
    plt.plot(eps,rews, label='rewards')

    plt.subplot(121)
    plt.legend(loc='upper center', ncol=3)
    # plt.legend(loc='center right',bbox_to_anchor=(1, 0.5))
    plt.subplot(122)
    lgd = plt.legend(loc='lower center')

    plot_filename = filename.replace('logs', 'plots')
    plot_filename = plot_filename.replace('.csv', '_new.png')
    plt.savefig(plot_filename, bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Not enough arguments provided')

    filename = sys.argv[1]
    main(filename)


