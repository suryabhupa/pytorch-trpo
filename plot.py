import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main(filename):

    colors = sns.color_palette("hls", 3)

    data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward', '0_step', '1_step', '2_step', '3_step'])

    eps = data['episode']
    rews = data['average_reward']
    var = np.vstack((data['0_step'], data['1_step'], data['2_step'], data['3_step'])).T

    plt.figure(figsize=(12, 4))

    ax = plt.subplot(121)
    ax.set_title('variance')
    plt.plot(eps,var[:,0], label='0-step')
    plt.plot(eps,var[:,1], label='1-step')
    plt.plot(eps,var[:,2], label='2-step')
    plt.plot(eps,var[:,3], label='3-step')

    ax = plt.subplot(122)
    ax.set_title('average rewards')
    plt.plot(eps,rews, label='rewards')

    plt.subplot(121)
    plt.legend(loc='lower center')
    plt.subplot(122)
    lgd = plt.legend(loc='lower center')

    plot_filename = filename.replace('logs', 'plots')
    plot_filename = plot_filename.replace('.csv', '.png')
    plt.savefig(plot_filename, bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Not enough arguments provided')

    filename = sys.argv[1]
    main(filename)


