import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from scipy.signal import savgol_filter

def main(filenames):

    plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)

    for filename in filenames:
        data = np.genfromtxt(filename, delimiter=',', skip_header=2, skip_footer=1, names=['episode', 'last_reward', 'average_reward']) 
        eps = data['episode']
        rews = data['average_reward']

        ax.set_title('average rewards')
        if "gae_Reacher" in filename:
            color = 'r'
            label = 'gae'
        elif "qe_Reacher" in filename:
            color = 'b'
            label = 'qe'
        elif "qae_Reacher" in filename:
            color = 'g'
            label = 'qae'
 
        if "seed-10" in filename:
            plt.plot(eps, savgol_filter(rews, 9, 5), color, label=label)

        plt.plot(eps, rews, color, alpha=0.20)

    plt.subplot(111)
    lgd = plt.legend(loc='lower center', ncol=3)

    plot_filename = "plots/reacher_rews.png"
    plt.savefig(plot_filename, bbox_extra_artists=(lgd,))


if __name__ == '__main__':
    filenames = [
        "logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-10.csv",
        "logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-20.csv",
        "logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-30.csv",
        "logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-40.csv", 
        "logs/qe_oracle_gae_Reacher-v1_eg-nan_20171220-1910_seed-50.csv",
        "logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-10.csv",
        "logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-20.csv",
        "logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-30.csv",
        "logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-40.csv",
        "logs/qe_oracle_qe_Reacher-v1_eg-nan_20171220-1911_seed-50.csv",
        "logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-10.csv",
        "logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-20.csv",
        "logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-30.csv",
        "logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-40.csv",
        "logs/qe_oracle_qae_Reacher-v1_eg-nan_20171220-1911_seed-50.csv",
    ]

    main(filenames)


