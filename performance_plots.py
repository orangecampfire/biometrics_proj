import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from sklearn import metrics


############# functions ##########################################################

def get_eer(far, frr, thresholds):
    distances = []
    for i in range(len(far)):
        distances.append(abs(far[i] - frr[i]))
    eer_index = np.argmin(distances)
    eer = (far[eer_index] + frr[eer_index]) / 2.0
    optimal_threshold = thresholds[eer_index]
    return eer, eer_index, optimal_threshold

def compute_rates(gen_scores, imp_scores, thresholds):
    far = []
    frr = []
    tar = []
    
    for t in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for g_s in gen_scores:
            if g_s >= t:
                tp += 1
            else:
                fn += 1
                
        for i_s in imp_scores:
            if i_s >= t:
                fp += 1
            else:
                tn += 1
                    
        far.append(fp / (fp + tn))
        frr.append(fn / (fn + tp))
        tar.append(tp / (tp + fn))
        
    return far, frr, tar


def plot_scoreDist(gen_scores, imp_scores, far, frr, eer_index, optimal_threshold, plot_title):
    plt.figure()
    plt.hist(gen_scores, color='green', bins=50, density=True, lw=2, histtype='step', hatch='//', label='Genuine Scores')
    plt.hist(imp_scores, color='red', bins=50, density=True, lw=2, histtype='step', hatch='\\', label='Impostor Scores')
    plt.plot([optimal_threshold,optimal_threshold], [0, 10], '--', color="black", lw=2)
    plt.text(optimal_threshold+0.05, 10, "Score threshold, t=%.2f, at EER\nFPR=%.2f, FNR=%.2f" % (optimal_threshold, far[eer_index], frr[eer_index]), style='italic', fontsize=12, 
        bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    plt.xlim([-0.05,1.05])
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('Matching Score', fontsize = 15, weight = 'bold')
    plt.ylabel('Score Frequency', fontsize = 15, weight = 'bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Score Distribution Plot\nSystem %s' % (plot_title), fontsize = 15, weight = 'bold')
    plt.close()
    return

def performance(gen_scores, imp_scores, plot_title, num_thresholds):          
        thresholds = np.linspace(0, 1, num_thresholds)
        far, frr, tar = compute_rates(gen_scores, imp_scores, thresholds)    
        eer, eer_index, optimal_threshold = get_eer(far, frr, thresholds)
        plot_scoreDist(gen_scores, imp_scores, far, frr, eer_index, optimal_threshold, plot_title)
        print(optimal_threshold)
        return optimal_threshold