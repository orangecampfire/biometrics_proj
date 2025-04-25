import matplotlib.pyplot
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from sklearn import metrics


############# functions ##########################################################
def get_dprime(genuine_scores, impostor_scores):
        epsilon = 0.0000001
        x = np.mean(genuine_scores) - np.mean(impostor_scores)
        y = np.sqrt(0.5 * (np.var(genuine_scores) + np.var(impostor_scores)))
        return x / (y + epsilon)

def plot_det_curve(FPR, FNR, plot_title): 
        plt.figure()
        plt.plot(FPR, FNR, lw=2, color='green', label='DET Curve')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='k')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel("False Pos. Rate", fontsize = 12)    
        plt.ylabel("False Neg. Rate", fontsize = 12)
        plt.title('Detection Error Tradeoff Curve', fontsize=12, weight=5)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.savefig(f"plots/det_curve.{plot_title}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

def plot_roc_curve(FPR, TPR, plot_title):
        plt.figure()
        plt.plot(FPR, TPR, lw=2, color='green', label='ROC Curve')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='k')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel("True Pos. Rate", fontsize = 12)
        plt.xlabel("False Pos. Rate", fontsize = 12)
        plt.title(f"Receiver Operating Characteristic", fontsize = 12, weight = 5)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(f"plots/roc_curve_{plot_title}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

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
                    
        if (fp + tn) > 0:
            far.append(fp / (fp + tn))
        else:
            far.append(0)

        if (fn + tp) > 0:
            frr.append(fn / (fn + tp))
        else:
            frr.append(0)

        if (tp + fn) > 0:
            tar.append(tp / (tp + fn))
        else:
            tar.append(0)
        
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
    plt.savefig(f'plots/Score_Dist{plot_title}.png',dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    return

def performance(gen_scores, imp_scores, plot_title, num_thresholds):          
        thresholds = np.linspace(0, 1, num_thresholds)
        far, frr, tar = compute_rates(gen_scores, imp_scores, thresholds)    
        eer, eer_index, optimal_threshold = get_eer(far, frr, thresholds)
        plot_scoreDist(gen_scores, imp_scores, far, frr, eer_index, optimal_threshold, plot_title)
        plot_roc_curve(far, tar, plot_title)
        plot_det_curve(far, frr, plot_title)
        get_dprime(gen_scores, imp_scores)
        print(optimal_threshold)
        return optimal_threshold