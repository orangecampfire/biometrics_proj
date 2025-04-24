''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import get_images
import get_landmarks
import performance_plots
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
import pandas as pd # type: ignore
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm


image_directory = 'C:/Users/david/Documents/USF/USF/Classes/Spring_25/Biometrics/Proj/caltech_brighter'
X, y = get_images.get_images(image_directory)
print(f"Loaded X shape: {len(X)}, Loaded y shape: {len(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

X_train, y_train = get_landmarks.get_landmarks(X_train, y_train, 'landmarks/', 5, False)
X_val, y_val = get_landmarks.get_landmarks(X_val, y_val, 'landmarks/', 5, False)
X_test, y_test = get_landmarks.get_landmarks(X_test, y_test, 'landmarks/', 5, False)

'''
-----------------------------------------------------------------
'''

''' Matching and Decision - Classifer 1 '''
clf = ORC(knn())
clf.fit(X_train, y_train)
matching_scores_knn = clf.predict_proba(X_val)

# Tuning the sytem
gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores_knn = pd.DataFrame(matching_scores_knn, columns=classes)

for i in range(len(y_val)):    
    scores = matching_scores_knn.loc[i]
    mask = scores.index.isin([y_val[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])
    
threshold_knn = performance_plots.performance(gen_scores, imp_scores, 'kNN_decision_fusion', 100)

# Testing the system - getting a decision
matching_scores_knn = clf.predict_proba(X_test)
matching_scores_knn = pd.DataFrame(matching_scores_knn, columns=classes)

gen_scores_knn = []
imp_scores_knn = []
for i in range(len(y_test)):    
    scores = matching_scores_knn.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores_knn.extend(scores[mask])
    imp_scores_knn.extend(scores[~mask])

'''
-----------------------------------------------------------------
'''

''' Matching and Decision - Classifer 2 '''
clf = ORC(svm(probability=True))
clf.fit(X_train, y_train)
matching_scores_svm = clf.predict_proba(X_val)

# Tuning the system
gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores_svm = pd.DataFrame(matching_scores_svm, columns=classes)

for i in range(len(y_val)):    
    scores = matching_scores_svm.loc[i]
    mask = scores.index.isin([y_val[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])
    
threshold_svm = performance_plots.performance(gen_scores, imp_scores, 'SVM_decision_fusion', 100)

# Testing the system - getting a decision
matching_scores_svm = clf.predict_proba(X_test)
matching_scores_svm = pd.DataFrame(matching_scores_svm, columns=classes)

gen_scores_svm = []
imp_scores_svm = []
for i in range(len(y_test)):    
    scores = matching_scores_svm.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores_svm.extend(scores[mask])
    imp_scores_svm.extend(scores[~mask])
    
'''
Fuse decisions
'''
correct_authentications = 0
for i in range(len(gen_scores_knn)):
    decision_knn = False
    decision_svm = False
    if gen_scores_knn[i] >= threshold_knn:
        decision_knn = True
        if gen_scores_svm[i] >= threshold_svm:
            decision_svm = True
            if decision_knn and decision_svm:
                correct_authentications += 1
                
for i in range(len(imp_scores_knn)):
    decision_knn = False
    decision_svm = False
    if imp_scores_knn[i] < threshold_knn:
        decision_knn = True
        if imp_scores_svm[i] < threshold_svm:
            decision_svm = True
            if decision_knn and decision_svm:
                correct_authentications += 1

all_authentications = len(gen_scores_knn) + len(imp_scores_knn)
accuracy = correct_authentications / all_authentications


print("")
print(f'Accuracy: {accuracy}')

# 1️ Feature Extraction Function
def extract_features(data):
    num_samples = data.shape[0]
    features = np.zeros((num_samples, 10))  

    for i in range(num_samples):
        landmarks = data[i]  
        pairwise_distances = euclidean_distances(landmarks, landmarks)
        features[i] = pairwise_distances[np.triu_indices(5, 1)]  

    return features

# 2️ Load Data
X_raw = np.load(r"C:/Users/david/Documents/USF/USF/Classes/Spring_25/Biometrics/CP2-finished/X-5-Caltech.npy")
y = np.load(r"C:/Users/david/Documents/USF/USF/Classes/Spring_25/Biometrics/CP2-finished/y-5-Caltech.npy")

# Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 3️ Extract Features
X = extract_features(X_raw)

# 4️ Split Data (33% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 5️ Train One-vs-Rest k-NN Classifier
clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
clf.fit(X_train, y_train)

# 6️ Compute Matching Scores (Predict Probabilities)
matching_scores = clf.predict_proba(X_test)

# 7 Separate Genuine & Impostor Scores
genuine_scores = []
impostor_scores = []

for i, label in enumerate(y_test):
    predicted_label = np.argmax(matching_scores[i])  
    max_score = np.max(matching_scores[i])  

    if predicted_label == label:
        genuine_scores.append(max_score)  
    else:
        impostor_scores.append(max_score)  

# 8️ Evaluate Performance

fpr, tpr, _ = roc_curve(y_test, matching_scores[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

def det_curve(fpr, fnr):
    plt.figure(figsize=(6, 6))
    plt.plot(norm.ppf(fpr), norm.ppf(fnr), color='red', lw=2, label="DET Curve")
    plt.xlabel("False Alarm Rate (FPR)")
    plt.ylabel("Miss Rate (FNR)")
    plt.title("Detection Error Tradeoff (DET) Curve")
    plt.legend()
    plt.grid()
    plt.show()

fnr = 1 - tpr
det_curve(fpr, fnr)

plt.figure(figsize=(6, 6))
plt.hist(genuine_scores, bins=20, alpha=0.6, label="Genuine Scores", color='blue')
plt.hist(impostor_scores, bins=20, alpha=0.6, label="Impostor Scores", color='red')
plt.xlabel("Matching Score")
plt.ylabel("Frequency")
plt.title("Distribution of Genuine and Impostor Scores")
plt.legend()
plt.show()