''' Imports '''
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import get_images
import get_landmarks
import performance_plots

import pandas as pd
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
from sklearn.ensemble import RandomForestClassifier as rf


''' Load and Split Data '''
image_directory = 'caltech_old'
X, y = get_images.get_images(image_directory)
print(f"Loaded X shape: {len(X)}, Loaded y shape: {len(y)}")

# Single stratified split (no double-split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Get facial landmarks
X_train, y_train = get_landmarks.get_landmarks(X_train, y_train, 'landmarks/', 5, False)
X_test, y_test = get_landmarks.get_landmarks(X_test, y_test, 'landmarks/', 5, False)

'''
-----------------------------------------------------------------
'''


''' Matching and Decision - Classifier 1: kNN '''
clf_knn = ORC(knn())
clf_knn.fit(X_train, y_train)

matching_scores_knn = clf_knn.predict_proba(X_test)
classes_knn = clf_knn.classes_
matching_scores_knn = pd.DataFrame(matching_scores_knn)
matching_scores_knn.columns = range(matching_scores_knn.shape[1])

gen_scores_knn = []
imp_scores_knn = []
for i in range(len(y_test)):
    scores = matching_scores_knn.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores_knn.extend(scores[mask])
    imp_scores_knn.extend(scores[~mask])

threshold_knn = performance_plots.performance(gen_scores_knn, imp_scores_knn, 'kNN_decision_fusion', 100)

'''
-----------------------------------------------------------------
'''


''' Matching and Decision - Classifier 2: SVM '''
clf_svm = ORC(svm(probability=True))
clf_svm.fit(X_train, y_train)

matching_scores_svm = clf_svm.predict_proba(X_test)
matching_scores_svm = pd.DataFrame(matching_scores_svm)
matching_scores_svm.columns = range(matching_scores_svm.shape[1])

gen_scores_svm = []
imp_scores_svm = []
for i in range(len(y_test)):
    scores = matching_scores_svm.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores_svm.extend(scores[mask])
    imp_scores_svm.extend(scores[~mask])

threshold_svm = performance_plots.performance(gen_scores_svm, imp_scores_svm, 'SVM_decision_fusion', 100)

'''
-----------------------------------------------------------------
'''


''' Matching and Decision - Classifier 3: Random Forest '''
clf_rf = ORC(rf(n_estimators=100, random_state=42))
clf_rf.fit(X_train, y_train)

matching_scores_rf = clf_rf.predict_proba(X_test)
classes_rf = clf_rf.classes_
matching_scores_rf = pd.DataFrame(matching_scores_rf)
matching_scores_rf.columns = range(matching_scores_rf.shape[1])

gen_scores_rf = []
imp_scores_rf = []
for i in range(len(y_test)):
    scores = matching_scores_rf.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores_rf.extend(scores[mask])
    imp_scores_rf.extend(scores[~mask])

threshold_rf = performance_plots.performance(gen_scores_rf, imp_scores_rf, 'RF_decision_fusion', 100)

'''
-----------------------------------------------------------------
'''


''' Fuse decisions from all classifiers '''
correct_authentications = 0
for i in range(len(gen_scores_knn)):
    d1 = gen_scores_knn[i] >= threshold_knn
    d2 = gen_scores_svm[i] >= threshold_svm
    d3 = gen_scores_rf[i] >= threshold_rf

    if d1 and d2 and d3:
        correct_authentications += 1

for i in range(len(imp_scores_knn)):
    d1 = imp_scores_knn[i] < threshold_knn
    d2 = imp_scores_svm[i] < threshold_svm
    d3 = imp_scores_rf[i] < threshold_rf

    if d1 and d2 and d3:
        correct_authentications += 1

all_authentications = len(gen_scores_knn) + len(imp_scores_knn)
if all_authentications > 0:
    accuracy = correct_authentications / all_authentications
else:
    accuracy = 0

print(f"Final System Accuracy (fusion): {accuracy:.2%}")