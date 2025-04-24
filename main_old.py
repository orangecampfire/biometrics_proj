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
from sklearn.ensemble import RandomForestClassifier as rf


image_directory = 'caltech_old'
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



'''
-----------------------------------------------------------------
'''



''' Matching and Decision - Classifier 3 '''
clf = ORC(rf(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)
matching_scores_rf = clf.predict_proba(X_val)

# Tuning the system
gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores_rf = pd.DataFrame(matching_scores_rf, columns=classes)

for i in range(len(y_val)):    
    scores = matching_scores_rf.loc[i]
    mask = scores.index.isin([y_val[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])

threshold_rf = performance_plots.performance(gen_scores, imp_scores, 'RF_decision_fusion', 100)

# Testing the system - getting a decision
matching_scores_rf = clf.predict_proba(X_test)
matching_scores_rf = pd.DataFrame(matching_scores_rf, columns=classes)

gen_scores_rf = []
imp_scores_rf = []
for i in range(len(y_test)):    
    scores = matching_scores_rf.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores_rf.extend(scores[mask])
    imp_scores_rf.extend(scores[~mask])


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
accuracy = correct_authentications / all_authentications