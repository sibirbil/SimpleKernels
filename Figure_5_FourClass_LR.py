from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from time import time
import DataReader as DR
from sklearn.linear_model import LogisticRegression

def find_anchors_from_class_0(X, y):
    y_u = np.unique(y)
    points_of_class1 = X[y == y_u[1]]
    distances = []
    # find all distances
    for a in points_of_class1:
        distances.append(linalg.norm(X[y == y_u[0]] - a, axis=1, ord=1))
    # get minimum distances
    distances = np.array(distances)
    min_distance = np.min(distances, axis=0)
    min_distance = min_distance / np.std(min_distance)

    inds = []
    # get the indicies of the points whose minimum distance is greater than the lower bound
    inds.append(min_distance >= np.min(min_distance) + np.std(min_distance) * 0.5)
    inds.append(min_distance <= np.max(min_distance))
    all_inds = np.all(inds, axis=0)
    tempX = X[y == y_u[0]]
    return tempX[all_inds]


def find_anchors_from_class_1(X, y):
    y_u = np.unique(y)
    points_of_class0 = X[y == y_u[0]]
    distances = []
    # find all distances
    for a in points_of_class0:
        distances.append(linalg.norm(X[y == y_u[1]] - a, axis=1, ord=1))
    # get minimum distances
    distances = np.array(distances)
    min_distance = np.min(distances, axis=0)
    min_distance = min_distance / np.std(min_distance)
    inds = []
    # get the indicies of the points whose minimum distance is greater than the lower bound
    inds.append(min_distance >= np.min(min_distance) + np.std(min_distance) * 0.5)
    inds.append(min_distance <= np.max(min_distance))
    all_inds = np.all(inds, axis=0)
    tempX = X[y == y_u[1]]
    return tempX[all_inds]


def map_min_1_2(X, anchors):
    temp = []
    # begin: argmin
    for a in anchors[0]:
        temp.append(linalg.norm(X - a, axis=1, ord=1))
    temp = np.array(temp)
    mins1 = np.min(temp, axis=0)
    mins1 = mins1 / np.std(mins1)
    temp = []
    for a in anchors[1]:
        temp.append(linalg.norm(X - a, axis=1, ord=1))
    temp = np.array(temp)
    mins2 = np.min(temp, axis=0)
    mins2 = mins2 / np.std(mins2)
    nz = [mins1 != 0, mins2 != 0]
    d = np.all(nz, axis=0)
    mins1[mins1 == 0] = np.mean(mins1[d])
    mins2[mins2 == 0] = np.mean(mins2[d])
    # end: argmin
    X1 = np.hstack((X, mins1.reshape((len(X), 1))))
    return np.hstack((X1, mins2.reshape((len(X), 1))))


X, y = DR.fourclass()
y_unique = np.unique(y)
random_state = 42
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
cv_acc_training, cv_acc_test, cv_training_time = [], [], []

XD_k=[]
y_k=[]
# 10-fold stratified cross validation
for o_train_index, o_test_index in cv_outer.split(X, y):
    X_train_k, y_train_k, X_test_k, y_test_k = X[o_train_index], y[o_train_index], X[o_test_index], y[
        o_test_index]
    sets_of_anchors = []
    sets_of_anchors.append(find_anchors_from_class_0(X_train_k, y_train_k))
    sets_of_anchors.append(find_anchors_from_class_1(X_train_k, y_train_k))
    XD_train_k = map_min_1_2(X_train_k, sets_of_anchors)
    XD_test_k = map_min_1_2(X_test_k, sets_of_anchors)
    XD_k=XD_train_k
    y_k=y_train_k
    t_start = time()
    #clf = LinearSVC(C=1, dual=False).fit(XD_train_k, y_train_k)
    clf = LogisticRegression(C=1, solver='lbfgs', penalty='l2', dual=False, random_state=random_state).fit(XD_train_k, y_train_k)

    cv_training_time.append(round(time() - t_start, 4))
    cv_acc_training.append(accuracy_score(y_train_k, clf.predict(XD_train_k)))
    cv_acc_test.append(accuracy_score(y_test_k, clf.predict(XD_test_k)))

score_training = round(100 * np.mean(cv_acc_training), 2)
score_test = round(100 * np.mean(cv_acc_test), 2)
train_time = round(np.mean(cv_training_time), 4)
std = round(np.std(cv_acc_test), 4)



plt.scatter(XD_k[y_k == y_unique[0]][:, -2], XD_k[y_k == y_unique[0]][:, -1], c='red',
                                edgecolors='k')
plt.scatter(XD_k[y_k == y_unique[1]][:, -2], XD_k[y_k == y_unique[1]][:, -1], c='blue',
                                edgecolors='k')
plt.xlabel(r'$||x_i-a_i^{(1)}||$')
plt.ylabel(r'$||x_i-a_i^{(2)}||$')
plt.text(XD_k[:, -2].max() - 0.5, XD_k[:, -1].max() - 1,
         "Training Acc.   :  %.1f\nTest Acc.         :  %.1f\nTraining T.       :  %.3f\nStandard D.    :  %.3f" % (
             score_training, score_test, train_time, std), size=10,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

plt.title('Fourclass: Logistic_Reg_LBFGS_L2')
plt.show()
