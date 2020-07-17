# We want to note that, the template of this figure is taken from  the following page of scikit-learn 
#https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html#sphx-glr-auto-examples-miscellaneous-plot-kernel-approximation-py


from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn import datasets, svm, pipeline
from numpy import  linalg
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# anchor point selection for multi-class problems
def find_anchors_from_class_yi(X, y,yi):
    points_of_class_yi = X[y != yi]
    distances = []
     # find all distances
    for a in points_of_class_yi:
        distances.append(linalg.norm(X[y == yi] - a, axis=1, ord=1))
    # get minimum distances
    distances = np.array(distances)
    min_distance = np.min(distances, axis=0)
    inds = []
    # get the indicies of the points whose minimum distance is greater than the lower bound
    inds.append(min_distance >= np.percentile(min_distance,30))
    # get the indicies of the points whose minimum distance is lower than the upper bound
    inds.append(min_distance <= np.percentile(min_distance,50))
    all_inds = np.all(inds, axis=0)
    tempX = X[y == yi]
    return tempX[all_inds]

#implementation of the multi-class feture map
def map_min_M_1(X, anchorsets):
    X1=np.copy(X)
    for anchors in anchorsets:
        temp = []
        for a in anchors:
            temp.append(linalg.norm(X - a, axis=1, ord=1))
        temp = np.array(temp)
        X1 = np.hstack((X1, (np.min(temp, axis=0).reshape((len(X), 1)))))
    return X1

#get the Optical Recognition of Handwritten Digits dataset from Scikit-learn
digits = datasets.load_digits(n_class=10)
n_samples = len(digits.data)
data = digits.data / 16.
data -= data.mean(axis=0)

#split the dataset as training and testing parts
X_train,X_test,y_train,y_test=train_test_split(data, digits.target,test_size=0.3, random_state=42, stratify= digits.target, shuffle=True)

#exact rbf performance
start= time()
rbf_clf=svm.SVC(C=1,kernel='rbf', gamma=0.2).fit(X_train,y_train)
rbf_time=(time()-start)
rbf_score= accuracy_score(y_test,rbf_clf.predict(X_test))

#linear performance
start= time()
linear_clf=LinearSVC(C=1,dual=False).fit(X_train,y_train)
linear_time=(time()-start)
linear_score= accuracy_score(y_test,linear_clf.predict(X_test))


#find the anchor sets for each classes
anchors=[]
y_u=np.unique(y_train)
for yi in y_u:
    anchors.append(find_anchors_from_class_yi(X_train,y_train,yi))

# mapping performance: r'$\phi_{1,M}(x)$'
#map  the training and test parts with found anchor sets
XD_train,XD_test=map_min_M_1(X_train,anchors),map_min_M_1(X_test,anchors)
start= time()
linear_map_1_clf=LinearSVC(C=1,dual=False).fit(XD_train,y_train)
linear_map_1_time=(time()-start)
linear_map_1_score= accuracy_score(y_test,linear_map_1_clf.predict(XD_test))




# create pipeline from kernel approximation
feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", LinearSVC(C=1,dual=False))])
nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("svm", LinearSVC(C=1,dual=False))])

#number of input features
dimension=len((X_train[0]))
#sampling parameters for approximation methods
sample_sizes =30 * np.arange(1, 10)
fourier_scores = []
nystroem_scores = []
fourier_times = []
nystroem_times = []
for D in sample_sizes :
    fourier_approx_svm.set_params(feature_map__n_components=D)
    nystroem_approx_svm.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm.fit(X_train, y_train)
    nystroem_times.append(time() - start)
    start = time()
    fourier_approx_svm.fit(X_train, y_train)
    fourier_times.append(time() - start)
    fourier_score = fourier_approx_svm.score(X_test, y_test)
    nystroem_score = nystroem_approx_svm.score(X_test, y_test)
    nystroem_scores.append(nystroem_score)
    fourier_scores.append(fourier_score)


# plot the results:
plt.figure(figsize=(16, 4))
accuracy = plt.subplot(121)
# second y axis for timings
timescale = plt.subplot(122)


accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [rbf_score,rbf_score], label="Exact RBF")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [rbf_time, rbf_time], '--', label='Exact RBF')
accuracy.plot(sample_sizes, fourier_scores, label="RBF approx. with Fourier")
timescale.plot(sample_sizes, fourier_times, '--',
               label='RBF approx. with Fourier')
accuracy.plot(sample_sizes, nystroem_scores, label="RBF approx. with Nystroem")
timescale.plot(sample_sizes, nystroem_times, '--',
               label='RBF approx. with Nystroem')
accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [linear_map_1_score,linear_map_1_score], label= r'$\phi_{1,2}$')
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [linear_map_1_time, linear_map_1_time], '--', label=r'$\phi_{1,2}$')

accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [linear_score,linear_score], label="Linear")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [linear_time, linear_time], '--', label='Linear')





# vertical line for  dimensionality
accuracy.plot([dimension+len(y_u), dimension+len(y_u)], [0, 1],'--',color='black',alpha=0.3)

# legends and labels
accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
x_ticks=[dimension+len(y_u)]
x_tick_labels=['d+M']
accuracy.set_xticks(x_ticks)
accuracy.set_xticklabels(x_tick_labels)

accuracy.set_ylim(np.min(fourier_scores)-0.01, 1)

accuracy.set_xlabel("Sampling steps = transformed feature dimension")
timescale.set_xlabel("Sampling steps = transformed feature dimension")
accuracy.set_ylabel("Classification accuracy")
timescale.set_ylabel("Training time in seconds")
accuracy.legend(loc='best')
timescale.legend(loc='best')
plt.tight_layout()

#arange the decimals of the numbers 
y_tick_labels=[item.get_text()[:4] for item in accuracy.get_yticklabels()]
accuracy.set_yticklabels(y_tick_labels)

y_tick_labels=[item.get_text()[:4] for item in timescale.get_yticklabels()]
timescale.set_yticklabels(y_tick_labels)
print()



plt.show()
