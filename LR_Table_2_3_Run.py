from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC  # Linear SVM for feture maps
from sklearn.svm import SVC  # Dual SVM for Kernels
from time import time
import numpy as np
import FeatureMaps as maps
import DataReader as DR
from sklearn.linear_model import LogisticRegression
#all datasets
datasets=[DR.Splice, DR.Wilt,DR.Guide1, DR.Spambase, DR.Phoneme, DR. Magic, DR.Adult ]
#all dataset names
data_names=['Splice','Wilt', 'Guide 1', 'Spambase', 'Phoneme','Magic','Adult' ]
# set mapping funtions
mapping_functions = [maps.linear, maps.phi_p_1, maps.phi_p_1, maps.phi_p_d, maps.phi_p_d]

# set p values according to the mapping functions in respected order in given Table 1
# p value of the linear kernel is used as a dummy variable and others in respected order in given Table 1
p_vals = [0, 1, 2, 1, 2]

# set mapping function an kernel names  in respected order in given Table 1
kernel_names = ['LIN', r'$\phi_{1,1}$', r'$\phi_{2,1}$', r'$\phi_{1,d}$', r'$\phi_{2,d}$']

# set of POL  kernel parameters
d_params = [2, 3, 4]
# set  of RBF kernel parameters
g_params = np.power(10.0, range(-5, 5))

# set of  C parameters
c_params = np.power(10.0, range(-5, 5))

# random state for splitting
random_state = 42

# a dictionary to store  all results
results = {'Dataset': [], 'Training Acc.': [], 'Test Acc.': [], 'Training Time': []}

for i in range(len(datasets)):
    results['Dataset'].append(data_names[i])
    X_train, X_test, y_train, y_test = datasets[i]()
    # set dictionaries to store performance metrics
    over_all_performance_metrics = {'Kernel': [], 'Training Acc': [], 'Test Acc': [], 'Training Time': []}
    # 2-fold cross validation object
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

    for m in range(len(mapping_functions)):
        over_all_performance_metrics['Kernel'].append(kernel_names[m])
        acc_by_param = []
        # begin: grid search
        for ci in c_params:
            all_acc = []
            # begin: two-fold
            for inner_train_index, inner_test_index in inner_cv.split(X_train, y_train):
                # begin: scaling
                scaler = StandardScaler()
                X_inner_train = scaler.fit_transform(X_train[inner_train_index])
                X_inner_test = scaler.transform(X_train[inner_test_index])
                # end: scaling
                y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
                #clf = LinearSVC(C=ci, dual=False).fit(mapping_functions[m](X_inner_train, p=p_vals[m]), y_inner_train)
                clf=LogisticRegression(C=ci, solver='lbfgs', penalty='l2', dual=False).fit(mapping_functions[m](X_inner_train, p=p_vals[m]), y_inner_train)
                all_acc.append(
                    accuracy_score(y_inner_test, clf.predict(mapping_functions[m](X_inner_test, p=p_vals[m]))))
            # end: two-fold
            acc_by_param.append(np.mean(all_acc))
        # end: grid search
        # get best hyperparameters
        best_c = c_params[np.argmax(acc_by_param)]
        scaler = StandardScaler()
        # being: scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # end: scaling
        s = time()
        XD_train = mapping_functions[m](X_train_scaled, p=p_vals[m])
        #clf = LinearSVC(C=best_c, dual=False).fit(XD_train, y_train)
        clf = LogisticRegression(C=best_c, solver='lbfgs', penalty='l2', dual=False).fit(XD_train, y_train)
        over_all_performance_metrics['Training Time'].append(round(time() - s, 4))
        over_all_performance_metrics['Training Acc'].append(
            round(100 * accuracy_score(y_train, clf.predict(XD_train)), 2))
        over_all_performance_metrics['Test Acc'].append(
            round(100 * accuracy_score(y_test, clf.predict(mapping_functions[m](X_test_scaled, p=p_vals[m]))), 2))


    results['Training Time'].append(over_all_performance_metrics['Training Time'])
    results['Training Acc.'].append(over_all_performance_metrics['Training Acc'])
    results['Test Acc.'].append(over_all_performance_metrics['Test Acc'])

print('***** Table 2: Test Accuracies')
n_datasets = len(datasets)
n_kenels = len(kernel_names)
for d in range(n_datasets):
    print()
    print('***' + results['Dataset'][d])
    temp = ''
    for k in range(n_kenels):
        temp += kernel_names[k] + '\t' + str(results['Test Acc.'][d][k]) + '\t' + '\t'
    print(temp)

print('***** Table 3: Training Times in Seconds')
n_datasets = len(datasets)
n_kenels = len(kernel_names)
for d in range(n_datasets):
    print()
    print('***' + results['Dataset'][d])
    temp = ''
    for k in range(n_kenels):
        temp += kernel_names[k] + '\t' + str(results['Training Time'][d][k]) + '\t' + '\t'

    print(temp)