from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #Linear SVM for feture maps
from sklearn.svm import SVC #Dual SVM for Kernels
from time import time
import numpy as np
import FeatureMaps as maps
import DataReader as DR

#all datasets
datasets= [DR.australian,DR.fourclass,DR.ionosphere,DR.heart,DR.pima,DR.wprognostic,DR.bupa,DR.fertility,DR.wdiagnostic]

#all dataset names
data_names=['Australian','Fourclass', 'Ionosphere', 'Heart', 'PIMA', 'W. Prognostic','Bupa', 'Fertility', 'W. Diagnostic' ]


#mapping funtions
mapping_functions=[maps.linear, maps.phi_p_1,maps.phi_p_1,maps.phi_p_d,maps.phi_p_d]

#p values according to the mapping functions in respected order in given Table 1
#p value of the linear kernel is used as a dummy variable
p_vals= [0,1,2,1,2]

#mapping and kernel names
kernel_names=['LIN',r'$\phi_{1,1}$',r'$\phi_{2,1}$',r'$\phi_{1,d}$',r'$\phi_{2,d}$','POL','RBF']

#set of POL kernel degree pameters 
d_params = [2, 3, 4]

#set of  RBF kernel gamma parameters
g_params = np.power(10.0, range(-5, 5))

#set of C parameters
c_params = np.power(10.0, range(-5, 5))

# random state for spliting the dataset
random_state=42

# a dictionary to store  all results
results={'Dataset':[], 'Training Acc.':[], 'Test Acc.':[], 'Training Time':[]}

for  i in range(len(datasets)):
    X,y=datasets[i]()
    #a dictionary to store performance metrics
    average_performance_metrics = {'Kernel': [], 'Training Acc': [], 'Test Acc': [], 'Training Time': []}
    #10-fold cros validation object 
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    #2-fold cross validation object
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

    for m in range(len(mapping_functions)):
        average_performance_metrics['Kernel'].append(kernel_names[m])
        performance_metrics = {'Train_Acc': [], 'Test_Acc': [], 'Train_Time': []}
        #begin: 10-fold
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            acc_by_param = []
            #begin: grid search
            for ci in c_params:
                all_acc = []
                #begin: two-fold
                for inner_train_index, inner_test_index in inner_cv.split(X_train, y_train):
                    #begin: scaling
                    scaler = StandardScaler()
                    X_inner_train = scaler.fit_transform(X_train[inner_train_index])
                    X_inner_test = scaler.transform(X_train[inner_test_index])
                    #end: scaling
                    y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
                    clf = LinearSVC(C=ci, dual=False).fit(mapping_functions[m](X_inner_train, p=p_vals[m]), y_inner_train)
                    all_acc.append(accuracy_score(y_inner_test, clf.predict(mapping_functions[m](X_inner_test, p=p_vals[m]))))
                #end: two-fold
                acc_by_param.append(np.mean(all_acc))
            #end: grid search
            #get best hyperparameters
            best_c = c_params[np.argmax(acc_by_param)]
            #begin: scaling
            scaler = StandardScaler()
            X_train_scaled=scaler.fit_transform(X_train)
            X_test_scaled=scaler.transform(X_test)
            #end: scaling
            s = time()
            XD_train = mapping_functions[m](X_train_scaled, p=p_vals[m])
            clf = LinearSVC(C=best_c, dual=False).fit(XD_train, y_train)
            performance_metrics['Train_Time'].append(round(time() - s, 4))
            performance_metrics['Train_Acc'].append(accuracy_score(y_train, clf.predict(XD_train)))
            performance_metrics['Test_Acc'].append(accuracy_score(y_test, clf.predict(mapping_functions[m](X_test_scaled, p=p_vals[m]))))
        #end: 10-fold
        average_performance_metrics['Training Acc'].append(round(100 * np.mean(performance_metrics['Train_Acc']), 2))
        average_performance_metrics['Test Acc'].append(round(100 * np.mean(performance_metrics['Test_Acc']), 2))
        average_performance_metrics['Training Time'].append( round(np.mean(performance_metrics['Train_Time']), 4))
    
    average_performance_metrics['Kernel'].append('POL')  
    performance_metrics = {'Train_Acc': [], 'Test_Acc': [], 'Train_Time': []}
    #begin: 10-fold for Pol Kernel
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc_by_param = {'ci': [], 'di': [], 'inner_test_acc': []}
        #begin: grid search
        for ci in c_params:
            for di in d_params:
                all_acc = []
                #begin: two-fold
                for inner_train_index, inner_test_index in inner_cv.split(X_train, y_train):
                    #begin: scaling
                    scaler = StandardScaler()
                    X_inner_train = scaler.fit_transform(X_train[inner_train_index])
                    X_inner_test = scaler.transform(X_train[inner_test_index])
                    #end: scaling
                    y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
                    clf = SVC(kernel='poly', C=ci, degree=di).fit(X_inner_train, y_inner_train)
                    all_acc.append(accuracy_score(y_inner_test, clf.predict(X_inner_test)))
                #end: two-fold
                acc_by_param['ci'].append(ci)
                acc_by_param['di'].append(di)
                acc_by_param['inner_test_acc'].append(np.mean(all_acc))
        #end: grid search
        #get best hyperparameters
        best_ind = np.argmax(acc_by_param['inner_test_acc'])
        best_c, best_d = acc_by_param['ci'][best_ind], acc_by_param['di'][best_ind]
        #begin: scaling
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_test_scale = scaler.transform(X_test)
        #end: scaling
        s = time()
        clf = SVC(kernel='poly', C=best_c, degree=best_d).fit(X_train_scale, y_train)
        performance_metrics['Train_Time'].append(round(time() - s, 4))
        performance_metrics['Train_Acc'].append(accuracy_score(y_train, clf.predict(X_train_scale)))
        performance_metrics['Test_Acc'].append(accuracy_score(y_test, clf.predict(X_test_scale)))
     #end: 10-fold
    average_performance_metrics['Training Acc'].append(round(100 * np.mean(performance_metrics['Train_Acc']), 2))
    average_performance_metrics['Test Acc'].append(round(100 * np.mean(performance_metrics['Test_Acc']), 2))
    average_performance_metrics['Training Time'].append(round(np.mean(performance_metrics['Train_Time']), 4))
    
    average_performance_metrics['Kernel'].append('RBF')
    performance_metrics = {'Train_Acc': [], 'Test_Acc': [], 'Train_Time': []}
    #begin: 10-fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc_by_param = {'ci': [], 'gi': [], 'inner_test_acc': []}
        #begin: grid search
        for ci in c_params:
            for gi in g_params:
                all_acc = []
                #begin: two-fold
                for inner_train_index, inner_test_index in inner_cv.split(X_train, y_train):
                    #begin: scaling
                    scaler = StandardScaler()
                    X_inner_train = scaler.fit_transform(X_train[inner_train_index])
                    X_inner_test = scaler.transform(X_train[inner_test_index])
                    #end: scaling
                    y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
                    clf = SVC(kernel='rbf', C=ci, gamma=gi).fit(X_inner_train, y_inner_train)
                    all_acc.append(accuracy_score(y_inner_test, clf.predict(X_inner_test)))
                #end: two-fold
                acc_by_param['ci'].append(ci)
                acc_by_param['gi'].append(gi)
                acc_by_param['inner_test_acc'].append(np.mean(all_acc))
        #end: grid search
        #get best hyperparameters
        best_ind = np.argmax(acc_by_param['inner_test_acc'])
        best_c, best_g = acc_by_param['ci'][best_ind], acc_by_param['gi'][best_ind]
        #begin: scaling
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_test_scale = scaler.transform(X_test)
        #end: scaling
        s = time()
        clf = SVC(kernel='rbf', C=best_c, gamma=best_g).fit(X_train_scale, y_train)
        performance_metrics['Train_Time'].append(round(time() - s, 4))
        performance_metrics['Train_Acc'].append(accuracy_score(y_train, clf.predict(X_train_scale)))
        performance_metrics['Test_Acc'].append(accuracy_score(y_test, clf.predict(X_test_scale)))
    #end: 10-fold
    average_performance_metrics['Training Acc'].append(round(100 * np.mean(performance_metrics['Train_Acc']), 2))
    average_performance_metrics['Test Acc'].append(round(100 * np.mean(performance_metrics['Test_Acc']), 2))
    average_performance_metrics['Training Time'].append(round(np.mean(performance_metrics['Train_Time']), 4))
    
    #store all results
    results['Dataset'].append(data_names[i])
    results['Training Acc.'].append(average_performance_metrics['Training Acc'])
    results['Test Acc.'].append(average_performance_metrics['Test Acc'])
    results['Training Time'].append(average_performance_metrics['Training Time'])

print('***** Table 1: Average Test Accuracies')
n_datasets=len(datasets)
n_kenels=len(kernel_names)
for d in range(n_datasets):
    print()
    print('***'+ results['Dataset'][d])
    temp=''
    for k in range(n_kenels):
        temp+=kernel_names[k] +'\t'+ str(results['Test Acc.'][d][k])+ '\t'+ '\t'

    print(temp)