import os.path
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
my_path = os.path.abspath("./DataSets")

#random state for the datasets, Spambase, Phoneme, Magic for which training and test parts are not provided
random_state=42

# *************************************************************************************
def Adult():
    # X_train,y_train=train_Data[0],train_Data[1]
    train_Data = load_svmlight_file(os.path.abspath("./DataSets") + "/LibSVMDataSets" + "/Adult_Training", 123, multilabel=False)
    # X_test, y_test=test_Data[0],test_Data[1]
    test_Data = load_svmlight_file(os.path.abspath("./DataSets") + "/LibSVMDataSets" + "/Adult_Test", 123, multilabel=False)
    # X_train,X_test,y_train,y_test
    return train_Data[0].toarray(), test_Data[0].toarray(), train_Data[1], test_Data[1]


# *************************************************************************************
def Spambase():
    f = open(my_path +  "/CIDatasets"+ "/Spambase.txt")
    X, y = [], []
    for line in f:
        line = line.split(',')
        y.append(int(line[-1]))
        X.append([float(line[i]) for i in range(len(line) - 1) if line[i] != ''])

    return train_test_split(np.array(X), np.array(y), test_size=0.3, random_state=random_state, stratify=np.array(y), shuffle=True)
# *************************************************************************************
def Magic():
    f = open(my_path + "/CIDatasets"+ "/Magic.txt")
    X, y = [], []
    for line in f:
        line = line.split(',')
        y.append((line[-1][0]))
        X.append([float(line[i]) for i in range(len(line) - 1) if line[i] != ''])
    return train_test_split(np.array(X), np.array(y), test_size=0.3, random_state=random_state, stratify=np.array(y), shuffle=True)

# *************************************************************************************
def Phoneme():
    f = open(my_path +"/CIDatasets"+ "/Phoneme.txt")
    X, y = [], []
    for line in f:
        line = line.split(',')
        y.append((line[-1]))
        X.append([float(line[i]) for i in range(len(line) - 1) if line[i] != ''])
    return  train_test_split(np.array(X), np.array(y), test_size=0.3, random_state=random_state, stratify=np.array(y), shuffle=True)
# *************************************************************************************
def Guide1():
    # X_train,y_train=train_Data[0],train_Data[1]
    train_Data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Guide_1_Training.txt", 4, multilabel=False)
    # X_test, y_test=test_Data[0],test_Data[1]
    test_Data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Guide_1_Test.txt", 4, multilabel=False)
    # X_train,X_test,y_train,y_test
    return train_Data[0].toarray(), test_Data[0].toarray(), train_Data[1], test_Data[1]
# *************************************************************************************
def Wilt():
    f = open(my_path +  "/CIDatasets"+"/Wilt_Training.txt")
    X_training, y_training = [], []
    for line in f:
        line = line.split(',')
        y_training.append(line[0])
        X_training.append(
            [float(line[i]) for i in range(1, len(line)) if line[i] != '' and line[i] != '\n'])
    f = open(my_path +  "/CIDatasets"+"/Wilt_Test.txt")
    X_test, y_test = [], []
    for line in f:
        line = line.split(',')
        y_test.append(line[0])
        X_test.append(
            [float(line[i]) for i in range(1, len(line)) if line[i] != ' ' and line[i] != '\n'])
    return np.array(X_training), np.array(X_test), np.array(y_training), np.array(y_test)
# *************************************************************************************
def Splice():
    # X_train,y_train=train_Data[0],train_Data[1]
    train_Data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Splice_Training.txt", 60, multilabel=False)
    # X_test, y_test=test_Data[0],test_Data[1]
    test_Data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Splice_Test.txt", 60, multilabel=False)
    # X_train,X_test,y_train,y_test
    return train_Data[0].toarray(), test_Data[0].toarray(), train_Data[1], test_Data[1]

# *************************************************************************************
def australian():
    # X_train,y_train=train_Data[0],train_Data[1]
    data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Australian", 14, multilabel=False)
    # X,y
    return data[0].toarray(), data[1]
# *************************************************************************************
def fourclass():
    # X_train,y_train=train_Data[0],train_Data[1]
    data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Fourclass", 2, multilabel=False)
    # X,y
    return data[0].toarray(), data[1]

# *************************************************************************************
def ionosphere():
    f = open(my_path+ "/CIDatasets"+ "/Ionosphere.txt")
    X, y = [], []
    for line in f:
        line = line.split(',')
        y.append(line[-1].replace('\n', ''))
        X.append([float(line[i]) for i in range(len(line) - 1) if line[i] != ' '])
    return np.array(X), np.array(y)
 # *************************************************************************************
def heart():
    # X_train,y_train=train_Data[0],train_Data[1]
    data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Heart.txt", 13, multilabel=False)
    # X,y
    return data[0].toarray(), data[1]
# *************************************************************************************
def pima():
        # X_train,y_train=train_Data[0],train_Data[1]
        data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Pima_Indians.txt", 8, multilabel=False)
        # X,y
        return data[0].toarray(), data[1]

# *************************************************************************************
def wprognostic():
        f = open(my_path+ "/CIDatasets"+"/wisconsinPrognosis.txt")
        X, y = [], []
        for line in f:
            line = line.split(',')
            y.append(line[1].replace('\n', ''))
            X.append([float(line[i]) for i in range(2,len(line)) if line[i] != ' '])
        return np.array(X), np.array(y)
# *************************************************************************************
def bupa():
        f = open(my_path +"/CIDatasets"+ "/BupaLiver.txt")
        X, y = [], []
        for line in f:
            line = line.split(',')
            y.append(int(line[-1].replace('\n', '')))
            X.append([float(line[i]) for i in range(len(line) - 1) if line[i] != ' '])
        return np.array(X), np.array(y)
# *************************************************************************************
def fertility():
        f = open(my_path + "/CIDatasets"+"/Fertility.txt")
        X, y = [], []
        for line in f:
            line = line.split(',')
            y.append(line[-1])
            X.append([float(line[i]) for i in range(0, len(line)-1) if line[i] != ' ' and line[i] != '\n'])
        return np.array(X), np.array(y)
# *************************************************************************************
def wdiagnostic():
        # X_train,y_train=train_Data[0],train_Data[1]
        data = load_svmlight_file(my_path + "/LibSVMDataSets" + "/Winsconsin_Diagnonis.txt", 10, multilabel=False)
        # X,y
        return data[0].toarray(), data[1]
# *************************************************************************************
