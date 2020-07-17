import numpy as np
from numpy import linalg

# p is used as a dummy variable for practical reasons
def linear(X,p=1):
    return X

# use when dataset scaled with choice of a=0
def phi_p_1(X, p=1):
    return np.hstack((X, (linalg.norm(X, axis=1, ord=p) ** p).reshape((len(X), 1))))


# use when dataset scaled with choice of a=0
def phi_p_d(X, p=1):
    return np.hstack((X, np.abs(X) ** p))



# use when multiple anchor points are provided
def min_phi_p_1(X,anchors=[],p=1):
    if len(anchors)>=2:
        temp=[]
        for a in anchors:
            temp.append(linalg.norm(X-a,axis=1,ord=p)**p)
        return np.hstack((X,np.min(temp,axis=0).reshape((len(X), 1))))
    else:
        print("The  anchor point list should include at least two anchors!")

