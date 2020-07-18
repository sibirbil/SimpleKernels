# Low-dimensional Interpretable Kernels with Conic Discriminant Functions for Classification

We propose several simple feature maps that lead to a collection of
interpretable kernels with varying degrees of freedom. We make sure
that the increase in the dimension of input data with each proposed
feature map is extremely low, so that the resulting models can be
trained quickly, and the obtained results can easily be
interpreted. The details of this study is given in our
[paper](./InterpretableConicKernels.pdf).


# Required packages

All our codes are implemented in Pyhton 3.7 and we use the following
packages:

   1. [Numpy](https://numpy.org)
   2. [Scikit-learn](https://scikit-learn.org/stable/index.html)
   3. [Matplotlib](https://matplotlib.org/)


# Tutorials

We provide the following tutorials to demonstrate our implementation.

* For the proposed feature maps, we refer to the pages
  [FeatureMaps](./FeatureMaps.html) and [Kernels](./Kernels.html). We
  also provide the same tutorials as two notebooks, [notebook one](./FeatureMaps.ipynb) and
  [notebook two](./Kernels.ipynb), respectively.

* To obtain a row of Table 1, we refer to page [Table
  1](./Table_1.html) or to the [notebook](./Table_1.ipynb).

* To obtain a row of Table 2 and Table 3, we refer to the page
  [Table_2_3](./Table_2_3.html) or to the
  [notebook](./Table_2_3.ipynb).

* To obtain Figure 3, we refer to page [Figure_3](./Figure_3.html)
  or to the [notebook](./Figure_3.ipynb).

* To obtain Figure 5, we refer to page [Figure_5](./Figure_5.html)
  or to the [notebook](./Figure_5.ipynb).


# Reproducing our results

We provide the following scripts to reproduce the numerical
experiments that we have reported in our paper.

* [Table_1_Run.py](./Table_1_Run.py)

* [Table_2_3_Run.py](./Table_2_3_Run.py)

* [Figure_3_Run.py](./Figure_3_Run.py) 

* [Figure_5_Phoneme_Run.py](./Figure_5_Phoneme_Run.py) 

* [Figure_5_FourClass_Run.py](./Figure_5_FourClass_Run.py) 

* [Figure_6_Run.py](./Figure_6_Run.py)

# Other tutorials with various machine learning problems (work-in-progress)

* [In this tutorial](./LogisticRegression.html), we explain how to
  apply the proposed feature maps with **logistic regression** on
  binary and multi-class classificiation problems. Notebook version of
  this tutorial can be found [here](./LogisticRegression.ipynb).

* We also provide the following code snippets that reproduce Tables
  2-3 and Figure 5 in our paper, but this time, with logistic regression.
    
    - [Table_2_3_LR_Run.py](./LR_Table_2_3_Run.py)
    - [Figure_5_Phoneme_LR_Run.py](./Figure_5_Phoneme_LR.py) 
    - [Figure_5_FourClass_LR_Run.py](./Figure_5_FourClass_LR.py)
