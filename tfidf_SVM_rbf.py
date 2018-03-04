'''
==================
RBF SVM parameters
==================

This example illustrates the effect of the parameters ``gamma`` and ``C`` of
the Radial Basis Function (RBF) kernel SVM.

Intuitively, the ``gamma`` parameter defines how far the influence of a single
training example reaches, with low values meaning 'far' and high values meaning
'close'. The ``gamma`` parameters can be seen as the inverse of the radius of
influence of samples selected by the model as support vectors.

The ``C`` parameter trades off misclassification of training examples against
simplicity of the decision surface. A low ``C`` makes the decision surface
smooth, while a high ``C`` aims at classifying all training examples correctly
by giving the model freedom to select more samples as support vectors.

The first plot is a visualization of the decision function for a variety of
parameter values on a simplified classification problem involving only 2 input
features and 2 possible target classes (binary classification). Note that this
kind of plot is not possible to do for problems with more features or target
classes.

The second plot is a heatmap of the classifier's cross-validation accuracy as a
function of ``C`` and ``gamma``. For this example we explore a relatively large
grid for illustration purposes. In practice, a logarithmic grid from
:math:`10^{-3}` to :math:`10^3` is usually sufficient. If the best parameters
lie on the boundaries of the grid, it can be extended in that direction in a
subsequent search.

Note that the heat map plot has a special colorbar with a midpoint value close
to the score values of the best performing models so as to make it easy to tell
them appart in the blink of an eye.

The behavior of the model is very sensitive to the ``gamma`` parameter. If
``gamma`` is too large, the radius of the area of influence of the support
vectors only includes the support vector itself and no amount of
regularization with ``C`` will be able to prevent overfitting.

When ``gamma`` is very small, the model is too constrained and cannot capture
the complexity or "shape" of the data. The region of influence of any selected
support vector would include the whole training set. The resulting model will
behave similarly to a linear model with a set of hyperplanes that separate the
centers of high density of any pair of two classes.

For intermediate values, we can see on the second plot that good models can
be found on a diagonal of ``C`` and ``gamma``. Smooth models (lower ``gamma``
values) can be made more complex by selecting a larger number of support
vectors (larger ``C`` values) hence the diagonal of good performing models.

Finally one can also observe that for some intermediate values of ``gamma`` we
get equally performing models when ``C`` becomes very large: it is not
necessary to regularize by limiting the number of support vectors. The radius of
the RBF kernel alone acts as a good structural regularizer. In practice though
it might still be interesting to limit the number of support vectors with a
lower value of ``C`` so as to favor models that use less memory and that are
faster to predict.

We should also note that small differences in scores results from the random
splits of the cross-validation procedure. Those spurious variations can be
smoothed out by increasing the number of CV iterations ``n_splits`` at the
expense of compute time. Increasing the value number of ``C_range`` and
``gamma_range`` steps will increase the resolution of the hyper-parameter heat
map.

'''
from sklearn import svm         # To use sklearn svm
from sklearn.feature_extraction.text import TfidfVectorizer  # use sklearn tfidf
from BoW_Naive_Bayes import *
from tfidf_naive import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from time import time


def tfidf_Support_Vector_Machines(sample_size = 100, nth_day = 5,test_size_percentage = 0.33, exist_file=1, save = 1):
    t0=time()
    X = tfidf_process(sample_size,nth_day, exist_file, save)
    y = return_process(sample_size,nth_day).ravel()
    
    count_train, count_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size_percentage, random_state=42)
    clf = svm.SVC().fit(count_train, y_train)   # use svm to fit data
    #print(clf)
    insample_pred=clf.predict(count_train)
    pred = clf.predict(count_test)
    outsample_accuracy_score = metrics.accuracy_score(y_test, pred)   
    insample_accuracy_score = metrics.accuracy_score(y_train, insample_pred)
    t1=time()
    print ('tfidf_Support_Vector_Machines takes %f' %(t1-t0))
    print("insample_accuracy_score,outsample_accuracy_score",insample_accuracy_score,outsample_accuracy_score)
    return pred
    
def tfidf_Support_Vector_Machines_plot(sample_size = 100, nth_day = 5,\
                                       test_size_percentage = 0.33, exist_file=1, save = 1):
    t0=time()
    print(__doc__)
    
    # Utility function to move the midpoint of a colormap to be around
    # the values of interest.
    class MidpointNormalize(Normalize):
        
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)
        
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))
    
    
    X = tfidf_process(sample_size,nth_day, exist_file, save)
    y = return_process(sample_size,nth_day).ravel()
    
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    
    scores_outsample=[0]   # list to record score
    scores_insample=[0]
    
    count_train, count_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size_percentage, random_state=42)
    
    
    best_insample_score=0
    best_insample_parameter=[0,0]   # (C,gamma) 
    best_outsample_score=0
    best_outsample_parameter=[0,0]  # (C,gamma)  
    
        
    for C in C_range:
        for gamma in gamma_range:
            clf = svm.SVC(C=C,gamma=gamma).fit(count_train, y_train)   # use svm to fit data
            #print(clf)
            insample_pred=clf.predict(count_train)
            pred = clf.predict(count_test)
            outsample_accuracy_score = metrics.accuracy_score(y_test, pred)   
            insample_accuracy_score = metrics.accuracy_score(y_train, insample_pred)
            
            if outsample_accuracy_score>max(scores_outsample):
                best_outsample_score=outsample_accuracy_score
                best_outsample_parameter[0]=C
                best_outsample_parameter[1]=gamma
            if insample_accuracy_score>max(scores_insample):
                best_insample_score=insample_accuracy_score
                best_insample_parameter[0]=C
                best_insample_parameter[1]=gamma
                
            scores_outsample.append(outsample_accuracy_score)
            scores_insample.append(insample_accuracy_score)
            print ('C:{},gamma:{}'.format(C, gamma))
            print("insample_accuracy_score,outsample_accuracy_score",insample_accuracy_score,outsample_accuracy_score)


    print("Insample: The best parameters with C: %s gamma: %s  a score of %s"
          % (best_insample_parameter[0],best_insample_parameter[1],best_insample_score))

    print("Outsample: The best parameters with C: %s gamma: %s  a score of %s"
          % (best_outsample_parameter[0],best_outsample_parameter[1],best_outsample_score))

    
    # #############################################################################

    # draw visualization of parameter effect
    
    scores_outsample_grid = np.array(scores_outsample[1:]).reshape(len(C_range),len(gamma_range))
    scores_insample_grid=np.array(scores_insample[1:]).reshape(len(C_range),len(gamma_range))
    #print(scores_outsample[1:])
    #print(len(scores_outsample[1:]))
    
    # Draw heatmap of the validation accuracy as a function of gamma and C
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.
    
    # OutSample plot
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores_outsample_grid, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.4))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Outsample: Validation accuracy')
    plt.show()
    
    # inSample plot
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores_insample_grid, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.5))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Insample: Validation accuracy')
    plt.show()
  
    t1=time()
    print ('tfidf_Support_Vector_Machines_plot takes %f' %(t1-t0))
    
if __name__ == '__main__':
    sample_size = len(Return_old_dict)
    #sample_size = 100
    nth_day = 5
    test_size_percentage = 0.33
    exist_file=1
    save = 1
    #result = tfidf_Support_Vector_Machines_plot(sample_size, nth_day, test_size_percentage, exist_file, save)    
    result = tfidf_Support_Vector_Machines(sample_size, nth_day, test_size_percentage, exist_file, save)
