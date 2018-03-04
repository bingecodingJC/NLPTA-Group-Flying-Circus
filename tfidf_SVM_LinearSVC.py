'''
Apply LinearSVC

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
    clf = svm.LinearSVC().fit(count_train, y_train)   # use svm to fit data
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
    
    C_range = np.logspace(-2, 0, 3)
    loss_range = ['hinge','squared_hinge']
    
    scores_outsample=[0]   # list to record score
    scores_insample=[0]
    
    count_train, count_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size_percentage, random_state=42)
    
    
    best_insample_score=0
    best_insample_parameter=[0,0]   # (C,loss) 
    best_outsample_score=0
    best_outsample_parameter=[0,0]  # (C,loss)  
    
        
    for C in C_range:
        for loss in loss_range:
            clf = svm.LinearSVC(C=C,loss=loss).fit(count_train, y_train)   # use svm to fit data
            #print(clf)
            insample_pred=clf.predict(count_train)
            pred = clf.predict(count_test)
            outsample_accuracy_score = metrics.accuracy_score(y_test, pred)   
            insample_accuracy_score = metrics.accuracy_score(y_train, insample_pred)
            
            if outsample_accuracy_score>max(scores_outsample):
                best_outsample_score=outsample_accuracy_score
                best_outsample_parameter[0]=C
                best_outsample_parameter[1]=loss
            if insample_accuracy_score>max(scores_insample):
                best_insample_score=insample_accuracy_score
                best_insample_parameter[0]=C
                best_insample_parameter[1]=loss
                
            scores_outsample.append(outsample_accuracy_score)
            scores_insample.append(insample_accuracy_score)
            print ('C:{},loss:{}'.format(C, loss))
            print("insample_accuracy_score,outsample_accuracy_score",insample_accuracy_score,outsample_accuracy_score)


    print("Insample: The best parameters with C: %s loss: %s  a score of %s"
          % (best_insample_parameter[0],best_insample_parameter[1],best_insample_score))

    print("Outsample: The best parameters with C: %s loss: %s  a score of %s"
          % (best_outsample_parameter[0],best_outsample_parameter[1],best_outsample_score))

    
    # #############################################################################

    # draw visualization of parameter effect
    
    scores_outsample_grid = np.array(scores_outsample[1:]).reshape(len(C_range),len(loss_range))
    scores_insample_grid=np.array(scores_insample[1:]).reshape(len(C_range),len(loss_range))
    #print(scores_outsample[1:])
    #print(len(scores_outsample[1:]))
    
    # Draw heatmap of the validation accuracy as a function of loss and C
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
    plt.xlabel('loss')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(loss_range)), loss_range)#, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Outsample: Validation accuracy')
    plt.show()
    
    # inSample plot
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores_insample_grid, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.5))
    plt.xlabel('loss')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(loss_range)), loss_range)#, rotation=45)
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
