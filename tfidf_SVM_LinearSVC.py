'''
There are two functions in the code below, and both functions use sklearn svm
with LinearSVC kernal. 
    
The first function will show the insample accuracy score, outsample accuracy 
score, and the running time using input of tfidf process.

The second function will show the insample accuracy score, outsample accuracy 
score by changing the different parameters, that is C and loss function. 
In addition,it will draw a heatmap of insampe and outsample accuracy scores by 
different parameters. Lastly, it will print the best insample acccuracy score, 
outsample accuracy score, and the running time. 

@author: kuochenghao
@date: Mar 16 2018

'''

# To use sklearn svm
from sklearn import svm       
# To use functions from tfidf_naive.py
from tfidf_naive import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from time import time



def tfidf_Support_Vector_Machines(sample_size = 100, nth_day = 5,\
                                  test_size_percentage = 0.33, \
                                  exist_file=1, save = 1):
    t0=time()
    X = tfidf_process(sample_size,nth_day, exist_file, save)
    y = return_process(sample_size,nth_day).ravel()
    # To split the data into training and test data
    count_train, count_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size_percentage, random_state=42)
    # Use svm to fit data, and choose LinearSVC kernal
    clf = svm.LinearSVC().fit(count_train, y_train)  
    #print(clf)
    insample_pred=clf.predict(count_train)
    pred = clf.predict(count_test)   
    outsample_accuracy_score = metrics.accuracy_score(y_test, pred)   
    insample_accuracy_score = metrics.accuracy_score(y_train, insample_pred)
    t1=time()
    print ('tfidf_Support_Vector_Machines takes %f' %(t1-t0))
    print("insample_accuracy_score,outsample_accuracy_score",\
          insample_accuracy_score,outsample_accuracy_score)
    return pred
    
def tfidf_Support_Vector_Machines_plot(sample_size = 100, nth_day = 5,\
                                       test_size_percentage = 0.33, \
                                       exist_file=1, save = 1):
    t0=time()
    #print(__doc__)
    
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
    
    # Define the parameter ranges
    C_range = np.logspace(-4, 1, 6)
    loss_range = ['hinge','squared_hinge']
    
    # Ｌist to record accuracy scores with different parameters 
    scores_outsample=[0]   
    scores_insample=[0]
    
    count_train, count_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size_percentage, random_state=42)
    
    
    best_insample_score=0
    best_insample_parameter=[0,0]   # (C,loss) 
    best_outsample_score=0
    best_outsample_parameter=[0,0]  # (C,loss)  
    
    # The for loop is to calulate different accuracy scores by changing 
    # parameters    
    for C in C_range:
        for loss in loss_range:
            # Use svm to fit data, and choose LinearSVC kernal
            clf = svm.LinearSVC(C=C,loss=loss).fit(count_train, y_train)   
            #print(clf)
            insample_pred=clf.predict(count_train)
            pred = clf.predict(count_test)
            outsample_accuracy_score = metrics.accuracy_score(y_test, pred)   
            insample_accuracy_score = metrics.accuracy_score(y_train, insample_pred)
            
            # To record the info of the best insample and outsample scores 
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
            print("insample_accuracy_score,outsample_accuracy_score",\
                  insample_accuracy_score,outsample_accuracy_score)


    print("Insample: The best parameters with C: %s loss: %s  a score of %s"
          % (best_insample_parameter[0],best_insample_parameter[1],\
             best_insample_score))

    print("Outsample: The best parameters with C: %s loss: %s  a score of %s"
          % (best_outsample_parameter[0],best_outsample_parameter[1],\
             best_outsample_score))

    
    # #########################################################################
    # draw visualization of parameter effect
    
    scores_outsample_grid = np.array(scores_outsample[1:]).reshape\
    (len(C_range),len(loss_range))
    scores_insample_grid=np.array(scores_insample[1:]).reshape\
    (len(C_range),len(loss_range))
    #print(scores_outsample[1:])
    #print(len(scores_outsample[1:]))
    
    # Draw heatmap of the validation accuracy as a function of loss and C
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. 
    
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

'''
====================================
Explain the inputs of the functions
====================================
1. sample_size: use total samples(len(Return_old_dict)) or part of the samples(100)
2. nth_day: to classify nth_day cumulative return ,and the range is from 1 to 5.
3. test_size_percentage: the percentage of the samples for testing.    
4. exist_file: if the folder exists a file of tfidf_process result, choose 1. 
   Others choose 0. Please note it takes time for tfidf_process, so it's better
   to save a file in folder for the first time.
5. save: whether to save a tfidf_process result in folder. 1 for save. 0 for not save.
'''

if __name__ == '__main__':
    #sample_size = len(Return_old_dict)
    sample_size = 100
    nth_day = 5
    test_size_percentage = 0.33
    exist_file=0
    save = 1
    #result = tfidf_Support_Vector_Machines_plot(sample_size, nth_day, test_size_percentage, exist_file, save)    
    result = tfidf_Support_Vector_Machines(sample_size, nth_day, test_size_percentage, exist_file, save)
