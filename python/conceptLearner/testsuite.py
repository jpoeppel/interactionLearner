# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:16:14 2015
TESTSUITE to try out different models
@author: jpoeppel
"""
from sklearn import linear_model 
from sklearn.gaussian_process import GaussianProcess

if __name__ == "__main__":

#
    trainData = [[0.18894831,-1.34927749,0.025, -0.4792413,0.04334731,0.],
                 [0.18894831,-1.34927749,0.025,0.,0.,0.],
                 [0.2,-1.4,0.025,0.,0.,0.],
                 [0.1,-1.2,0.025,0.,0.,0.],
                 [-0.15,1.4,0.025,0.,0.,0.]]
    trainResultsX = [-0.29029299, 0.18894831, 0.2, 0.1,-0.15]   
    
    testData = [[0.2,10,0.025, 0.0,0.0,0.]]
    
#    
#    trainData = [[0.18894831,-1.34927749,0.025, 0,1,0,0, -0.4792413,0.04334731,0.,1, 0.72874806,-0.82986592,0.],
#                 [0.18894831,-1.34927749,0.025,0.,1.,0.,0.,0.,0.,0.,1,-0.4792413,0.04334731,0.],
#                 [0.2,-1.4,0.025,0.,1.,0.,0.,0.,0.,0.,1,-0.4792413,0.04334731,0.],
#                 [0.1,-1.2,0.025,0.,1.,0.,0.,0.,0.,0.,1,-0.4792413,0.04334731,0.],
#                 [-0.15,1.4,0.025,0.,1.,0.,0.,0.,0.,0.,1,-0.4792413,0.04334731,0.]]
#    trainResultsX = [-0.29029299, 0.18894831, 0.2, 0.1,-0.15]
#    
#    testData = [[0.18,0,0.025, 0,1,0,0, 0.0,0.0,0.,1, 0.72874806,-0.82986592,0.]]
        
    gp = GaussianProcess(corr='cubic')
    gp.fit(trainData,trainResultsX)
    print gp.predict(testData[0])
#    clf = linear_model.SGDRegressor()
#    for i in range(len(trainData)):
#        clf.partial_fit([trainData[i]],[trainResultsX[i]])
#    print clf.predict(testData[0])
    
    
    
