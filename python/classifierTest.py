#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:27:57 2015

@author: jpoeppel
"""

import numpy as np

from sklearn import svm
from sklearn import tree
from sklearn import ensemble

from conceptLearner import network
from sklearn.externals.six import StringIO
import pydot

#f = open("../data/actionVectorsTestNew", "r")
#lines = f.readlines()
#f.close()
#
#f = open("../data/actionVectorsTestNew", "w")
#for l in lines:
#    f.write(l[:l.rfind(";")] + "\n")
#f.close()
#data = np.loadtxt("../data/actionVectors2_n", skiprows=1, delimiter=";")
#data = np.loadtxt("../data/actionVectors2_Clean2.csv", skiprows=1)
data = np.loadtxt("../data/actionVectors10Runs", skiprows=1, delimiter=";")

data = np.loadtxt("../data/actionVectors10Runs_Clean.csv", skiprows=1)
#data = np.loadtxt("../data/actionVectors3_Clean.csv", skiprows=1)
featureNames = np.array(["sid","oid","dist","closing","contact","relPosX", "relPosY", "relPosZ",
                         "relVlX", "relVlY", "relVlZ", "closingDivDist", "closing1", "closing2", 
                         "closing1DivDist", "closing2DivDist"])
                         
mask = np.array([0,1,2,5,6,7,8,9,10,12,13,14,15]) #No closing,closingDivDist
mask = np.array([0,1,2,3,5,6,7,8,9,10,11]) # No closing1,closing2,closing1DivDist, closing2DivDist
mask = np.array([0,1,2,3,4,5,6,8,9,11,12,13,14,15]) # No relPosz, relVelZ
mask = np.array([0,1,2,4,5,6,8,9,12,13])
mask = np.array(range(16))


nth = 4
numSplits = 10

trainErrorsSVM = np.zeros(numSplits)
trainErrorsDT = np.zeros(numSplits)
trainErrorsForest = np.zeros(numSplits)
trainErrorsLVQ = np.zeros(numSplits)

testErrorsSVM = np.zeros(numSplits)
testErrorsDT = np.zeros(numSplits)
testErrorsForest = np.zeros(numSplits)
testErrorsLVQ = np.zeros(numSplits)

for i in xrange(numSplits):
    np.random.shuffle(data)
    print "Testrun ", i
    trainSet = data[::nth,mask]
    trainLabel = data[::nth,16]
    
    testSet = np.copy(data[:,mask])
    testSet = np.delete(testSet, np.s_[::nth],0)
    testLabel = np.copy(data[:,16])
    testLabel = np.delete(testLabel, np.s_[::nth],0)
    
    svmModel = svm.SVC(kernel='rbf',class_weight='auto')
    svmModel.fit(trainSet, trainLabel)
    predictionSVM = svmModel.predict(trainSet)
    trainErrorsSVM[i] = np.mean(predictionSVM != trainLabel)
    print "Trainerror for svm: ", trainErrorsSVM[i]
    predictionSVM = svmModel.predict(testSet)
    testErrorsSVM[i] =  np.mean(predictionSVM != testLabel)
    print "Testerror for svm: ", testErrorsSVM[i]
    
    
    dtModel = tree.DecisionTreeClassifier(class_weight='auto')
    dtModel.fit(trainSet,trainLabel)
    predictionDT = dtModel.predict(trainSet)
    trainErrorsDT[i] =  np.mean(predictionDT != trainLabel)
    print "Trainerror for DT: ", trainErrorsDT[i]
    predictionDT = dtModel.predict(testSet)
    testErrorsDT[i] = np.mean(predictionDT != testLabel)
    print "Testerror for DT: ", testErrorsDT[i]
    
    
    forestModel = ensemble.RandomForestClassifier(class_weight='auto')
    forestModel.fit(trainSet, trainLabel)
    predictionForest = forestModel.predict(trainSet)
    trainErrorsForest[i] = np.mean(predictionForest != trainLabel)
    print "Trainerror for Forest: ", trainErrorsForest[i]
    predictionForest = forestModel.predict(testSet)
    testErrorsForest[i] = np.mean(predictionForest != testLabel)
    print "Testerror for Forest: ", testErrorsForest[i]
    
#    lvqModel = network.LVQNeuralNet(len(trainSet[0]))
#    lvqModel.trainOffline(trainSet, trainLabel, 1)
#    
#    predictionLVQ = [lvqModel.classify(x) for x in trainSet]
#    trainErrorsLVQ[i] = np.mean(predictionLVQ != trainLabel)
#    print "Trainerror for LVQ: ", trainErrorsLVQ[i]
#    predictionLVQ = [lvqModel.classify(x) for x in testSet]
#    testErrorsLVQ[i] = np.mean(predictionLVQ != testLabel)
#    print "Testerror for LVQ: ", testErrorsLVQ[i]


print "Average trainError SVM: ", np.mean(trainErrorsSVM)
print "Average trainError DT: ", np.mean(trainErrorsDT)
print "Average trainError Forest: ", np.mean(trainErrorsForest)
#print "Average trainError LVQ: ", np.mean(trainErrorsLVQ)

print "Average testError SVM: ", np.mean(testErrorsSVM)
print "Average testError DT: ", np.mean(testErrorsDT)
print "Average testError Forest: ", np.mean(testErrorsForest)
#print "Average testError LVQ: ", np.mean(testErrorsLVQ)

dot_data = StringIO()
tree.export_graphviz(dtModel, out_file=dot_data, feature_names=featureNames[mask])
graph = pydot.graph_from_dot_data(dot_data.getvalue())
if graph != None:
    graph.write_pdf("../data/ActionTreeOffline.pdf")