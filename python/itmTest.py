# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:19:03 2015

@author: jpoeppel
"""

import numpy as np
import conceptLearner.itm as ITM2
import conceptLearner.topoMaps as ITM
from conceptLearner.network import Node
import time

mask = [3,4,5,9,10]
mask = range(13)
#mask = [3,4,5,7,8,10,11]

def trainITM(net, inputs, outputs):
    for i in xrange(len(inputs)):
        net.train(Node("", wIn=inputs[i,:], wOut=outputs[i,:]))
        
def trainITM2(net, inputs, outputs):
    for i in xrange(len(inputs)):
        net.update(np.copy(inputs[i,:]), outputs[i,:], etaIn=0.0, etaOut=0.0, etaA=0.0)

if __name__ == "__main__":
    np.set_printoptions(precision=3,suppress=True)
#    data = np.loadtxt("../testData.txt", delimiter=";")
    trainDataPush = np.loadtxt("../trainDataPush20.txt", delimiter=";")
    testDataPush = np.loadtxt("../testDataPush.txt", delimiter=";")
    
#    inputs = data[:,:13]
#    outputs = data[:,13:]
    
    inputMask = np.arange(len(trainDataPush))[:700]
#    inputMask = np.array([0,57,114])
    
    inputs = trainDataPush[inputMask,:13]
    outputs = trainDataPush[inputMask,13:]
    
    testinputs = trainDataPush[:,:13]
    testoutputs = trainDataPush[:,13:]

    print "means: ", np.mean(inputs, axis=0)
    print "std: ", np.std(inputs, axis=0)
    print "max-min: ", (np.max(inputs, axis=0)-np.min(inputs, axis=0))
    normedInputs = inputs
#    normedInputs = (inputs-np.mean(inputs)) / np.std(inputs, axis=0)
#    normedInputs = (inputs-np.min(inputs, axis=0))/(np.max(inputs, axis=0)-np.min(inputs, axis=0))
    
    itm = ITM.ITM()
    itm2 = ITM2.ITM()    
    
    t0 = time.time()
    trainITM(itm, normedInputs, outputs)
    t1 = time.time()
    trainITM2(itm2, normedInputs, outputs)
    t2 = time.time()
    
    print "trained itm in: ", t1-t0
    print "trained itm2 in: ", t2-t1
    
    print "#inserts itm: ", itm.inserts
    print "#inserts itm2: ", itm2.inserts
        
    print "#nodes itm: ", len(itm.nodes)
    print "#nodes itm2: ", len(itm2.nodes)
    
    t0 = time.time()
    numErrors = 0
    totalErr = 0.0
    print "testing itm"
    for i in xrange(len(testinputs)):
        pred = itm.predict(testinputs[i,:])
        norm = np.linalg.norm(pred- testoutputs[i])
        totalErr += norm
        if norm >0.003:
#            print "input: {}, output: {}, pred: {}".format(inputs[i], outputs[i], pred)
            numErrors  += 1

    t1 = time.time()       
    numErrors2 = 0
    totalErr2 = 0.0
    print "testing itm2"
    for i in xrange(len(testinputs)):
        pred = itm2.test(testinputs[i,:])
        norm = np.linalg.norm(pred- testoutputs[i])
        totalErr2 += norm
        if norm >0.003:
#            print "input: {}, output: {}, pred: {}".format(inputs[i], outputs[i], pred)
            numErrors2 += 1
    t2 = time.time()

    print "tested itm in: ", t1-t0
    print "tested itm2 in: ", t2-t1    
    
    print "#errors itm: ", numErrors
    print "total error itm: ", totalErr
    print "#errors itm2: ", numErrors2
    print "total error itm2: ", totalErr2
    
#    print itm2.test(np.array([-]))
    
#    print "winners itm: ", itm.winners
#    print "winners itm2: ", itm2.winners
