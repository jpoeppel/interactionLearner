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

def trainITM(net, inputs, outputs):
    for i in xrange(len(inputs)):
        net.train(Node("", wIn=inputs[i,:][mask], wOut=outputs[i,:]))
        
def trainITM2(net, inputs, outputs):
    for i in xrange(len(inputs)):
        net.update(inputs[i,:][mask], outputs[i,:], etaIn=0.0, etaOut=0.5, etaA=0.2)

if __name__ == "__main__":
    np.set_printoptions(precision=3,suppress=True)
#    data = np.loadtxt("../testData.txt", delimiter=";")
    trainDataPush = np.loadtxt("../trainDataPush20.txt", delimiter=";")
    testDataPush = np.loadtxt("../testDataPush.txt", delimiter=";")
    
#    inputs = data[:,:13]
#    outputs = data[:,13:]
    
    inputs = trainDataPush[:,:13]
    outputs = trainDataPush[:,13:]

    print "means: ", np.mean(inputs, axis=0)

    itm = ITM.ITM()
    itm2 = ITM2.ITM()    
    
    t0 = time.time()
    trainITM(itm, inputs[:700,:], outputs[:700,:])
    t1 = time.time()
    trainITM2(itm2, inputs[:700,:], outputs[:700,:])
    t2 = time.time()
    
    print "trained itm in: ", t1-t0
    print "trained itm2 in: ", t2-t1
    
    print "#inserts itm: ", itm.inserts
    print "#inserts itm2: ", itm2.inserts
        
    print "#nodes itm: ", len(itm.nodes)
    print "#nodes itm2: ", len(itm2.nodes)
    
    print "testing itm: "
    numErrors = 0
    totalErr = 0.0
    for i in xrange(len(inputs)):
        pred = itm.predict(inputs[i,:][mask])
        norm = np.linalg.norm(pred- outputs[i])
        totalErr += norm
        if norm >0.003:
#            print "input: {}, output: {}, pred: {}".format(inputs[i], outputs[i], pred)
            numErrors  += 1

            
    print "testing itm2: "
    numErrors2 = 0
    totalErr2 = 0.0
    for i in xrange(len(inputs)):
        pred = itm2.test(inputs[i,:][mask])
        norm = np.linalg.norm(pred- outputs[i])
        totalErr2 += norm
        if norm >0.003:
#            print "input: {}, output: {}, pred: {}".format(inputs[i], outputs[i], pred)
            numErrors2 += 1
            
    print "#errors itm: ", numErrors
    print "total error itm: ", totalErr
    print "#errors itm2: ", numErrors2
    print "total error itm2: ", totalErr2
    
#    print "winners itm: ", itm.winners
#    print "winners itm2: ", itm2.winners
