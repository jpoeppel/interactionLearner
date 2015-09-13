# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:19:03 2015

@author: jpoeppel
"""

import numpy as np
import itm as ITM2
import topoMaps as ITM
from network import Node
import time

def trainITM(net, inputs, outputs):
    for i in xrange(len(inputs)):
        net.train(Node("", wIn=inputs[i,:], wOut=outputs[i,:]))
        
def trainITM2(net, inputs, outputs):
    for i in xrange(len(inputs)):
        net.update(inputs[i,:], outputs[i,:])

if __name__ == "__main__":
    np.set_printoptions(precision=3,suppress=True)
    data = np.loadtxt("testData.txt", delimiter=";")
    inputs = data[::8,:13]
    outputs = data[::8,13:]

    itm = ITM.ITM()
    itm2 = ITM2.ITM()    
    
    t0 = time.time()
    trainITM(itm, inputs, outputs)
    t1 = time.time()
    trainITM2(itm2, inputs, outputs)
    t2 = time.time()
    
    print "trained itm in: ", t1-t0
    print "trained itm2 in: ", t2-t1
    
    print "#inserts itm: ", itm.inserts
    print "#inserts itm2: ", itm2.inserts
        
    print "#nodes itm: ", len(itm.nodes)
    print "#nodes itm2: ", len(itm2.nodes)
    
    print "testing itm: "
    numErrors = 0
    for i in xrange(len(inputs)):
        pred = itm.predict(inputs[i])
        if np.linalg.norm(pred- outputs[i]) >0.001:
#            print "input: {}, output: {}, pred: {}".format(inputs[i], outputs[i], pred)
            numErrors  += 1

            
    print "testing itm2: "
    numErrors2 = 0
    for i in xrange(len(inputs)):
        pred = itm2.test(inputs[i])
        if np.linalg.norm(pred- outputs[i]) >0.001:
#            print "input: {}, output: {}, pred: {}".format(inputs[i], outputs[i], pred)
            numErrors2 += 1
            
    print "#errors itm: ", numErrors
    print "#errors itm2: ", numErrors2
