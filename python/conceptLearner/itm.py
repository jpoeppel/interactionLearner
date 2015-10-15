#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:06:08 2015

@author: jpoeppel
"""

import numpy as np
from numpy import dot as npdot
import collections
from operator import itemgetter

EMAX = 0.001
EMAX_2 = EMAX**2
EMAX05_2 = (0.5*EMAX)**2

#For besttwo
SIGMAE = 0.05

WINNER = 0
BESTTWO = 1
NEIGHBOUR = 2
TESTMODE = BESTTWO
        
class Node(object):
    __slots__=('inp','out','id','neig','A')
    def __init__(self, input_array, id=-1, output=np.zeros(1)):
        self.id = id
        self.inp =  np.copy(input_array)
        self.out = np.copy(output)
        self.A = np.zeros((len(output),len(input_array)))
        self.neig = {}
        
    def addNeighbour(self, nId, n):
        self.neig[nId] = n
        
    def remNeighbour(self,nId):
        del self.neig[nId]
        
    def __repr__(self):
        return str(self.id)


class ITM(object):
    
    def __init__(self):
        self.nodes = collections.OrderedDict()
        self.ids = []
        self.idCounter = 0
        self.valAr = np.array([n.inp for n in self.nodes.values()])
#        self.nodes = []
        self.inserts = 0
        self.winners = []
#        self.nodes= {}
#        self.nodes = np.zeros # Try storing all nodes in one nparray that needs to be reshaped
        
        
    def train(self, node):
        self.update(np.concatenate((node.wIn,node.action)), node.wOut)
        
    def update(self, x, y, etaIn=0.0, etaOut=0.0, etaA=0.0, testMode=None):
        #Get winners:
        if len(self.nodes) > 1:
            numpyVals = self.valAr - x 
            sortedIndices = np.argsort(np.linalg.norm(numpyVals, axis=1))
            w = self.nodes[self.ids[sortedIndices[0]]]
            wI =  w.id
            self.winners.append(w.id)
            s = self.nodes[self.ids[sortedIndices[1]]]
            sI = s.id
            wsdif = w.inp-s.inp
            #Adapt winner
            dif = x-w.inp
            ndif = npdot(dif,dif)
            dwIn = etaIn*dif
            w.inp += dwIn
            cor = npdot(w.A,dif)
            dwout = etaOut*(y-w.out+cor) + np.dot(w.A,dwIn)
            w.out += dwout
            if ndif > 0.0:
                w.A += etaA*np.outer((y-w.out+cor), dif/ndif)
            #Add edge
            w.addNeighbour(sI,s)
            s.addNeighbour(wI,w)
            #Get expected output according to currently used prediction scheme
            expOut = self.test(x, sortedIndices, testMode)
#            #Check neighbours
            for nI, n in w.neig.items():
                if n.id != s.id and npdot(wsdif,n.inp-s.inp) < 0:
#                if nI != s.id and npdot(np.concatenate((w.inp,w.out))-np.concatenate((s.inp,s.out)),np.concatenate((n.inp,n.out))-np.concatenate((s.inp,s.out))) <0:
#                if n.id != s.id and npdot(w-s,n-s) < 0:
                    n.remNeighbour(wI)
                       
                    if len(n.neig) == 0:
                         #delete n
                        self.deleteNode(nI)
                    w.remNeighbour(nI)
            #Check for new node
#            if npdot(np.concatenate((w.inp,w.out))-np.concatenate((x,y)),np.concatenate((s.inp,s.out))-np.concatenate((x,y))) > 0 and np.linalg.norm(np.concatenate((x,y))-np.concatenate((w.inp,w.out))) > EMAX:
#            print "ndif: ", ndif
#            print "thales: ", npdot(w.inp-x,s.inp-x)
            outNorm = np.linalg.norm(y)
#            outNorm = npdot(y,y)
            if outNorm != 0:
                outputDim = np.floor(np.log10(outNorm))-1
            else:
                outputDim = -3
#            print "outputDim: ", outputDim
#            if npdot(expOut-y,expOut-y) > 10**outputDim:
            if np.linalg.norm(expOut-y) > 10**outputDim:
#            if npdot(w.inp-x,s.inp-x) > 0 and ndif > EMAX_2:
#                nI = len(self.nodes)
                nI = self.idCounter
                n= Node(x,nI,y)
                self.nodes[nI] = n
#                self.nodes.append(n)
                self.inserts += 1
                self.ids.append(nI)
                self.idCounter += 1
                self.valAr = np.array([node.inp for node in self.nodes.values()])
#                self.valAr = np.array([np.concatenate((node.inp,node.out)) for node in self.nodes.values()])
                w.addNeighbour(nI, n)
                n.addNeighbour(wI, w)
#            
            if npdot(wsdif,wsdif) < EMAX05_2:
                if len(self.nodes) > 2:
                    self.deleteNode(sI)             
        else:
#            To few nodes
#            nI = len(self.nodes)
            nI = self.idCounter
            self.nodes[nI] = Node(x,nI,y)
            self.ids.append(nI)
            self.idCounter += 1
            self.valAr = np.array([n.inp for n in self.nodes.values()])
#            self.valAr = np.array([np.concatenate((node.inp,node.out)) for node in self.nodes.values()])
            self.inserts += 1
            
        pass
    
    def deleteNode(self, nodeId):
        for nI, n in self.nodes[nodeId].neig.items():
            n.remNeighbour(nodeId)
            if len(n.neig) == 0 and len(self.nodes) > 2:
                self.deleteNode(nI)
        del self.nodes[nodeId]
        self.valAr = np.array([n.inp for n in self.nodes.values()])
#        self.valAr = np.array([np.concatenate((node.inp,node.out)) for node in self.nodes.values()])
#        self.ids = [n.id for n in self.nodes.values()]
        self.ids.remove(nodeId)
    
    def test(self, x, sortedIndices = None, testMode=None):
#        numpyVals = np.array([n.inp for n in self.nodes.values()])-x
        if sortedIndices == None:
            numpyVals = self.valAr - x
            sortedIndices = np.argsort(np.linalg.norm(numpyVals, axis=1))
        if testMode == None:
            testMode = TESTMODE
        
        if testMode == WINNER:
    #        ids = [n.id for n in self.nodes.values()]
            w =  self.nodes[self.ids[sortedIndices[0]]]
            print "x in : ", x
            print "winner in: ", w.inp
            print "winner out: ", w.out
            return w.out+npdot(w.A,x-w.inp)
        elif testMode == BESTTWO:
            if len(self.nodes) > 1:
                
                w = self.nodes[self.ids[sortedIndices[0]]]           
                s = self.nodes[self.ids[sortedIndices[1]]]
                print "x in : ", x
                print "winner in: ", w.inp
                print "winner out: ", w.out
                print "second in: ", s.inp
                print "second out: ", s.out
                norm = np.exp(-np.linalg.norm(x-w.inp)**2/(SIGMAE**2))
                res = norm*(w.out+npdot(w.A,x-w.inp))
                wc = np.exp(-np.linalg.norm(x-s.inp)**2/(SIGMAE**2))
                res += wc*(s.out+npdot(s.A,x-s.inp))
                norm += wc
                if norm != 0:
                    return res/norm
                else:
                    return res
            else:
                return self.nodes[self.ids[sortedIndices[0]]].out
        elif testMode == NEIGHBOUR:
            w = self.nodes[self.ids[sortedIndices[0]]]    
            norm = np.exp(-np.linalg.norm(x-w.inp)**2/(SIGMAE**2))
            res = norm*(w.out+npdot(w.A,x-w.inp))
            for nI, n in w.neig.items():
                wc = np.exp(-np.linalg.norm(x-n.inp)**2/(SIGMAE**2))
                res += wc*(n.out+npdot(n.A,x-n.inp))
                norm += wc
            if norm != 0:
                return res/norm
            else:
                return res


    
if __name__ == "__main__":
#    a = np.array([1,2,3])
#    b = [4,5,6]
#    n = Node(a)
#    n + b
#    n2 = Node(b)
#    print n
#    print type(n)
#    c = np.array([n,n2])
##    print c[1]
##    print type(c)
##    print type(c[0])
#    n3 = Node([a,b])
##    print n3
##    print type(n3)
    inputs = [[0,0],[0,1],[1,1],[1,0]]
    outputs = [np.array([0]),np.array([1]),np.array([0]),np.array([1])]
    net = ITM()
    for ins,out in zip(inputs, outputs):
        net.update(ins, out)
        
    print net.test([0.5,0.5])
    
    d = collections.OrderedDict()
    d[1] = 1
    d[2] = 2
    d[3] = 3
    print d.values()
    del d[2]
    d[4] = 4
    d[2] = 2
    print d.values()
#    
