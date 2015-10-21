#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:06:08 2015

@author: jpoeppel
"""

#import numpy as np

# Make numpy imports explicit to save some computational power
# since this class is the core bottleneck to the model
from numpy import argsort as npargsort
from numpy import array as nparray
from numpy import copy as npcopy
from numpy import dot as npdot
from numpy import exp as npexp
from numpy import floor as npfloor
from numpy import log10 as nplog10
from numpy import outer as npouter
from numpy import sqrt as npsqrt
from numpy import zeros as npzeros
from numpy.linalg import norm as npnorm


import collections
from operator import itemgetter
from config import WINNER, BESTTWO, NEIGHBOUR
from config import config
        
class Node(object):
    __slots__=('inp','out','id','neig','A')
    def __init__(self, input_array, id=-1, output=npzeros(1)):
        self.id = id
        self.inp =  npcopy(input_array)
        self.out = npcopy(output)
        self.A = npzeros((len(output),len(input_array)))
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
        self.valAr = nparray([n.inp for n in self.nodes.values()])
        self.inserts = 0
        self.updateCalls = 0
        
        
    def update(self, x, y, etaIn=0.0, etaOut=0.0, etaA=0.0, testMode=None):
        self.updateCalls += 1
        #Get winners:
        if len(self.nodes) > 1:
            numpyVals = self.valAr - x 
            sortedIndices = npargsort(npnorm(numpyVals, axis=1))
            w = self.nodes[self.ids[sortedIndices[0]]]
            wI =  w.id
            s = self.nodes[self.ids[sortedIndices[1]]]
            sI = s.id
            wsdif = w.inp-s.inp
            #Adapt winner
            dif = x-w.inp
            ndif = npdot(dif,dif)
            dwIn = etaIn*dif
            w.inp += dwIn
            cor = npdot(w.A,dif)
            dwout = etaOut*(y-w.out+cor) + npdot(w.A,dwIn)
            w.out += dwout
            if ndif > 0.0:
                w.A += etaA*npouter((y-w.out+cor), dif/ndif)
            #Add edge
#            w.addNeighbour(sI,s)
#            s.addNeighbour(wI,w)
            w.neig[sI] = s
            s.neig[wI] = w
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
            outNorm = npsqrt(npdot(y,y))#np.linalg.norm(y)
#            outNorm = npdot(y,y)
            if outNorm != 0:
                outputDim = npfloor(nplog10(outNorm))-1
            else:
                outputDim = -3
#            print "outputDim: ", outputDim
#            if npdot(expOut-y,expOut-y) > 10**outputDim:
            if npnorm(expOut-y) > 10**outputDim:
#            if npdot(w.inp-x,s.inp-x) > 0 and ndif > EMAX_2:
#                nI = len(self.nodes)
                nI = self.idCounter
                n= Node(x,nI,y)
                self.nodes[nI] = n
#                self.nodes.append(n)
                self.inserts += 1
                self.ids.append(nI)
                self.idCounter += 1
                self.valAr = nparray([node.inp for node in self.nodes.values()])
#                self.valAr = np.array([np.concatenate((node.inp,node.out)) for node in self.nodes.values()])
                w.addNeighbour(nI, n)
                n.addNeighbour(wI, w)
#            
            if npdot(wsdif,wsdif) < config.EMAX05_2:
                if len(self.nodes) > 2:
                    self.deleteNode(sI)             
        else:
#            To few nodes
#            nI = len(self.nodes)
            nI = self.idCounter
            self.nodes[nI] = Node(x,nI,y)
            self.ids.append(nI)
            self.idCounter += 1
            self.valAr = nparray([n.inp for n in self.nodes.values()])
#            self.valAr = np.array([np.concatenate((node.inp,node.out)) for node in self.nodes.values()])
            self.inserts += 1
            
        pass
    
    def deleteNode(self, nodeId):
        for nI, n in self.nodes[nodeId].neig.iteritems():
            n.remNeighbour(nodeId)
            if len(n.neig) == 0 and len(self.nodes) > 2:
                self.deleteNode(nI)
        del self.nodes[nodeId]
        self.valAr = nparray([n.inp for n in self.nodes.values()])
#        self.valAr = np.array([np.concatenate((node.inp,node.out)) for node in self.nodes.values()])
#        self.ids = [n.id for n in self.nodes.values()]
        self.ids.remove(nodeId)
    
    def test(self, x, sortedIndices = None, testMode=None):
#        numpyVals = np.array([n.inp for n in self.nodes.values()])-x
        if len(self.nodes) == 0:
            return 0
        if sortedIndices == None:
            numpyVals = self.valAr - x
            sortedIndices = npargsort(npnorm(numpyVals, axis=1))
        if testMode == None:
            testMode = config.TESTMODE
        
        if testMode == WINNER:
    #        ids = [n.id for n in self.nodes.values()]
            w =  self.nodes[self.ids[sortedIndices[0]]]
#            print "x in : ", x
#            print "winner in: ", w.inp
#            print "winner out: ", w.out
            return w.out+npdot(w.A,x-w.inp)
        elif testMode == BESTTWO:
            if len(self.nodes) > 1:
                
                w = self.nodes[self.ids[sortedIndices[0]]]           
                s = self.nodes[self.ids[sortedIndices[1]]]
#                print "x in : ", x
#                print "winner in: ", w.inp
#                print "winner out: ", w.out
#                print "second in: ", s.inp
#                print "second out: ", s.out
                norm = npexp(-npnorm(x-w.inp)**2/(config.SIGMAE**2))
                res = norm*(w.out+npdot(w.A,x-w.inp))
                wc = npexp(-npnorm(x-s.inp)**2/(config.SIGMAE**2))
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
            norm = np.exp(-npnorm(x-w.inp)**2/(config.SIGMAE**2))
            res = norm*(w.out+npdot(w.A,x-w.inp))
            for nI, n in w.neig.iteritems():
                wc = np.exp(-npnorm(x-n.inp)**2/(config.SIGMAE**2))
                res += wc*(n.out+npdot(n.A,x-n.inp))
                norm += wc
            if norm != 0:
                return res/norm
            else:
                return res
