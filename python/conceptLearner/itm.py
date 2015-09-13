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

class Node(np.ndarray):
    __slots__=('out','id','neig','A')
    def __new__(cls, input_array, id=-1, output = np.zeros(1)):
        obj = np.asarray(input_array).view(cls)
        obj.out = output
        obj.A = np.zeros((len(output),len(input_array)))
        obj.neig = {}
        obj.id = id
        return obj
        
    def addNeighbour(self, nId, n):
        self.neig[nId] = n
        
    def remNeighbour(self,nId):
        del self.neig[nId]
        
    def __array_finalize__(self, obj):
#        pass
#        print "finalize"
        if obj is None: return
        self.out = getattr(obj, 'out', None)
        self.A = getattr(obj, 'A', None)
        self.neig = getattr(obj, 'neig', {})
        self.id = getattr(obj,'id', -1)
        
class Node2(object):
    __slots__=('inp','out','id','neig','A')
    def __init__(self, input_array, id=-1, output=np.zeros(1)):
        self.id = id
        self.inp =  np.asarray(input_array)
        self.out = output
        self.A = np.zeros((len(output),len(input_array)))
        self.neig = {}
        
    def addNeighbour(self, nId, n):
        self.neig[nId] = n
        
    def remNeighbour(self,nId):
        del self.neig[nId]


class ITM(object):
    
    def __init__(self):
        self.nodes = collections.OrderedDict()
        self.inserts = 0
#        self.nodes= {}
#        self.nodes = np.zeros # Try storing all nodes in one nparray that needs to be reshaped
        
        
        
    def update(self, x, y, etaIn=0.0, etaOut=0.0, etaA=0.0):
        #Get winners:
        if len(self.nodes) > 1:
#            numpyVals = Node(self.nodes.values())-x
            numpyVals = np.array([n.inp for n in self.nodes.values()])-x
            sortedIndices = np.argsort(np.linalg.norm(numpyVals, axis=1))
#            sortedIndices = np.argsort([npdot(n-x,n-x) for n in self.nodes.values()])
#            ds = sorted([(np.linalg.norm(n.inp-x), n) for n in self.nodes.values()], key=itemgetter(0))
#            w = ds[0][1]
#            s = ds[1][1]
            w = self.nodes[sortedIndices[0]]
            wI =  w.id
            s = self.nodes[sortedIndices[1]]
            sI = s.id
            #Adapt winner
#            dwIn = etaIn*(x-w)
#            w += dwIn
#            w.out += etaOut*(y-w.out+npdot(w.A,x-w)) + np.dot(w.A,dwIn)
#            w.A = etaA*np.outer((y-w.out+npdot(w.A,x-w)), (x-w)/np.linalg.norm(x-w))
            #Add edge
            w.addNeighbour(sI,s)
            s.addNeighbour(wI,w)
#            #Check neighbours
            for nI, n in w.neig.items():
                if n.id != s.id and npdot(w.inp-s.inp,n.inp-s.inp) < 0:
#                if n.id != s.id and npdot(w-s,n-s) < 0:
                    n.remNeighbour(wI)
                       
                    if len(n.neig) == 0:
                         #delete n
                        self.deleteNode(nI)
                    w.remNeighbour(nI)
            #Check for new node
            if npdot(w.inp-x,s.inp-x) > 0 and np.linalg.norm(x-w.inp) > EMAX:
#            if npdot(w-x,s-x) > 0 and np.linalg.norm(w-x) > EMAX:
                nI = len(self.nodes)
                n= Node2(x,nI,y)
                self.nodes[nI] = n
                self.inserts += 1
#                w.addNeighbour(nI, n)
#                n.addNeighbour(wI, w)
#            
#            if np.linalg.norm(w-s) < 0.5*EMAX:
#                if len(self.nodes) > 2:
#                    self.deleteNode(sI)             
        else:
#            To few nodes
            print "adding node"
            nI = len(self.nodes)
            self.nodes[nI] = Node2(x,nI,y)
            self.inserts += 1
        pass
        print "end of update"
    
    def deleteNode(self, nodeId):
        for nI, n in self.nodes[nodeId].neig.items():
            n.remNeighbour(nodeId)
            if len(n.neig) == 0:
                self.deleteNode(nI)
        del self.nodes[nodeId]
    
    def test(self, x):
        numpyVals = np.array([n.inp for n in self.nodes.values()])-x
        sortedIndices = np.argsort(np.linalg.norm(numpyVals, axis=1))
        return self.nodes[sortedIndices[0]].out
#        ds = sorted([(npdot(n.inp-x,n.inp-x), n) for n in self.nodes.values()], key=itemgetter(0))
#        return ds[0][1].out

    
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
#    
