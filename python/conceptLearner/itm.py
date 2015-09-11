#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:06:08 2015

@author: jpoeppel
"""

import numpy as np

EMAX = 0.001

class Node(np.ndarray):
    
    def __new__(cls, input_array, id=-1, output = np.zeros(1)):
        obj = np.asarray(input_array).view(cls)
        obj.out = output
        obj.A = np.zeros((len(input_array),len(output)))
        obj.neig = {}
        obj.id = id
        return obj
        
    def addNeighbour(self, nId, n):
        self.neig[nId] = n
        
    def remNeighbour(self,nId):
        del self.neig[nId]
        if len(self.neig) == 0:
            return False
        return True
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.out = getattr(obj, 'out', None)
        self.A = getattr(obj, 'A', None)
        self.neig = getattr(obj, 'neig', {})
        self.id = getattr(obj,'id', -1)

class ITM(object):
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        
        
    def update(self, x, y, etaIn=0.0, etaOut=0.0, etaA=0.0):
        #Get winners:
        if len(self.nodes) > 1:
            numpyVals = np.array(self.nodes.values())
            sortedIndices = np.argsort(np.linalg.norm(numpyVals, axis=1))
            w = numpyVals[sortedIndices[0]]
            wI =  w.id
            s = numpyVals[sortedIndices[1]]
            sI = s.id
            #Adapt winner
            dwIn = etaIn*(x-w)
            w += dwIn
            w.out += etaOut*(y-w.out+np.dot(w.A,x-w)) + np.dot(w.A,dwIn)
            w.A = etaA*np.dot((y-w.out+np.dot(w.A,x-w)), (x-w)/np.linalg.norm(x-w))
            
            #Add edge
            w.addNeighbour(sI)
            s.addNeighbour(wI)
            #Check neighbours
            for nI, n in w.neig.items():
                if np.dot(w-s,n-s) < 0:
                    if n.remNeighbour(wI):
                        #delete n
                        self.deleteNode(nI)
                    w.remNeighbour(nI)
            #Check for new node
            if np.dot(w-x,s-x) > 0 and np.linalg.norm(x-w) > EMAX:
                nI = len(self.nodes)
                n= Node(x,nI,y)
                self.nodes[nI] = n
                w.addNeighbour(nI, n)
                n.addNeighbour(wI, w)
            
            if np.linalg.norm(w,s) < 0.5*EMAX:
                if len(self.nodes) > 2:
                    self.deleteNode(sI)             
        else:
            #To few nodes
            nI = len(self.nodes)
            self.nodes[nI] = Node(x,nI,y)
            self.edges[nI] = []
        pass
    
    def deleteNode(self, nodeId):
        for nI, n in self.nodes[nodeId].neig:
            if n.remNeighbour(nodeId):
                self.deleteNode(nI)
        del self.nodes[nodeId]
    
    def test(self, x):
#        sortedNodes = np.sort(np.linalg.norm(np.array(self.nodes.values()), axis=1))
        sortedNodes = np.sort(self.nodes.values())
        return sortedNodes[0].out
    
if __name__ == "__main__":
    a = [1,2,3]
    b = [4,5,6]
    n = Node(a)
    n2 = Node(b)
    print n
    print type(n)
    c = np.array([n,n2])
    print c[1]
    print type(c)
    print type(c[0])
