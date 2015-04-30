# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:32:22 2015

@author: jpoeppel
"""

import numpy as np

class Node(object):
    def __init__(self, name, pos=np.array([]), wIn=np.array([]), action = np.array([]), wOut = np.array([]), A = np.array([])):
        self.name = name
        self.pos = pos
        self.neighbours = {}
        self.wIn = wIn
        if hasattr(action, "__len__"):
            self.action = action
            lA = len(action)
        else:
            self.action = np.array([action])
            lA = 1
        
        if hasattr(wOut, "__len__"):
            self.wOut = wOut
            lO = len(wOut)
        else:
            self.wOut = np.array([wOut])
            lO = 1
        self.A = np.zeros((lO, len(wIn)+lA))
    
    def addNeighbour(self,n):
        self.neighbours[n.name] = n
        
    def removeNeighbour(self, name):
        del self.neighbours[name]
    
    def adapt(self, x, eta):
        dwIn = eta*(x.wIn - self.wIn)
        self.wIn += dwIn
        da = eta*(x.action - self.action)
        self.action += da
        dwInA = np.concatenate((dwIn,da))
        
        er = x.wOut-(self.wOut + self.A.dot(x.vecInA()-self.vecInA()))
        dwOut =  eta*er + self.A.dot(dwInA)
        self.wOut += dwOut
        d = x.vecInA()-self.vecInA()
        self.A += eta*np.outer(er,d/(np.linalg.norm(d)**2))
        
    def vec(self):
        return np.concatenate((self.wIn, self.action, self.wOut))
        
    def vecInOut(self):
        return np.concatenate((self.wIn, self.wOut))
        
    def vecInA(self):
        return np.concatenate((self.wIn, self.action))

class Network(object):
    def __init__(self):
        self.nodes = {}
        self.idCounter = 0
    
    def newNode(self, nodeName, pos=np.array([]), wIn=np.array([]), wOut = np.array([]), A = np.array([])):
        self.nodes[nodeName] = Node(nodeName, pos, wIn, wOut, A)
        self.idCounter += 1
    
    def addNode(self, node):
        self.nodes[node.name] = node
        self.idCounter += 1
    
    def addEdge(self, fromNode, toNode):
        self.nodes[fromNode].addNeighbour(self.nodes[toNode])
        self.nodes[toNode].addNeighbour(self.nodes[fromNode])
    
    def removeEdge(self, fromNode, toNode):
        self.nodes[fromNode].removeNeighbour(toNode)
        self.nodes[toNode].removeNeighbour(fromNode)
        if len(self.nodes[fromNode].neighbours) == 0:
            del self.nodes[fromNode]
        if len(self.nodes[toNode].neighbours) == 0:
            del self.nodes[toNode]
            
    def removeNode(self, node):
        for n in node.neighbours.values():
            self.removeEdge(node.name, n.name)
        if self.nodes.has_key(node.name):
            del self.nodes[node.name]
    