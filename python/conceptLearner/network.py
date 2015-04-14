# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:32:22 2015

@author: jpoeppel
"""

import numpy as np

class Node(object):
    def __init__(self, name, pos=np.array([]), wIn=np.array([]), wOut = np.array([]), A = np.array([])):
        self.name = name
        self.pos = pos
        self.neighbours = {}
        self.wIn = wIn
        if hasattr(wOut, "__len__"):
            self.wOut = wOut
        else:
            self.wOut = np.array([wOut])
        self.A = A
    
    def addNeighbour(self,n):
        self.neighbours[n.name] = n
        
    def removeNeighbour(self, name):
        del self.neighbours[name]
    
    def adapt(self, x, eta):
        self.wIn += eta*(x.wIn - self.wIn)
        self.wOut += eta*(x.wOut - self.wOut)
        
    def vec(self):
        return np.concatenate((self.wIn, self.wOut))

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
    