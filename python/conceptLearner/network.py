# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:32:22 2015

@author: jpoeppel
"""

import numpy as np
from sets import Set

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
            
class TreeNode(object):
    
    def __init__(self, pre = None, elem = None):
        self.predecessor = pre
        self.leftChild = None
        self.rightChild = None
        self.isLeave = True
        self.ruleKey = None
        self.ruleValue = None
        self.load = []
        if elem != None:
            self.load.append(elem)
            
        
    def getChild(self, values):
        print "Child ruleKey: {}, ruleValue: {}".format(self.ruleKey, self.ruleValue)
        if np.all(values[self.ruleKey] < self.ruleValue):
            return self.leftChild
        else:
            return self.rightChild
            
    def getBestChild(self, const, minima, maxima):
        if const.has_key(self.ruleKey):
            if np.all(const[self.ruleKey] < self.ruleValue):
                return self.leftChild
            else:
                return self.rightChild
        else:
            
#                
#        if np.all(maxima[self.ruleKey] < self.ruleValue):
#            return self.leftChild
#        if np.all(minima[self.ruleKey] > self.ruleValue):
#            return self.rightChild
            
    def addElement(self, elem):
        self.load.append(elem)
        if len(self.load) > 1:
            print "addElement before Split"
            splitAttrib, splitValue = self.getSplitAttrib()
            print "splitAttrib: ", splitAttrib
            if splitAttrib != None:
                lc = TreeNode(pre=self)
                rc = TreeNode(pre=self)
                self.leftChild = lc
                self.rightChild = rc
                #TODO get best Split already from getSplitAttrib!
                #TODO This does not take the minima into consideration!!
                for el in self.load:
#                    if np.all(el.maxima[splitAttrib] < splitValue):
                    if el.constants.has_key(splitAttrib):
                        if np.all(el.constants[splitAttrib] < splitValue):
                            lc.addElement(el)
                        else:
                            rc.addElement(el)
                    else:
                        lc.addElement(el)
                        rc.addElement(el)
                self.isLeave = False
                self.ruleKey = splitAttrib
                self.ruleValue = splitValue
                self.load = None
                return lc, rc
            
        return None, None
        
    def getSharedConsts(self):
        constsKeys = Set(self.load[0].constants)
        for el in self.load:
            constsKeys = constsKeys.intersection(Set(el.constants.keys()))
        return constsKeys
            
    def getSplitAttrib(self):
        splitValues = {}
        biggestMin = {}
        smallestMax = {}
        for k in self.load[0].minima.keys():
#        for k in self.getSharedConsts():
            biggestMin[k] = -float('inf')
            smallestMax[k] = float('inf')
            for el in self.load:
                if el.constants.has_key(k):
                    if np.all(biggestMin[k] < el.constants[k]):
                        biggestMin[k] = el.constants[k]
                    if np.all(smallestMax[k] > el.constants[k]):
                        smallestMax[k] = el.constants[k]
#                if np.all(biggestMin[k] < el.minima[k]):
#                    biggestMin[k] = el.minima[k]
#                if np.all(smallestMax[k] > el.maxima[k]):
#                    smallestMax[k] = el.maxima[k]
                    
            splitValues[k] = 0.5*(biggestMin[k] + smallestMax[k])
            
#        print "splitvalues: ", splitValues
        splits = {}
        for k,v in splitValues.items():
            splits[k] = 0
            for el in self.load:
#                if np.all(el.maxima[k] < v):
                if el.constants.has_key(k): # Acs that do not have that constant will be present in both children so do not count them here
                    if np.all(el.constants[k] < v):
                        splits[k] += 1
                    else:
                        splits[k] -= 1
        bestSplit = min(splits.items(), key=lambda pair: abs(pair[1]))
#        print "splits: ", splits
#        print "load: ", [c.variables for c in self.load]
#        print "load numRefs: ", [len(c.refCases) for c in self.load]
#        print "load maxima: ", [c.maxima for c in self.load]
#        print "load minima: ", [c.minima for c in self.load]
#        print "Best split: {}, {}".format(bestSplit[0], splitValues[bestSplit[0]])
        if abs(bestSplit[1]) == len(self.load):
            return None, None
        else:
            return bestSplit[0], splitValues[bestSplit[0]]
        
        
class Tree(object):
    
    def __init__(self):
        self.leaves = []
        self.root = None
        
    def addElement(self, element, const, minima, maxima):
        node = self.root
        while node != None and not node.isLeave:
            node = node.getBestChild(const, minima, maxima)
        
        if node != None:
            lc, rc = node.addElement(element)
            if lc != None:
                self.leaves.remove(node)
                self.leaves.append(lc)
                self.leaves.append(rc)
        else:
            self.root = TreeNode(pre=None, elem=element)
            self.leaves.append(self.root)
            
            
    
    def removeElement(self, element):
        targetLeave = None
        for l in self.leaves:
            for load in l.load:
                if load == element:
                    targetLeave = l
                    break
        if targetLeave != None:
            targetLeave.load.remove(element)
        #Check if tree needs to be reordered because of empty leave
        if len(targetLeave.load) == 0:            
            self.leaves.remove(targetLeave)
            pre = targetLeave.predecessor
            if pre != None:
                other = pre.leftChild if pre.leftChild != targetLeave else pre.rightChild
                other.predecessor = pre.predecessor
                if pre.predecessor.leftChild == pre:
                    pre.predecessor.leftChild = other
                else:
                    pre.predecessor.rightChild = other
    
    def getElements(self, values):
        print "get Elements for values: ", values
        node = self.root
        while node != None and not node.isLeave:
            node = node.getChild(values)
        if node != None:
            return node.load
        else:
            return None
        
    