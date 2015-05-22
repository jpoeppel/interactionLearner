# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:32:22 2015

@author: jpoeppel
"""

import numpy as np
from sets import Set
import copy

EQU = 0
CMP = 1

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
        self.tree = None
    
    def newNode(self, nodeName, pos=np.array([]), wIn=np.array([]), wOut = np.array([]), A = np.array([])):
        self.nodes[nodeName] = Node(nodeName, pos, wIn, wOut, A)
        self.idCounter += 1
    
    def addNode(self, node):
        self.nodes[node.name] = copy.deepcopy(node)
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
        self.ruleType = None
        self.load = []
        self.numOfElements = 0
        if elem != None:
            self.load.append(elem)
            
        
    def getChild(self, values):
        print "Child ruleKey: {}, ruleValue: {}, ruleType: {}".format(self.ruleKey, self.ruleValue, "EQU" if self.ruleType == EQU else "CMP")
        if not self.isLeave:
            print "splitting into left: {}, right: {}".format(self.leftChild.numOfElements, self.rightChild.numOfElements)
        if self.ruleType == EQU:
            if np.linalg.norm(values[self.ruleKey] - self.ruleValue) < 0.01:
                return self.leftChild
            else:
                return self.rightChild
        else:
            if np.all(values[self.ruleKey] < self.ruleValue):
                return self.leftChild
            else:
                return self.rightChild
            
#    def getBestChild(self, const, minima, maxima):
#        if const.has_key(self.ruleKey):
#            if np.all(const[self.ruleKey] < self.ruleValue):
#                return self.leftChild
#            else:
#                return self.rightChild
#        else:
#            
#                
#        if np.all(maxima[self.ruleKey] < self.ruleValue):
#            return self.leftChild
#        if np.all(minima[self.ruleKey] > self.ruleValue):
#            return self.rightChild
            
    def addElements(self, elements, leaves):
        self.numOfElements += len(elements)
        if self.isLeave:
            self.load.extend(elements)
            leaves.append(self)
            self.reorganise(leaves)
            
    def reorganise(self, leaves):
        if len(self.load) > 1:
#            print "addElement before Split"
            splitAttrib, splitValue = self.getSplitAttrib()
#            print "splitAttrib: ", splitAttrib
            if splitAttrib != None:
                lc = TreeNode(pre=self)
                rc = TreeNode(pre=self)
                self.leftChild = lc
                self.rightChild = rc
                #TODO get best Split already from getSplitAttrib!
                toAddLeft = []
                toAddRight = []
                for el in self.load:
#                    if np.all(el.maxima[splitAttrib] < splitValue):
                        #TODO This does not take the minima into consideration!!
                    if el.constants.has_key(splitAttrib):
                        if splitValue[1] == EQU:
                            if np.linalg.norm(el.constants[splitAttrib] - splitValue[0]) < 0.01:
                                toAddLeft.append(el)
#                                lc.addElement(el, leaves)
                            else:
                                toAddRight.append(el)
#                                rc.addElement(el, leaves)
                        elif splitValue[1] == CMP:
                            if np.all(el.constants[splitAttrib] <= splitValue[0]):
#                                lc.addElement(el, leaves)
                                toAddLeft.append(el)
                            else:
#                                rc.addElement(el, leaves)
                                toAddRight.append(el)
                        else:
                            raise NotImplementedError("ruleType: {} unknown".format(splitValue[1]))
                    else:
                        if splitValue[1] == CMP:
#                            lc.addElement(el, leaves)
#                            rc.addElement(el, leaves)
                            toAddLeft.append(el)
                            toAddRight.append(el)
                            self.numOfElements += 1
                        else:
#                            rc.addElement(el, leaves)
                            toAddRight.append(el)
                lc.addElements(toAddLeft, leaves)
                rc.addElements(toAddRight, leaves)
                self.isLeave = False
                leaves.remove(self)
                self.ruleKey = splitAttrib
                self.ruleValue = splitValue[0]
                self.ruleType = splitValue[1]
                self.load = []
            
    def addElement(self, elem, leaves):
        self.numOfElements += 1
        if self.isLeave:
            self.load.append(elem)
            leaves.append(self)
            self.reorganise(leaves)
        else:
            if elem.constants.has_key(self.ruleKey):
                if np.all(elem.constants[self.ruleKey] <= self.ruleValue):
                    self.leftChild.addElement(elem, leaves)
                else:
                    self.rightChild.addElement(elem, leaves)
            else:
                self.leftChild.addElement(elem, leaves)
                self.rightChild.addElement(elem, leaves)
        
        
    def getSharedConsts(self):
        sharedKeys = Set(self.load[0].constants)
        allKeys = Set(sharedKeys)
        for el in self.load:
            sharedKeys = sharedKeys.intersection(Set(el.constants.keys()))
            allKeys.update(el.constants.keys())
        difKeys = allKeys.difference(sharedKeys)
        return allKeys, difKeys
            
    def getSplitAttrib(self):
        splitValues = {}
        biggestMin = {}
        smallestMax = {}
        allKeys, difKeys = self.getSharedConsts()
#        for k in self.load[0].minima.keys():
        for k in allKeys:
            biggestMin[k] = -float('inf')
            smallestMax[k] = float('inf')
            for el in self.load:
                if el.constants.has_key(k):
                    if np.all(biggestMin[k] <= el.constants[k]):
                        biggestMin[k] = el.constants[k]
                    if np.all(smallestMax[k] > el.constants[k]):
                        smallestMax[k] = el.constants[k]
                
#                if np.all(biggestMin[k] < el.minima[k]):
#                    biggestMin[k] = el.minima[k]
#                if np.all(smallestMax[k] > el.maxima[k]):
#                    smallestMax[k] = el.maxima[k]
                    
            if np.linalg.norm(biggestMin[k] - smallestMax[k])< 0.01 and k in difKeys:
                splitValues[k] = (0.5*(biggestMin[k] + smallestMax[k]), EQU)
            else:
                splitValues[k] = (0.5*(biggestMin[k] + smallestMax[k]), CMP)
            
#        print "splitvalues: ", splitValues
        splits = {}
        for k,v in splitValues.items():
            splits[k] = 0
            for el in self.load:
#                if np.all(el.maxima[k] < v):
                if el.constants.has_key(k): # Acs that do not have that constant will be present in both children so do not count them here
                    if v[1] == EQU:
                        if np.linalg.norm(el.constants[k] - v[0]) < 0.01:
                            splits[k] += 1
                        else:
                            splits[k] -= 1
                    elif v[1] == CMP:
                        if np.all(el.constants[k] <= v[0]):
                            splits[k] += 1
                        else:
                            splits[k] -= 1
                    else:
                        raise NotImplementedError("ruleType: {} unknown".format(v[1]))
        bestSplit = min(splits.items(), key=lambda pair: abs(pair[1]))
        print "splits: ", splits
        print "load: ", [c.variables for c in self.load]
        print "load constants: ", [c.constants for c in self.load]
        print "load numRefs: ", [len(c.refCases) for c in self.load]
        print "Best split: {}, {}".format(bestSplit[0], splitValues[bestSplit[0]])
        if abs(bestSplit[1]) == len(self.load):
            return None, None
        else:
            return bestSplit[0], splitValues[bestSplit[0]]
        
        
class Tree(object):
    
    def __init__(self):
        self.leaves = []
        self.root = None
        
    def addElement(self, element, const, minima, maxima):
#        node = self.root
        if self.root != None:
            self.root.addElement(element, self.leaves)
        else:
#        while node != None and not node.isLeave:
#            node = node.getBestChild(const, minima, maxima)
#        
#        if node != None:
#            node.addElement(element, self.leaves)
#        else:
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
    

    def addElements(self, elements):
        self.root = TreeNode(pre=None)
        self.root.addElements(elements, self.leaves)
    
    def getElements(self, values):
        node = self.root
        while node != None and not node.isLeave:
            node = node.getChild(values)
        if node != None:
            return node.load
        else:
            return None
        
    