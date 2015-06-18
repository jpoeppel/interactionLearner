# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:33:38 2015

@author: jpoeppel
"""

from network import Network
import numpy as np
import math

EMAX = 0.05
ETA = 0.01
SIGMAE = 0.5

WINNER = 0
NEIGHBOURS = 1
BESTTWO = 2
LINEAR = 3
PREDICTIONMODE = LINEAR

class ITM(Network):
    
    def __init__(self):
        super(ITM, self).__init__()
        pass
        
    def getWinners(self, x):
        if len(self.nodes)< 2:
            return None, None
        else:
            minDist = float('inf')
            secDist = float('inf')
            minNode = None
            secNode = None
            for n in self.nodes.values():
                d = np.linalg.norm(n.vec()-x)
                if d < minDist:
                    minDist = d
                    minNode = n
                elif d < secDist:
                    secDist = d
                    secNode = n
            return minNode, secNode
            
    def train(self, x):
        """
        Parameters
        ==========
        x : Node
        """
        name = self.idCounter
        x.name = name
        nearest, second = self.getWinners(x.vec())
        if nearest != None and second != None:
            nearest.adapt(x, ETA)
            self.addEdge(nearest.name, second.name)
            for n in nearest.neighbours.values():
                if n != second and np.dot(nearest.vec()-second.vec(), n.vec()-second.vec()) < 0:
                    self.removeEdge(nearest.name, n.name)
            if np.dot(nearest.vec()-x.vec(),second.vec()-x.vec()) > 0 and np.linalg.norm(x.vec()-nearest.vec()) > EMAX:
                self.addNode(x)
#                x.adapt(nearest, ETA)
                self.addEdge(nearest.name, name)
            if np.linalg.norm(nearest.vec()-second.vec()) < 0.5*EMAX:
                self.removeNode(second)
        else:
            self.addNode(x)
            
    def getAction(self, wOut):
        if not hasattr(wOut, "__len__"):
            wOut = np.array([wOut])
        minDist = float('inf')
        minNode = None
        
        for n in self.nodes.values():
#            print "n.wIn: {}, n.wOut: {}, wIn: {}, wOut: {}".format(n.wIn, n.wOut, wIn, wOut)
            d = np.linalg.norm(wOut-n.wOut)
            if d < minDist:
                minDist = d
                minNode = n
        if minNode != None:
            if PREDICTIONMODE == WINNER:
                return minNode.action, minNode.wIn, minNode.wOut
            elif PREDICTIONMODE == NEIGHBOURS:
                raise NotImplementedError()
            elif PREDICTIONMODE == LINEAR:
                return minNode.action, minNode.wIn, minNode.wOut
            else:
                raise AttributeError("Unsupported Predictionmode used: ", PREDICTIONMODE)
        else:
            print "No minNode found: NrNodes: {}".format(len(self.nodes))
            return None, None, None

    def getBestAbsAction(self, wOut):
        if not hasattr(wOut, "__len__"):
            wOut = np.array([wOut])
        minDist = float('inf')
        minNode = None
        
        for n in self.nodes.values():
#            print "n.wIn: {}, n.wOut: {}, wIn: {}, wOut: {}".format(n.wIn, n.wOut, wIn, wOut)
            d = np.linalg.norm(np.abs(wOut)-np.abs(n.wOut))
            if d < minDist:
                minDist = d
                minNode = n
        if minNode != None:
            if PREDICTIONMODE == WINNER:
                return minNode.action, minNode.wIn, minNode.wOut
            elif PREDICTIONMODE == NEIGHBOURS:
                raise NotImplementedError()
            elif PREDICTIONMODE == LINEAR:
                return minNode.action, minNode.wIn, minNode.wOut
            else:
                raise AttributeError("Unsupported Predictionmode used: ", PREDICTIONMODE)
        else:
            print "number of nodes: ", len(self.nodes)
            print "last d: ", d
            return None, None, None
        
    def predictAction(self, wIn, wOut):
        if not hasattr(wOut, "__len__"):
            wOut = np.array([wOut])
        w = np.concatenate((wIn,wOut))
        minDist = float('inf')
        minNode = None
        
        for n in self.nodes.values():
#            print "n.wIn: {}, n.wOut: {}, wIn: {}, wOut: {}".format(n.wIn, n.wOut, wIn, wOut)
            d = np.linalg.norm(n.vecInOut()-w)
#            d = np.linalg.norm(wOut-n.wOut)
            if d < minDist:
                minDist = d
                minNode = n
        if minNode != None:
#            print "minNode wIn: ", minNode.wIn
#            print "minNode action: ", minNode.action
#            print "minNode wOut: ", minNode.wOut
            if PREDICTIONMODE == WINNER:
                return minNode.action
            elif PREDICTIONMODE == NEIGHBOURS:
#                norm = math.exp(-np.linalg.norm(w-minNode.vecInOut())**2/(SIGMAE**2))
                norm = math.exp(-0.5*np.linalg.norm(wOut-minNode.wOut)/(SIGMAE**2))
                res = norm*minNode.action
                for n in minNode.neighbours.values():
#                    wc = math.exp(-np.linalg.norm(w-n.vecInOut())**2/(SIGMAE**2))
                    wc = math.exp(-0.5*np.linalg.norm(wOut-n.wOut)/(SIGMAE**2))
                    norm += wc
                    res += wc * n.action
                return res/norm
            elif PREDICTIONMODE == LINEAR:
                return minNode.action #TODO make real linear
                
    def predict(self, wIn):
        
        minDist = float('inf')
        secDist = float('inf')
        minNode = None
        secNode = None
        for n in self.nodes.values():
            d = np.linalg.norm(n.vecInA()-wIn)
            if d < minDist:
                minDist = d
                minNode = n
            elif d < secDist:
                secDist = d
                secNode = n
        if minNode != None:
            if PREDICTIONMODE == WINNER:
                return minNode.wOut
            elif PREDICTIONMODE == LINEAR:
#                print "MinNode: ", minNode
                return minNode.wOut + minNode.A.dot(wIn-minNode.vecInA())
            elif PREDICTIONMODE == NEIGHBOURS:                    
                norm = math.exp(-np.linalg.norm(wIn-minNode.vecInA())**2/(SIGMAE**2))
                res = norm*minNode.wOut
                for n in minNode.neighbours.values():
                    wc = math.exp(-np.linalg.norm(wIn-n.vecInA())**2/(SIGMAE**2))
                    norm += wc
                    res += wc * n.wOut
                    
                return res/norm
            elif PREDICTIONMODE == BESTTWO:
                if secNode != None:
                    norm = math.exp(-np.linalg.norm(wIn-minNode.vecInA())**2/(SIGMAE**2))
                    res = norm*minNode.wOut
                    wc = math.exp(-np.linalg.norm(wIn-secNode.vecInA())**2/(SIGMAE**2))
                    res += wc*secNode.wOut
                    norm += wc
                    return res/norm
                else:
                    return minNode.wOut
  