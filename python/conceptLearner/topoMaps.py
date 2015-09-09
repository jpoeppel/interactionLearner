# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:33:38 2015

@author: jpoeppel
"""

from network import Network, Node
import numpy as np
from numpy import dot as npdot
import math
from operator import itemgetter

EMAX = 0.001
ETA = 0.0
#For neighbours
SIGMAE = 0.05

WINNER = 0
NEIGHBOURS = 1
BESTTWO = 2
LINEAR = 3
PREDICTIONMODE = BESTTWO

#mask = np.array([4,5,6,7,8,9])
mask = range(11)

class ITM(Network):
    
    def __init__(self):
        super(ITM, self).__init__()
        pass
        
    def getWinners(self, x):
        if len(self.nodes)< 2:
            return None, None
        else:
            ds = sorted([(npdot(n.vec()-x,n.vec()-x), n) for n in self.nodes.values()], key=itemgetter(0))
            return ds[0][1], ds[1][1]
            
    def train(self, x):
        """
        Parameters
        ==========
        x : Node
        """
#        print "training: ", x.wOut
#        for n in self.nodes.values():
#            print "Nodes train: ", n.vec()
        name = self.idCounter
        x.name = name
        nearest, second = self.getWinners(x.vec())
        if nearest != None and second != None:
            nearest.adapt(x, ETA)
            self.addEdge(nearest.name, second.name)
            for n in nearest.neighbours.values():
                if n != second and npdot(nearest.vec()-second.vec(), n.vec()-second.vec()) < 0:
                    self.removeEdge(nearest.name, n.name)
            np.set_printoptions(precision=3,suppress=True)
#            print "talis: ", npdot(nearest.vec()-x.vec(),second.vec()-x.vec())
#            print "dist: ", np.linalg.norm(x.vec()-nearest.vec())
#            print "talisOut: ",  np.dot(nearest.wOut-x.wOut, second.wOut-x.wOut)
#            print "xvec: ", x.vec()
#            print "nearestvec: ", nearest.vec()
#            print "secondvec: ", second.vec()
            if npdot(nearest.vec()-x.vec(),second.vec()-x.vec()) > 0 and np.linalg.norm(x.vec()-nearest.vec()) > EMAX:
#            if np.dot(nearest.wOut-x.wOut, second.wOut-x.wOut) > 0 and np.linalg.norm(x.wOut-nearest.wOut) > EMAX:
                self.addNode(x)
#                x.adapt(nearest, ETA)
#                print "adding new node: ", x.wOut
#                print "Dot: ", np.dot(nearest.vec()-x.vec(),second.vec()-x.vec())
#                print "dist: ", np.linalg.norm(x.vec()-nearest.vec())
#                print "x.vec: {}, nearest.vec: {}".format(x.vec(), nearest.vec())
                self.addEdge(nearest.name, name)
            if np.linalg.norm(nearest.vec()-second.vec()) < 0.5*EMAX:
                print "removing node"
                self.removeNode(second)
        else:
            self.addNode(x)
            print "adding node because there are not enough: ", x.vec()
            
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
            elif PREDICTIONMODE == BESTTWO:
                return minNode.action, minNode.wIn, minNode.wOut #TODO correct later
            else:
                raise AttributeError("Unsupported Predictionmode used: ", PREDICTIONMODE)
        else:
#            print "No minNode found: NrNodes: {}".format(len(self.nodes))
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
            elif PREDICTIONMODE == BESTTWO:
                return minNode.action, minNode.wIn, minNode.wOut #TODO make properly
            else:
                raise AttributeError("Unsupported Predictionmode used: ", PREDICTIONMODE)
        else:
#            print "number of nodes: ", len(self.nodes)
#            print "last d: ", d
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
#        print "num Nodes predict: ", len(self.nodes)
#        for n in self.nodes.values():
#            print "Nodes predict: ", n.wOut
#        minDist = float('inf')
#        secDist = float('inf')
        minNode = None
        secNode = None
        
#        for n in self.nodes.values():
#            d = np.linalg.norm(n.vecInA()-wIn)
#            if d < minDist:
#                minDist = d
#                minNode = n
#            elif d < secDist:
#                secDist = d
#                secNode = n
        if len(self.nodes) > 0:
            ds = sorted([(npdot(n.vecInA()-wIn,n.vecInA()-wIn), n) for n in self.nodes.values()], key=itemgetter(0))
            minNode = ds[0][1]
        
        if minNode != None:
            if PREDICTIONMODE == WINNER:
#                print "minNode win: ", minNode.vecInA()
#                print "number of nodes: ", len(self.nodes)
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
                
                
                if len(self.nodes) > 1:
#                    print "minNode wIn: ", minNode.wIn
#                    print "minNode wout: ", minNode.wOut
#                    print "second wIn: ", ds[1][1].wIn
#                    print "second wout: ", ds[1][1].wOut
                    secNode = ds[1][1]
                    norm = math.exp(-np.linalg.norm(wIn-minNode.vecInA())**2/(SIGMAE**2))
                    res = norm*minNode.wOut
                    wc = math.exp(-np.linalg.norm(wIn-secNode.vecInA())**2/(SIGMAE**2))
                    res += wc*secNode.wOut
                    norm += wc
                    if norm != 0:
                        return res/norm
                    else:
                        return res
                else:
                    return minNode.wOut
        else:
#            print "No minNode found!"
#            print "number of nodes: ", len(self.nodes)
            return 0.0
            
if __name__ == "__main__":
    
    X = np.arange(0, 2*math.pi, 0.01)
    Y = np.sin(X)

    trainData = X[::10]
    trainLabel = Y[::10]    
    itm = ITM()
    
    for x,y in zip(trainData, trainLabel):
        n = Node(0, wIn=x, wOut=y)
        itm.train(n)
        
    error = 0.0
#    for x,y in zip(X,Y):
#        pred = itm.predict(x)
#        error += np.linalg.norm(y-pred)
        
    preds = np.array(map(itm.predict, X)).flatten()
    error = np.linalg.norm(preds-Y)/len(X)

        
    print "avg error: ", error/len(X)
    print "numNodes: {}, numTrainData: {}".format(len(itm.nodes), len(trainData))