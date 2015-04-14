# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:33:38 2015

@author: jpoeppel
"""

from network import Network
import numpy as np
import math

EMAX = 0.5
ETA = 0.1
SIGMAE = 0.5

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
                self.addEdge(nearest.name, name)
            if np.linalg.norm(nearest.vec()-second.vec()) < 0.5*EMAX:
                self.removeNode(second)
        else:
            self.addNode(x)
                
    def predict(self, wIn):
        minDist = float('inf')
        secDist = float('inf')
        minNode = None
        for n in self.nodes.values():
            d = np.linalg.norm(n.wIn-wIn)
            if d < minDist:
                minDist = d
                minNode = n
        if minNode != None:
#            res = minNode.wOut
#            norm = 1
#            for n in minNode.neighbours.values():
#                wc = math.exp(-np.linalg.norm(wIn-n.wIn)**2/(SIGMAE**2))
#                norm += wc
#                res += wc * n.wOut
#            return res/norm
            return minNode.wOut
                
            
class SOM(Network):
  
  def __init__(self, numNodes=np.array([]),inDim=1, outDim=1):  
    s = np.size(numNodes)
    self.numNodes = numNodes
    if s==1:
      for j in range(numNodes[0]):
        self.newNode(j,pos= np.array([0,j]), wIn=np.random.rand(2)*2-1,wOut=np.random.rand(1)*2-1, A=np.random.rand(outDim, inDim))
      for i in range(numNodes-1):
        self.addEdge(i,i+1)
    else:
      w=numNodes[1]
      for i in range(numNodes[0]):
        for j in range(numNodes[1]):
            self.newNode(i+j*w,pos= np.array([i,j]), wIn=np.random.rand(inDim)*2-1,wOut=np.random.rand(outDim)*2-1, A=np.random.rand(outDim, inDim))
      
      for i in range(numNodes[0]):
        for j in range(numNodes[1]):
          if j != numNodes[1]-1:
            self.addEdge(i+j*w,i+(j+1)*w)
          if i != numNodes[0]-1:
            self.addEdge(i+j*w,i+1+j*w)
      
            
  def _getWinner(self, data):
    minDistance = float('Inf')
    minNode = None
    for n in self.nodes.values():
      d=np.linalg.norm(n.wIn-data,2)
      if d < minDistance:
        minDistance = d
        minNode = n
    return minNode
  
  def _neighbourFunc(self, node, winner,i,num):
    return exp(-np.linalg.norm(winner.pos-node.pos)/(2*(sigmaS-i*(sigmaS-sigmaE)/num)**2))


  def train(self, data):
    print "Training..."
    i=0.0
    num = len(data)
    for d in data:
      x1,x2,y = d
      x = np.array([x1,x2])
      n = self._getWinner(x)
      #er = (y-self.ask(x,usedInterpolation))
      
      #print "er:", er
      for node in self.nodes.values():
        er = y-(node.wOut + node.A.dot(x-node.wIn))
        hcn = self._neighbourFunc(node,n,i,num)
        #print "input:", x
        #print "wIn:", node.wIn
        #print "hcn:",hcn
        dwIn = (etaInS-i*(etaInS-etaInE)/num)*hcn*(x - node.wIn)
        #print "dwIn:" ,dwIn
        node.wIn += dwIn
        dwOut =  (etaOutS-i*(etaOutS-etaOutE)/num)*hcn*er + node.A.dot(dwIn)
        node.wOut += dwOut
        d = x-node.wIn
        node.A += etaA*hcn*np.outer(er,d/(np.linalg.norm(d)**2))
      i+=1.0

    print "Training done."

  def _getNeighbourhood(self,x,neighbourhood):
    if neighbourhood==1:
      winner = self._getWinner(x)
      tmp = winner.neighbours.values()
      tmp.append(winner)
      return tmp
    elif neighbourhood==2:
      tmp = []
      i=0
      while i < len(self.nodes.keys()):
        j = 0
        while j< len(tmp) and np.linalg.norm(tmp[j].wIn-x,2) < np.linalg.norm(self.nodes[i].wIn-x):
          j+=1
        tmp.insert(j, self.nodes[i])
        i+=1
      return tmp[:4]
    elif neighbourhood==3:
      return self.nodes.values()
    else:
      print "Neighbourhood not known"
      
      

  def ask(self, x, interpolation=1):
    if interpolation==1:
      return self._getWinner(x).wOut
    elif interpolation==2:
      norm = 0
      res = 0
      for n in self._getNeighbourhood(x,usedNeighbourhood):
        wc = exp(-np.linalg.norm(x-n.wIn)**2/(sigmaE**2))
        norm += wc
        res += wc * n.wOut
      return res/norm
    elif interpolation==3:
      n = self._getWinner(x)
      return n.wOut + n.A.dot(x-n.wIn)
    elif interpolation==4:
      norm = 0
      res = 0
      for n in self._getNeighbourhood(x,usedNeighbourhood):
        wc = exp(-np.linalg.norm(x-n.wIn)**2/(sigmaE**2))
        norm += wc
        res += wc * (n.wOut + n.A.dot(x-n.wIn))
      return res/norm
    else:
      print "Interpolation method not known"