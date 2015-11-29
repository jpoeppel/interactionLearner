#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:06:08 2015
Implementation of the Adapted Instantaneous Topological Mapping
@author: jpoeppel
"""

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
from configuration import WINNER, BESTTWO, NEIGHBOUR
from configuration import config
        
class Node(object):
    """
        Node class representing the network nodes.
        Uses slots for better efficiency.
    """
    __slots__=('inp','out','id','neig','A')
    def __init__(self, input_array, id=-1, output=npzeros(1)):
        """
            Constructor setting up local variables.
            
            Paramters
            --------
            input_array : np.ndarray
                Input vector of the node
            id : int
                Identifier of the node
            output : np.ndarray
                Output of the node
        """
        self.id = id
        self.inp =  npcopy(input_array)
        self.out = npcopy(output)
        self.A = npzeros((len(output),len(input_array)))
        self.neig = {}
        
    def addNeighbour(self, nId, n):
        """
            Add the given node as neighbour
            
            Parameters
            ----------
            nId : int
                Identifier of the new neighbour
            n : Node
                Reference to the new neighbour
            
        """
        self.neig[nId] = n
        
    def remNeighbour(self,nId):
        """
            Removes the neighbour with the given identifier.
            
            Parameters
            ----------
            nId : int
                Identifier of the neighbour to be deleted
        """
        del self.neig[nId]
        
    def __repr__(self):
        """
            String representation. Currently only the id is used.
        """
        return str(self.id)


class AITM(object):
    """
        Adapted instantenous topological mapping implementation.
    """
    
    def __init__(self):
        """
            Constructor setting up the AITM. 
            An orderedDict is used to store the nodes to have O(1)
            access times while preserving insertion order. This allows
            using the index in the (ordered) values to reference the
            node using an id-list.
        """
        self.nodes = collections.OrderedDict()
        self.ids = []
        self.idCounter = 0
        self.valAr = nparray([n.inp for n in self.nodes.values()])
        self.inserts = 0
        self.updateCalls = 0
        
        
    def update(self, x, y, etaIn=0.0, etaOut=0.0, etaA=0.0, testMode=None):
        """
            Updates the AITM
            
            Parameters
            ----------
            x : np.ndarray
                Input vector of the new data point
            y : np.ndarray
                Output vector of the new data point
            etaIn : float
                Learning rate to adapt node inputs
            etaOut : float
                Learning rate to adapt node outputs
            etaA : float
                Learning rate to adapt linear interpolation matrix A
            testMode : int (0,1,2) or None, optional
                Mode used for computing the networks output. 0 = Winner, 1 = BestTwo, 2 = Neighbour
                None uses the default which is defined by config.TESTMODE
        """
        self.updateCalls += 1
        
        if len(self.nodes) > 1:
            #Get winners:
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
            #Direct access is a little quicker
            w.neig[sI] = s
            s.neig[wI] = w
            #Get expected output according to currently used prediction scheme
            expOut = self.test(x, sortedIndices, testMode)
            #Check neighbours
            for nI, n in w.neig.items():
                if n.id != s.id and npdot(wsdif,n.inp-s.inp) < 0:
                    n.remNeighbour(wI)
                       
                    if len(n.neig) == 0:
                         #delete n
                        self.deleteNode(nI)
                    w.remNeighbour(nI)
            #Check for new node
            outNorm = npnorm(y)
            if outNorm != 0:
                outputDim = npfloor(nplog10(outNorm))#-1
                if outputDim == 0:
                    outputDim -= 1
            else:
                outputDim = -3
            if npnorm(expOut-y) > 10**outputDim:
                #Only insert new node if inputs are different enough
                if npnorm(w.inp-x) > config.EMAX:
                    nI = self.idCounter
                    n= Node(x,nI,y)
                    self.nodes[nI] = n
                    self.inserts += 1
                    self.ids.append(nI)
                    self.idCounter += 1
                    self.valAr = nparray([node.inp for node in self.nodes.values()])
#                    w.addNeighbour(nI, n)
#                    n.addNeighbour(wI, w)
                    w.neig[nI] = n
                    n.neig[wI] = w
                else:
                    #Adapt winning node's output
                    w.out += 0.5*(y-w.out)
            #Check if winner and second are too close to each other
            if npdot(wsdif,wsdif) < config.EMAX_2:
                if len(self.nodes) > 2:
                    self.deleteNode(sI)             
        else:
            #To few nodes
            nI = self.idCounter
            self.nodes[nI] = Node(x,nI,y)
            self.ids.append(nI)
            self.idCounter += 1
            self.valAr = nparray([n.inp for n in self.nodes.values()])
            self.inserts += 1
            
    
    def deleteNode(self, nodeId):
        """
            Deletes a node from the network. Makes sure to remove all neighbour references to the
            deleted node as well. Only deletes if more than 2 nodes are in the network
            
            Parameters
            ----------
            nodeId : int
                Identifier of the node to be deleted
        """
        for nI, n in self.nodes[nodeId].neig.iteritems():
            n.remNeighbour(nodeId)
            if len(n.neig) == 0 and len(self.nodes) > 2:
                self.deleteNode(nI)
        del self.nodes[nodeId]
        self.valAr = nparray([n.inp for n in self.nodes.values()])
        self.ids.remove(nodeId)
    
    def test(self, x, sortedIndices = None, testMode=None):
        """
            Query function for the network.
            
            Paramters
            --------
            x : np.ndarray
                Input that is to be tested
            sortedIndices : np.ndarray, optional
                Array filled with sorted indices. If given, the nodes will not be sorted against
                x anymore.
            testMode : int (0,1,2) or None, optional
                Mode used for computing the networks output. 0 = Winner, 1 = BestTwo, 2 = Neighbour
                None uses the default which is defined by config.TESTMODE
        """
        if len(self.nodes) == 0:
            return 0
        if sortedIndices == None:
            numpyVals = self.valAr - x
            sortedIndices = npargsort(npnorm(numpyVals, axis=1))
        if testMode == None:
            testMode = config.TESTMODE
        
        if testMode == WINNER:
            w =  self.nodes[self.ids[sortedIndices[0]]]
            return w.out+npdot(w.A,x-w.inp)
        elif testMode == BESTTWO:
            if len(self.nodes) > 1:
                
                w = self.nodes[self.ids[sortedIndices[0]]]           
                s = self.nodes[self.ids[sortedIndices[1]]]
                
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
            norm = npexp(-npnorm(x-w.inp)**2/(config.SIGMAE**2))
            res = norm*(w.out+npdot(w.A,x-w.inp))
            for nI, n in w.neig.iteritems():
                wc = npexp(-npnorm(x-n.inp)**2/(config.SIGMAE**2))
                res += wc*(n.out+npdot(n.A,x-n.inp))
                norm += wc
            if norm != 0:
                return res/norm
            else:
                return res
