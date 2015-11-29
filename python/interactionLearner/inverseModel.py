# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:31:29 2015
Implementation of the Averaging Inverse Model
@author: jpoeppel
"""

import numpy as np
from configuration import config
from operator import itemgetter

GREEDY_TARGET = True

class MetaNode(object):
    """
        Prototype for a given feature dimension and direction.
    """

    def __init__(self):
        """
            Setup of the local dictionaries to store the combination specific preconditions,
            weights, numbers and the size of the input features.
        """
        self.signCombinations= {}
        self.signCombinationSums= {}
        self.signCombinationNumbers = {}
        self.lenPreCons = 0

    def train(self, pre, dif):
        """
            Update function for the prototype. Determines the sign combination of the given
            input and updates the local dictionaries.
            
        Parameters
        ----------
        pre : np.ndarray
            Vector of preconditions
        dif : float
            Absolut difference value of the feature
        """
        #Compare incoming pres and find the things they have in common/are relevant for a given dif
        lPre = len(pre)
        self.lenPreCons = lPre
        curSigCom = []
        for i in xrange(lPre):
            if pre[i] < -config.metaNodeThr:
                curSigCom.append('-1')
            elif pre[i] > config.metaNodeThr:
                curSigCom.append('1')
            else:
                curSigCom.append('0')
        curSigCom = ";".join(curSigCom)
            
        #Has the problem of creating too many small combinations that might 
        # break the weighting
        # "Solved" by merging similar combinations during retrieval
        if curSigCom in self.signCombinations:
            self.signCombinations[curSigCom] += dif
            self.signCombinationSums[curSigCom] += dif*pre
            self.signCombinationNumbers[curSigCom] += 1
        else:
            self.signCombinations[curSigCom] = dif
            self.signCombinationSums[curSigCom] = dif*pre
            self.signCombinationNumbers[curSigCom] = 1
            
    def mergeCombinations(self):
        """
            Function that tries to merge equivalent combinations. A merged combination
            adds the parts (weight, sum and number) of its components.
            
            Returns
            -------
            resWeights: Dict
                Dictionary containing the weights of the merged combinations
            resSums: Dict
                Dictionary containing the sum of the preconditions for each merged combination
            resNumber: Dict
                Dictionary containing the number of instances of each merged combination
        """
        keys = [k for k in self.signCombinations.keys() if 
                self.signCombinationNumbers[k] >= np.mean(self.signCombinationNumbers.values())]
        sortedNumZeros = sorted(keys, key=lambda s: s.count('0'), reverse=True)
        resWeights = {}
        resSums = {}
        resNumber = {}
        while len(sortedNumZeros) > 0:
            curK = sortedNumZeros[0]
            tmpWeight = self.signCombinations[curK]
            tmpSum = np.copy(self.signCombinationSums[curK])
            tmpNumber = self.signCombinationNumbers[curK]
            toRem = []
            for k in sortedNumZeros[1:]:
                kInts = np.array([int(i) for i in curK.split(';')])
                k2Ints = np.array([int(i) for i in k.split(';')])
#                if not np.any(kInts*k2Ints < 0):
                if not np.any(kInts*k2Ints < 0) and \
                np.sum(kInts*k2Ints ==0)-np.sum(np.logical_and(kInts == 0,k2Ints==0)) < 2:
                    tmpSum += self.signCombinationSums[k]
                    tmpWeight += self.signCombinations[k]
                    tmpNumber += self.signCombinationNumbers[k]
                    toRem.append(k)
            sortedNumZeros = [k for k in sortedNumZeros[1:] if k not in toRem]
            curSigCom = []
            for i in xrange(len(tmpSum)):
                if tmpSum[i]/tmpWeight < -config.metaNodeThr:
                    curSigCom.append('-1')
                elif tmpSum[i]/tmpWeight > config.metaNodeThr:
                    curSigCom.append('1')
                else:
                    curSigCom.append('0')
            curSigCom = ";".join(curSigCom)
            resSums[curSigCom] = tmpSum
            resWeights[curSigCom] = tmpWeight
            resNumber[curSigCom] = tmpNumber
        
        return resWeights, resSums, resNumber

                        
                    
            
    def getPreconditions(self):
        """
            Returns up to two merged and averaged precondtions for this prototype
            
            Returns
            -------
            res1 : np.array
                Merged and averaged preconditions with the highest average contribution
            res2 : np.array or None
                If available merged and averaged preconditions with the second highest average
                contribution
        """
        res = np.zeros(self.lenPreCons)
        res2 = np.zeros(self.lenPreCons)
        mergedWeights, mergedSums, mergedNumbers = self.mergeCombinations()
        l = sorted([(k,mergedWeights[k]/mergedNumbers[k]) for k in mergedWeights.keys()], 
                    key=itemgetter(1), reverse=True)
        
        #TODO make better, relative weights often too similar, maybe harder threshold?
        # Ideally I would need a threshold that is not affected by the dominating combinations.
        if len(l) > 1 and l[1][1]/l[0][1] > 0.5:
            comb1 = l[0][0].split(";")
            comb2 = l[1][0].split(";")
            pre1 = mergedSums[l[0][0]]
            pre2 = mergedSums[l[1][0]]
            w1 = mergedWeights[l[0][0]]
            w2 = mergedWeights[l[1][0]]
            for i in xrange(len(comb1)):
                if comb1[i] == comb2[i]:# or comb1[i] == '0' or comb2[i] == '0':
                    res[i] = (pre1[i]+pre2[i])/(w1+w2)
                    res2[i] = res[i]
                else:
                    res[i] = pre1[i]/w1
                    res2[i] = pre2[i]/w2
            return res, res2
        else:
            return mergedSums[l[0][0]]/mergedWeights[l[0][0]], None
            
class MetaNetwork(object):
    """
        Network of the inverse model.
    """
    
    def __init__(self):
        """
            Setup of local variables
        """
        self.nodes = {}
        self.curIndex = None
        self.curSecIndex = None
        self.preConsSize = None
        self.difSize = None
        
        #For exploration
        self.targetIndex = None
        self.preConsToCheck = None
        self.preConsToTry = None
        self.preConIndex = 4  #Currently hard coded to only look at position
        self.tryNext = False

    
    def train(self, pre, difs):
        """
            Updates the network. Creates new nodes if required and trains the nodes
            defined by the difference vector.
            
            Parameters
            ---------
            pre : np.ndarray
                Current preconditions used as input for learning
            difs : np.ndarray
                Difference vector resulting from the preconditions
        """
        if self.preConsSize == None:
            self.preConsSize = len(pre)
        if self.difSize == None:
            self.difSize = len(difs)
#        targetIndexFound = False
        for i in xrange(len(difs)):
            #It appears smaller values break inverse model since the weights can 
            #get swapped for point symmetric preconditions
            if abs(difs[i]) > config.metaNetDifThr: 
                index = str(i*np.sign(difs[i]))
                if not index in self.nodes:
                    self.nodes[index] = MetaNode()
                self.nodes[index].train(pre,abs(difs[i]))

                if self.targetIndex != None and index == self.targetIndex:
                    self.targetIndex =None
                    self.preConIndex = 4  #For exploration
#                    targetIndexFound = True
                    
        ### For exploration            
#        if self.preConsToTry != None and np.linalg.norm(pre-self.preConsToTry) < 0.01:
#            if not targetIndexFound:
#                print "similar precons did not yield expected results."
#                print "targetIndex: ", self.targetIndex
#                print "actual difs: ", difs
#                self.tryNext = True

                
    def getPreconsToTry(self):
        """
            EXPERIMENTAL!!! (Requires uncomments in the train function)
            Function that tries to find preconditions that might increase its knowledge
            about the obejct interaction. 
            
            Returns
            ------
                np.ndarray
                Preconditions that might increase the models knowledge
        """
        if self.targetIndex == None:
            curKeys = self.nodes.keys()
            print "curKeys: ", curKeys
            for i in xrange(self.difSize):
                if str(1.0*i) in curKeys and not str(-1.0*i) in curKeys:
                    self.targetIndex = str(-1.0*i)
                    self.preConsToCheck = self.nodes[str(1.0*i)].getPreconditions()[0]
                    break
                if str(-1.0*i) in curKeys and not str(1.0*i) in curKeys:
                    self.targetIndex = str(1.0*i)
                    self.preConsToCheck = self.nodes[str(-1.0*i)].getPreconditions()[0]
                    break
                #TODO if no unkown key is left, look at "worst" key and improve that
                # figure out a way to measure which one is worst
        else:
            if self.tryNext:
                self.preConIndex += 1
            if self.preConIndex == 7:#len(self.preConsToCheck):
                self.targetIndex = None
                self.preConIndex = 4    
                return self.getPreconsToTry()
                
        if self.targetIndex == None:
            print "No key found to improve"
            return None
                
        print "targetIndex: ", self.targetIndex
        self.preConsToTry = np.copy(self.preConsToCheck)
        self.preConsToTry[self.preConIndex] *= -1
            

        return self.preConsToTry
        
                
    def getPreconditions(self, targetDifs):
        """
            Returns preconditions that should be suited to reduce the given target difs
            
            Parameters
            ----------
            targetDifs: np.ndarray
                Difference vector to the target
                
            Returns
            -------
                np.ndarray
                Preconditions that reduce the given target difs. Will be none, if no preconditions
                can be found due to missing prototypes
        """
        if GREEDY_TARGET:
            if self.curIndex != None:
                ind = float(self.curIndex)
                indSign = -1 if '-'in self.curIndex else 1
                #Consider making this a ratio of maximum/total difs so that it avoids jumping 
                #back and forth when it is already quite close to target
                #Also make sure not to ignore completly ruining other features when bad 
                #precondtiions are used
                if indSign == np.sign(targetDifs[abs(ind)]) and \
                abs(targetDifs[abs(ind)]) > config.metaNetIndexThr: 
                    preCons1, preCons2 = self.nodes[self.curIndex].getPreconditions()
                else:
                    self.curIndex = None
                    
            
            if self.curIndex == None:
                sortedDifs = np.argsort(abs(targetDifs))     
                i = 1
                while self.curIndex == None and i <= len(sortedDifs):
                    maxDif = sortedDifs[-i]
                    index = str(maxDif*np.sign(targetDifs[maxDif]))
                    if i+1 <= len(sortedDifs):
                        self.curSecIndex =str(sortedDifs[-i-1]*np.sign(targetDifs[sortedDifs[-i-1]]))
                    if not index in self.nodes:
                        #Try second highest index
                        i +=1
                        if i > len(sortedDifs):
                            return None
                    else:
                        self.curIndex = index
                        preCons1, preCons2 = self.nodes[index].getPreconditions()
                    
            #Determine which preconditions to use
            if preCons2 == None:
                return preCons1
            else:
                
                index2 = self.curSecIndex
                if not index2 in self.nodes:
                    return preCons1
                else:
                    #TODO norm difference not really suited here, because of features
                    #like velocity, which should not be considered
                    secCons1, secCons2 = self.nodes[index2].getPreconditions()
                    o1 = np.linalg.norm(secCons1-preCons1)
                    o2 = np.linalg.norm(secCons1-preCons2)
                    if secCons2 == None:
                        if o1 <= o2:
                            return preCons1
                        else:
                            return preCons2
                    else:
                        o3 = np.linalg.norm(secCons2-preCons1)
                        o4 = np.linalg.norm(secCons2-preCons2)
                        if min(o1,o3) <= min(o2,o4):
                            return preCons1
                        else:
                            return preCons2
        else:
            raise NotImplementedError("Currently only greedy is possible")