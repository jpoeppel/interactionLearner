# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:31:29 2015
Inverse model
@author: jpoeppel
"""

import numpy as np
from configuration import config
from operator import itemgetter

GREEDY_TARGET = True

class MetaNode(object):

    def __init__(self):
        self.signCombinations= {}
        self.signCombinationSums= {}
        self.signCombinationNumbers = {}
        self.lenPreCons = 0
        pass

    def train(self, pre, dif):
        """
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
#        curSigComInts = [int(i) for i in curSigCom]
        curSigCom = ";".join(curSigCom)
        #Problem since it removes the 0 features in combinations
        # This results that some combination features are not averaged even if they should be
#        keys = self.signCombinations.keys()
#        key = None
#        for k in keys:
#            kInts = [int(i) for i in k.split(';')]
#            correct = True
#            for i in xrange(len(kInts)):
#                if kInts[i]*curSigComInts[i] < 0:
#                    #wrong key
#                    correct = False
#                    break
#            if correct:
#                key = k
#        if key != None:
#            self.signCombinations[key] += dif
#            self.signCombinationSums[key] += dif*pre
#            self.signCombinationNumbers[key] += 1
#        else:
#            self.signCombinations[curSigCom] = dif
#            self.signCombinationSums[curSigCom] = dif*pre
#            self.signCombinationNumbers[curSigCom] = 1
            
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
#        kInts = [[int(i) for i in k.split(';')] for k in self.signCombinations.keys()]
        keys = [k for k in self.signCombinations.keys() if self.signCombinationNumbers[k] >= np.mean(self.signCombinationNumbers.values())]
        sortedNumZeros = sorted(keys, key=lambda s: s.count('0'), reverse=True)
        resWeights = {}
        resSums = {}
        resNumber = {}
        while len(sortedNumZeros) > 0:
            curK = sortedNumZeros[0]
            tmpWeight = self.signCombinations[curK]
            tmpSum = np.copy(self.signCombinationSums[curK])
            tmpNumber = self.signCombinationNumbers[curK]
#            resWeights[curK] = self.signCombinations[curK]
#            resSums[curK] = np.copy(self.signCombinationSums[curK])
#            resNumber[curK] = self.signCombinationNumbers[curK]
            toRem = []
            for k in sortedNumZeros[1:]:
                kInts = np.array([int(i) for i in curK.split(';')])
                k2Ints = np.array([int(i) for i in k.split(';')])
                if not np.any(kInts*k2Ints < 0):
                    tmpSum += self.signCombinationSums[k]
                    tmpWeight += self.signCombinations[k]
                    tmpNumber += self.signCombinationNumbers[k]
#                    resSums[curK] += self.signCombinationSums[k]
#                    resWeights[curK] += self.signCombinations[k]
#                    resNumber[curK] += self.signCombinationNumbers[k]
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
        res = np.zeros(self.lenPreCons)
        res2 = np.zeros(self.lenPreCons)
#        l = sorted([(k, v) for k,v in self.signCombinations.items()], key=itemgetter(1), reverse=True)
        mergedWeights, mergedSums, mergedNumbers = self.mergeCombinations()
        l = sorted([(k,mergedWeights[k]/mergedNumbers[k]) for k in mergedWeights.keys()], key=itemgetter(1), reverse=True)
#        l = sorted([(k,self.signCombinations[k]/self.signCombinationNumbers[k], self.signCombinationNumbers[k]) for k in self.signCombinations.keys()], key=itemgetter(1), reverse=True)
#        print "l: ", l
#        print "weight dif: ", self.signCombinations[l[1][0]]/self.signCombinations[l[0][0]]
        if len(l) > 1 and l[1][1]/l[0][1] > 0.5:
#        if len(l) > 1 and self.signCombinations[l[1][0]]/self.signCombinations[l[0][0]] >0.5 :
            comb1 = l[0][0].split(";")
            comb2 = l[1][0].split(";")
            pre1 = mergedSums[l[0][0]]
            pre2 = mergedSums[l[1][0]]
            w1 = mergedWeights[l[0][0]]
            w2 = mergedWeights[l[1][0]]
#            pre1 = self.signCombinationSums[l[0][0]]
#            pre2 = self.signCombinationSums[l[1][0]]
#            w1 = self.signCombinations[l[0][0]]#/self.signCombinationNumbers[l[0][0]]
#            w2 = self.signCombinations[l[1][0]]#/self.signCombinationNumbers[l[1][0]]
#            print "pre1: ", pre1
#            print "pre2: ", pre2
#            print "comb1: ", comb1
#            print "comb2: ", comb2
#            print "comb1 w: ", w1
#            print "comb2 w: ", w2
            for i in xrange(len(comb1)):
                if comb1[i] == comb2[i]:# or comb1[i] == '0' or comb2[i] == '0':
                    res[i] = (pre1[i]+pre2[i])/(w1+w2)
                    res2[i] = res[i]
                else:
                    res[i] = pre1[i]/w1
                    res2[i] = pre2[i]/w2
            return res, res2
        else:
            print "only one combination"
            return mergedSums[l[0][0]]/mergedWeights[l[0][0]], None
#            return self.signCombinationSums[l[0][0]]/self.signCombinations[l[0][0]], None
            
class MetaNetwork(object):
    
    def __init__(self):
        self.nodes = {}
        self.curIndex = None
        self.curSecIndex = None
        self.preConsSize = None
        self.difSize = None
        self.targetIndex = None
        self.preConsToCheck = None
        self.preConsToTry = None
        self.preConIndex = 4  #Currently hard coded to only look at position
        self.tryNext = False
        pass
    
    def train(self, pre, difs):
        if self.preConsSize == None:
            self.preConsSize = len(pre)
        if self.difSize == None:
            self.difSize = len(difs)
        targetIndexFound = False
#        print "difs: ", difs
#        print "training network with pre: ", pre
        for i in xrange(len(difs)):
            #It appears smaller values break inverse model since the weights can 
            #get swapped for point symmetric preconditions
            if abs(difs[i]) > config.metaNetDifThr: 
                index = str(i*np.sign(difs[i]))
                if not index in self.nodes:
                    self.nodes[index] = MetaNode()
#                print "training index: {} with dif: {}".format(index, difs[i])
#                print "precons: ",pre[[4,5,6,10,11]]
                self.nodes[index].train(pre,abs(difs[i]))

                if self.targetIndex != None and index == self.targetIndex:
                    print "target: {} successfully found.".format(index)
                    self.targetIndex =None
                    self.preConIndex = 4  #For exploration
                    targetIndexFound = True
                    
        ### For exploration            
        if self.preConsToTry != None:
            print "precons similarity: ", np.linalg.norm(pre-self.preConsToTry)
            print "given pres: ", pre
            print "desired pres: ", self.preConsToTry
        if self.preConsToTry != None and np.linalg.norm(pre-self.preConsToTry) < 0.01:
            print "similar precons reached: ", np.linalg.norm(pre-self.preConsToTry)
            if not targetIndexFound:
                print "similar precons did not yield expected results."
                print "targetIndex: ", self.targetIndex
                print "actual difs: ", difs
                self.tryNext = True

                
    def tobeNamed(self):
        """
            Function that tries to find preconditions that might increase its knowledge
            about the obejct interaction.
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
                return self.tobeNamed()
                
        if self.targetIndex == None:
            print "No key found to improve"
            return None
                
        print "targetIndex: ", self.targetIndex
        self.preConsToTry = np.copy(self.preConsToCheck)
        self.preConsToTry[self.preConIndex] *= -1
            

        return self.preConsToTry
        
                
    def getPreconditions(self, targetDifs):
        #TODO reset curSecIndex as well to avoid secondary oscilation!!
        if GREEDY_TARGET:
            if self.curIndex != None:
                ind = float(self.curIndex)
                indSign = -1 if '-'in self.curIndex else 1
                #Consider making this a ratio of maximum/total difs so that it avoids jumping back and forth when it is already quite close to target
                if indSign == np.sign(targetDifs[abs(ind)]) and abs(targetDifs[abs(ind)]) > config.metaNetIndexThr: 
                    print "working on curIndex: ", self.curIndex
                    preCons1, preCons2 = self.nodes[self.curIndex].getPreconditions()
                else:
                    self.curIndex = None
                    
            
            if self.curIndex == None:
                print "target difs: ", targetDifs
                sortedDifs = np.argsort(abs(targetDifs))     
                print "sortedDifs: ", sortedDifs
                maxDif = sortedDifs[-1]
                index = str(maxDif*np.sign(targetDifs[maxDif]))
                self.curSecIndex =str(sortedDifs[-2]*np.sign(targetDifs[sortedDifs[-2]]))
#                print "targetDifs: ", targetDifs
#                print "maxindex: ", index
                if not index in self.nodes:
                    print "index i {} for targetDif {}, not known".format(index, targetDifs[abs(float(index))])
                    print "nodes: ", self.nodes.keys()
                    print "targetDifs: ", targetDifs
                    return None
                else:
                    self.curIndex = index
                    print "precons for index: ", index
                    preCons1, preCons2 = self.nodes[index].getPreconditions()
                    
            if preCons2 == None:
                print "no alternative"
                return preCons1
            else:
                
                index2 = self.curSecIndex
                print "index2: ", index2
                if not index2 in self.nodes:
                    print "using pre1, index2 not found"
                    return preCons1
                else:
                    print "precons for index: ", index2
                    secCons1, secCons2 = self.nodes[index2].getPreconditions()
                    o1 = np.linalg.norm(secCons1-preCons1)
                    o2 = np.linalg.norm(secCons1-preCons2)
                    print "dist1: ", o1
                    print "dist2: ", o2
                    print "preCons1: ", preCons1
                    print "preCons2: ", preCons2
                    print "secCons1: ", secCons1
                    if secCons2 == None:
                        if o1 <= o2:
                            print "using pre1"
                            return preCons1
                        else:
                            print "using pre2"
                            return preCons2
                    else:
                        o3 = np.linalg.norm(secCons2-preCons1)
                        o4 = np.linalg.norm(secCons2-preCons2)
                        if min(o1,o3) <= min(o2,o4):
                            print "using pre1 sec"
                            return preCons1
                        else:
                            print "using pre2 sec"
                            return preCons2
                

        else:
            raise NotImplementedError("Currently only greedy is possible")