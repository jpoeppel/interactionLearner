#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:01:44 2015
Compact reimplementation of the interaction model very closely to the idea 
written in the thesis.
Already focuses on 2D only
By reimplementing from scatrch, we can hopefully avoid problems with the legacy components in 
the old model.

Replace all magic numbers with config lookups
@author: jpoeppel
"""

import numpy as np
from numpy import copy as npcopy
from numpy import round as npround
import common
from itm import ITM
from sets import Set
import copy
from configuration import config
from operator import itemgetter

GREEDY_TARGET = True

class Object(object):
    
    def __init__(self):
        self.id = 0
        if config.USE_DYNS:
            self.vec = np.zeros(6)
            self.lastVec = np.zeros(6)
        else:
            self.vec = np.zeros(3)
            self.lastVec = np.zeros(3)
    
    @classmethod
    def parse(cls, m):
        res = cls()
        res.id = m.id 
        if config.USE_DYNS:
            res.vec = np.zeros(6)
            res.vec[0] = npround(m.pose.position.x, config.NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, config.NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), config.NUMDEC)[2] #ori
            res.vec[3] = npround(m.linVel.x, config.NUMDEC) #linVelX
            res.vec[4] = npround(m.linVel.y, config.NUMDEC) #linVelY
            res.vec[5] = npround(m.angVel.z, config.NUMDEC) #angVel
        else:
            res.vec = np.zeros(3)
            res.vec[0] = npround(m.pose.position.x, config.NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, config.NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), config.NUMDEC)[2] #ori
        res.lastVec = npcopy(res.vec)
        return res
        
    @classmethod
    def fromInteractionState(cls, intState):
        #TODO cannot be used with dynamics as is
        o1 = cls()
        o2 = cls()
        o1.id = int(intState.vec[0])
        o2.id = int(intState.vec[1])
        p1 = np.ones(3)
        p1[:2] = intState.vec[2:4]
        p1 = np.dot(intState.trans, p1)
        o1.vec[:2] = p1[:2]
        o1.vec[2] = intState.vec[4] + intState.ori
        p2 = np.ones(3)
        p2[:2] = intState.vec[5:7]
        p2 = np.dot(intState.trans, p2)
        o2.vec[:2] = p2[:2]
        o2.vec[2] = 0.0
        return o1, o2
        
    def getKeyPoints(self):
        WIDTH = {15: 0.25, 8: 0.025} #Width from the middle point
        DEPTH = {15: 0.05, 8: 0.025} #Height from the middle point
        p1x = WIDTH[self.id]
        p2x = -p1x
        p1y = DEPTH[self.id]
        p2y = -p1y
        ang = self.vec[2]
        c = np.cos(ang)
        s = np.sin(ang)
        p1xn = p1x*c -p1y*s + self.vec[0]
        p1yn = p1x*s + p1y*c + self.vec[1]
        p2xn = p2x*c - p2y*s + self.vec[0]
        p2yn = p2x*s + p2y*c + self.vec[1]
        return np.array([npcopy(self.vec[0:2]), np.array([p1xn,p1yn]), np.array([p2xn,p2yn])])
    
class InteractionState(object):
    
    def __init__(self):
        self.id = ""
        self.vec = np.zeros(8)
        self.lastVec = np.zeros(8)
        self.trans = None
        self.invTrans = None
        self.ori = 0.0
        pass
    
    def update(self, newState):
        self.lastVec = npcopy(self.vec)
        self.vec = npcopy(newState.vec)
        self.ori = newState.ori
        self.trans = npcopy(newState.trans)
        self.invTrans = npcopy(newState.invTrans)
    
    @classmethod
    def fromObjectStates(cls, o1, o2):
        #TODO Does currently not use dynamics
        res = cls()
        res.id = str(o1.id) + "," + str(o2.id)
        
        res.vec[0] = o1.id
        res.vec[1] = o2.id
#        res.vec[2:5] = 0.0 #Does not need to be set since it will be zero in local coordinate system
        res.vec[5:7] = common.relPos(o1.vec[0:2], o1.vec[2], o2.vec[0:2])
#        res.vec[7] = o1.vec[2]
        """ Remove for now since this will most likely break itm performance...
            Can however only be removed under the assumption that the gripper always has 
            orientation 0
        """
        res.vec[7] = o2.vec[2]-o1.vec[2] 
        res.lastVec = npcopy(res.vec)
        res.trans = common.eulerPosToTransformation(o1.vec[2],o1.vec[0:2])
        res.invTrans = common.invertTransMatrix(res.trans)
        res.ori = o1.vec[2]
        return res

        
    def circle(self):
        o1,o2 = Object.fromInteractionState(self)
        dist = common.generalDist(o1.id, o1.vec[0:2], o1.vec[2], o2.id, o2.vec[:2], o2.vec[2])
        relPos = self.vec[5:7]
        if dist < 0.04:
            return 0.4*relPos/np.linalg.norm(relPos)
        elif dist > 0.06:
            return -0.4*relPos/np.linalg.norm(relPos)
        tangent = np.array([-relPos[1], relPos[0]])
        return -0.4*tangent/np.linalg.norm(tangent)
    
class WorldState(object):
    
    def __init__(self):
        self.objectStates = {}
        self.interactionStates = {}
        
        self.actuator = None
        
    def parseModels(self, models):
        for m in models:
            if m.name == "ground_plane" or "wall" in m.name or "Shadow" in m.name:
                continue
            else:
                tmp = Object.parse(m)               
                self.objectStates[tmp.id] = tmp
                if tmp.id == 8:
                    self.actuator = tmp
                
    def parseInteractions(self):
        for n1, os1 in self.objectStates.items():
            if n1 != 8:
                for n2, os2 in self.objectStates.items():
                    if n1 != n2:
                        intState = InteractionState.fromObjectStates(os1,os2)
                        self.interactionStates[intState.id] = intState
        
    def parse(self, gzWS):
        self.parseModels(gzWS.model_v.models)   
        self.parseInteractions()
        
    def finalize(self):
        """
            Recomputes the object states from the interaction states.
            In case more than one interactionState is given, only the
            last object state is kept. Will break with multiple objects!!!
        """
        for intState in self.interactionStates.values():
            o1, o2 = Object.fromInteractionState(intState)
            self.objectStates[o1.id] = o1
            self.objectStates[o2.id] = o2
            
        self.parseInteractions()
    
class Action(object):
    
    def __init__(self):
        self.vec = np.zeros(3)
        pass

class Episode(object):
    
    def __init__(self, pre, action, post):
        self.preState = pre
        self.action = action
        self.postState = post
        #Transform post to coordinate system of pre to compute proper difs
        postVec = npcopy(post.vec)
        oldPos = np.ones(3)
        oldPos[:2] = postVec[2:4]
        postVec[2:4] = np.dot(np.dot(pre.invTrans,post.trans), oldPos)[:2]
        postVec[4] += post.ori-pre.ori
        oldAltPos = np.ones(3)
        oldAltPos[:2] = postVec[5:7]
        postVec[5:7] = np.dot(np.dot(pre.invTrans,post.trans), oldAltPos)[:2]
        
        self.difs = postVec-pre.vec
        
    def getChangingFeatures(self):
        return np.where(abs(self.difs)>config.episodeDifThr)[0]

        
    
class AbstractCollection(object):
    
    def __init__(self, identifier, changingFeatures):
        self.id = identifier
        self.predictor = ITM()
        self.inverseModel = MetaNetwork()
        self.changingFeatures = npcopy(changingFeatures)
        self.storedEpisodes = []
    
    def update(self, episode):
        #Translation can be ignored since we are dealing with velocity
        transAction = np.dot(episode.preState.invTrans[:-1,:-1], episode.action) 
        vec = np.concatenate((episode.preState.vec, transAction))
        self.predictor.update(vec, episode.difs[self.changingFeatures], 
                              etaIn=config.aCEtaIn, etaOut=config.aCEtaOut, 
                              etaA=config.aCEtaA, testMode=config.aCTestMode)
        self.storedEpisodes.append(episode)
        self.inverseModel.train(vec, episode.difs[self.changingFeatures])
    
    def predict(self, intState, action):
        """
            Action needs to be transformed to local coordinate system!
        """
        res = InteractionState()
        res.id = intState.id
        res.trans = npcopy(intState.trans)
        res.invTrans = npcopy(intState.invTrans)
        res.ori = intState.ori
        res.lastVec = npcopy(intState.vec)
        res.vec = npcopy(intState.vec)
        transAction = np.dot(intState.invTrans[:-1,:-1], action)
        vec = np.concatenate((intState.vec, transAction))
        prediction = self.predictor.test(vec, testMode=config.aCTestMode)
        res.vec[self.changingFeatures] += prediction
        return res
        
    def getAction(self, difs):
        return self.inverseModel.getPreconditions(difs)

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
        curSigCom = ";".join(curSigCom)
        if curSigCom in self.signCombinations:
            self.signCombinations[curSigCom] += dif
            self.signCombinationSums[curSigCom] += dif*pre
            self.signCombinationNumbers[curSigCom] += 1
        else:
            self.signCombinations[curSigCom] = dif
            self.signCombinationSums[curSigCom] = dif*pre
            self.signCombinationNumbers[curSigCom] = 1
            
    def getPreconditions(self):
        res = np.zeros(self.lenPreCons)
        res2 = np.zeros(self.lenPreCons)
        l = sorted([(k,self.signCombinations[k]/self.signCombinationNumbers[k]) for k in self.signCombinations.keys()], key=itemgetter(1), reverse=True)
        if len(l) > 1:
            comb1 = l[0][0].split(";")
            comb2 = l[1][0].split(";")
            pre1 = self.signCombinationSums[l[0][0]]
            pre2 = self.signCombinationSums[l[1][0]]
            w1 = self.signCombinations[l[0][0]]
            w2 = self.signCombinations[l[1][0]]
            for i in xrange(len(comb1)):
                if comb1[i] == comb2[i] or comb1[i] == '0' or comb2[i] == '0':
                    res[i] = (pre1[i]+pre2[i])/(w1+w2)
                    res2[i] = res[i]
                else:
                    res[i] = pre1[i]/w1
                    res2[i] = pre2[i]/w2
            return res, res2
        else:
            return self.signCombinationSums[l[0][0]]/self.signCombinations[l[0][0]], None
            
class MetaNetwork(object):
    
    def __init__(self):
        self.nodes = {}
        self.curIndex = None
        self.curSecIndex = None
        self.preConsSize = None
        self.difSize = None
        self.targetIndex = None
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
                            
                
    def getPreconditions(self, targetDifs):
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
#                    self.curIndex = index
                    print "precons for index: ", index
                    preCons1, preCons2 = self.nodes[index].getPreconditions()
                    
            if preCons2 == None:
                print "no alternative"
                return preCons1
            else:
                index2 = self.curSecIndex
                if not index2 in self.nodes:
                    print "using pre1"
                    return preCons1
                else:
                    print "precons for index: ", index2
                    secCons1, secCons2 = self.nodes[index2].getPreconditions()
                    o1 = np.linalg.norm(secCons1-preCons1)
                    o2 = np.linalg.norm(secCons1-preCons2)
#                    print "dist1: ", o1
#                    print "dist2: ", o2
#                    print "preCons1: ", preCons1
#                    print "preCons2: ", preCons2
#                    print "secCons1: ", secCons1
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
                
    
class ACSelector(object):
    
    def __init__(self):
        self.classifier = ITM()
        self.isTrained = False
        
    def update(self, intState, action, respACId):
        transAction = np.dot(intState.invTrans[:-1,:-1], action) 
        vec = np.concatenate((intState.vec[config.aCSelectorMask], transAction))
        self.classifier.update(vec, np.array([respACId]), 
                               etaIn=config.aCSelectorEtaIn, 
                               etaOut=config.aCSelectorEtaOut, 
                               etaA=config.aCSelectorEtaA, 
                               testMode=config.aCSelectorTestMode)        
        self.isTrained = True
    
    def test(self, intState, action):
        transAction = np.dot(intState.invTrans[:-1,:-1], action) 
        if self.isTrained:
            vec = np.concatenate((intState.vec[config.aCSelectorMask], transAction))
#            return int(self.classifier.test(vec))
            #Only use winner to make prediction, interpolation does not really
            #make sense when having more then 2 classes
            return int(self.classifier.test(vec, testMode=config.aCSelectorTestMode)) 
        else:
            return None


class ModelInteraction(object):

    def __init__(self):
        self.acSelector = ACSelector()
        self.abstractCollections = {}
        self.curInteractionStates = {}
        self.featuresACMapping = {}
        self.training = True
        self.target = None
        self.targetFeatures = []
        pass    
    
    
    def getITMInformation(self):
        acSelString = "acSelector ITM: UpdateCalls: {}, Insertions: {}, final Number of nodes: {}\n"\
                                    .format(self.acSelector.classifier.updateCalls,
                                            self.acSelector.classifier.inserts, 
                                            len(self.acSelector.classifier.nodes))
        acString = ""
        for ac in self.abstractCollections.values():
            acString += "Abstract collection for features: {}, ITM: UpdateCalls: {}, Insertions: {}, final Number of nodes: {}\n"\
                        .format(ac.changingFeatures, ac.predictor.updateCalls, 
                                ac.predictor.inserts, len(ac.predictor.nodes))
        return acSelString + acString
        
    def setTarget(self, target):
        """
            Sets a target that is to be reached.
            Target is an object (maybe partially described)
            Parameters
            ----------
            target : Object
        """
        #Create target interaction state from object
        if target.id == 8:
            raise NotImplementedError("Currently only non acutator objects can be set as targets.")
        dummyObject = Object()
        dummyObject.id = 8
        self.target = InteractionState.fromObjectStates(target, dummyObject)
        self.targetFeatures = [2,3,4]
        
    def isTargetReached(self):
        targetInteraction = self.curInteractionStates[self.target.id]
        episode = Episode(targetInteraction, Action(), self.target)
        if np.linalg.norm(episode.difs[self.targetFeatures]) < 0.01:
            return True
        return False
        
    def getAction(self):
        if self.target is None:
#            return self.explore()
            return np.array([0.0,0.0])
        else:
            if self.isTargetReached():
                self.target = None
                print "Target reached!"
                return np.zeros(2)
            else:
                targetInteraction = self.curInteractionStates[self.target.id]
                #Create desired episode with zero action, in order to get required difference vector
                desiredEpisode = Episode(targetInteraction, Action(), self.target)
                #Only consider target differences
                
                targetChangingFeatures = [i for i in desiredEpisode.getChangingFeatures() if i in self.targetFeatures]
                targetDifs = desiredEpisode.difs[self.targetFeatures]
                maxIndex = np.argmax(abs(targetDifs))
                maxFeature = targetChangingFeatures[maxIndex]
                print "maxIndex: ", maxIndex
                featuresString = ",".join(map(str,targetChangingFeatures))
                print "targetChangingFeatures: ", targetChangingFeatures
                print "available ACs: ", self.featuresACMapping.keys()
                responsibleACs = []
    #            if featuresString in self.featuresACMapping:
    #                responsibleACs.append(self.abstractCollections[self.featuresACMapping[featuresString]])
    #            else:
                for k in self.featuresACMapping.keys():
                    if str(maxFeature) in k:
                        responsibleACs.append(self.abstractCollections[self.featuresACMapping[k]])
                
#                for ac in responsibleACs:
#                    if str(maxIndex*np.sign(targetDifs[maxIndex])) in ac.inverseModel.nodes:
                        
                preConditions = None
                i=0
                while preConditions == None:
                    preConditions = responsibleACs[i].getAction(targetDifs)
                    i+=1
                print "usedAC: ", i-1
                print "Preconditions: ", preConditions
                relTargetPos = preConditions[5:7]
                globTargetPos = np.ones(3)
                globTargetPos[:2] = np.copy(relTargetPos)
                globTargetPos = np.dot(targetInteraction.trans, globTargetPos)[:2]
                relAction = preConditions[8:10]
                curRelPos = targetInteraction.vec[5:7]
                globCurPos = np.ones(3)
                globCurPos[:2] = np.copy(curRelPos)
                globCurPos = np.dot(targetInteraction.trans, globCurPos)[:2]
                print "global Target: ", globTargetPos
                print "global Current: ", globCurPos
                print "rel Target: ", relTargetPos
                print "rel Pos: ", curRelPos
                
                difPos = globTargetPos-globCurPos
                globAction = np.dot(targetInteraction.trans[:2,:2],relAction)
                print "global Action: ", globAction
                wrongSides = curRelPos*relTargetPos < 0
                if np.any(wrongSides):
                    if max(abs(relTargetPos[wrongSides]-curRelPos[wrongSides])) > 0.05:
                        print "circling"
                        return targetInteraction.circle()
    
    
                if np.linalg.norm(difPos) > 0.05:
                    print "doing difpos"
                    return 0.3*difPos/np.linalg.norm(difPos)                
                print "global action"
                if np.linalg.norm(globAction) == 0.0:
                    return globAction
                return  0.3*globAction/np.linalg.norm(globAction)
                pass
    
    def update(self, curWorldState, usedAction):
        for intId, intState in curWorldState.interactionStates.items():
            if intId in self.curInteractionStates:
                #When training, update the ACs and the selector
                if self.training:
                    newEpisode = Episode(self.curInteractionStates[intId], usedAction, intState)
#                    print "newEpisode difs: ", newEpisode.difs
                    changingFeatures = newEpisode.getChangingFeatures()
                    featuresString = ",".join(map(str,changingFeatures))
#                    print "feature String: ", featuresString
                    #Create AC if not already known
                    if not featuresString in self.featuresACMapping:
                        newAC = AbstractCollection(len(self.abstractCollections),changingFeatures)
                        self.abstractCollections[newAC.id] = newAC
                        self.featuresACMapping[featuresString] = newAC.id
                        
                    self.abstractCollections[self.featuresACMapping[featuresString]].update(newEpisode)
                    self.acSelector.update(self.curInteractionStates[intId], usedAction, self.featuresACMapping[featuresString])
                    
#                    if not 0 in self.abstractCollections:
#                        self.abstractCollections[0] = AbstractCollection(0, np.array([2,3,4,5,6,7]))
#                    self.abstractCollections[0].update(newEpisode)
                    
                
                self.curInteractionStates[intId].update(intState)
            else:
                self.curInteractionStates[intId] = intState
        
    
    def predict(self, curWorldState, curAction):
        #TODO is curWorldState even needed here?
        newWS = WorldState()
        for intId, intState in curWorldState.interactionStates.items():
#        for intId, intState in self.curInteractionStates.items():
            acID = self.acSelector.test(intState, curAction)
#            if 0 in self.abstractCollections:
#                acID = 0
#            else:
#                acID = None
#            print "acID: ", acID
            if acID == None:
                newIntState = copy.deepcopy(intState)
            else:
#                print "changing features: ", self.abstractCollections[acID].changingFeatures
                newIntState = self.abstractCollections[acID].predict(intState, curAction)
            newWS.interactionStates[intId] = newIntState
        
        newWS.finalize()
        return newWS
        
        
        
    def resetObjects(self, curWS):
        self.curInteractionStates = {}
        for i in curWS.interactionStates.values():
            if i.id in self.curInteractionStates:
                self.curInteractionStates[i.id].update(i)
            else:
                self.curInteractionStates[i.id] = i