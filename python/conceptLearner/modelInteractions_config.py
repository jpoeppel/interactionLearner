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
import copy
from configuration import config
from inverseModel import MetaNetwork


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
        res.vec[7] = o2.vec[2]-o1.vec[2] 
        res.lastVec = npcopy(res.vec)
        res.trans = common.eulerPosToTransformation(o1.vec[2],o1.vec[0:2])
        res.invTrans = common.invertTransMatrix(res.trans)
        res.ori = o1.vec[2]
        return res

        
    def circle(self, toTarget = None):
        o1,o2 = Object.fromInteractionState(self)
        dist = common.generalDist(o1.id, o1.vec[0:2], o1.vec[2], o2.id, o2.vec[:2], o2.vec[2])
        relPos = self.vec[5:7]
        globRelPos = np.dot(self.trans[:-1,:-1], relPos)
        if dist < 0.04:
            return 0.4*globRelPos/np.linalg.norm(globRelPos)
        elif dist > 0.06:
            return -0.4*globRelPos/np.linalg.norm(globRelPos)
        tangent = np.array([-globRelPos[1], globRelPos[0]])
        if toTarget != None:
            angAct = np.arctan2(relPos[1],relPos[0])
            angTarget = np.arctan2(toTarget[1],toTarget[0])
            angDif = angTarget+np.pi - (angAct+np.pi)
            print "angDif: ", angDif
            if angDif > 0:
                if abs(angDif) < np.pi:
                    print "circling pos"
                    return 0.4*tangent/np.linalg.norm(tangent)
                else:
                    print "circling neg"
                    return -0.4*tangent/np.linalg.norm(tangent)
            else:
                if abs(angDif) < np.pi:
                    print "circling neg"
                    return -0.4*tangent/np.linalg.norm(tangent)
                else:
                    print "circling pos"
                    return 0.4*tangent/np.linalg.norm(tangent)
            
            return -0.4*tangent/np.linalg.norm(tangent)
        else:
            return 0.4*tangent/np.linalg.norm(tangent)
    
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
        if np.linalg.norm(self.difs) < 0.001:
            print "pre: {}".format(pre)
            print "post: {}".format(post)
            print "action: {}".format(action)
            raise NotImplementedError
        
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
        self.inverseModel = MetaNetwork()
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
                targetDifs = np.zeros(len(desiredEpisode.difs))#[self.targetFeatures]
                targetDifs[self.targetFeatures] = desiredEpisode.difs[self.targetFeatures]
#                maxIndex = np.argmax(abs(targetDifs))
#                maxFeature = targetChangingFeatures[maxIndex]
#                print "maxIndex: ", maxIndex
#                print "maxFeature: ", maxFeature
                featuresString = ",".join(map(str,targetChangingFeatures))
                print "featuresString: ", featuresString
                print "available ACs: ", self.featuresACMapping.keys()
#                responsibleACs = []

#                for k in self.featuresACMapping.keys():
#                    if featuresString in k:
#                        responsibleACs.append(self.abstractCollections[self.featuresACMapping[k]])
#                    else:
#                        if str(maxFeature) in k:
#                            responsibleACs.append(self.abstractCollections[self.featuresACMapping[k]])
#            
##                for ac in responsibleACs:
##                    if str(maxIndex*np.sign(targetDifs[maxIndex])) in ac.inverseModel.nodes:
#                        
#                preConditions = None
##                i=len(responsibleACs)-1
#                i=0
#                while preConditions == None:
#                    preConditions = responsibleACs[i].getAction(targetDifs)
#                    print "AC features: ", responsibleACs[i].changingFeatures
#                    i+=1
                
                preConditions = self.inverseModel.getPreconditions(targetDifs)
                print "usedAC: ", i-1
                print "Preconditions: ", preConditions
                relTargetPos = preConditions[5:7]
                globTargetPos = np.ones(3)
                globTargetPos[:2] = np.copy(relTargetPos)
                globTargetPos = np.dot(targetInteraction.trans, globTargetPos)[:2]
                relAction = preConditions[8:10]
                print "relAction: ", relAction
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
#                wrongSides = curRelPos*relTargetPos < 0
#                if np.any(wrongSides):
#                    if max(abs(relTargetPos[wrongSides]-curRelPos[wrongSides])) > 0.05:
#                        print "circling"
#                        return targetInteraction.circle()
    
                if np.linalg.norm(difPos) > 0.1:
                    print "circling, too far"
                    return targetInteraction.circle(relTargetPos)
                if np.linalg.norm(difPos) > 0.01:
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
                    transAction = np.dot(self.curInteractionStates[intId].invTrans[:-1,:-1], usedAction) 
                    vec = np.concatenate((self.curInteractionStates[intId].vec, transAction))
                    self.inverseModel.train(vec, newEpisode.difs)
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