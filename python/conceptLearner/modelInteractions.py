#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:01:44 2015
Compact reimplementation of the interaction model very closely to the idea 
written in the thesis.
Already focuses on 2D only
By reimplementing from scatrch, we can hopefully avoid problems with the legacy components in 
the old model.
@author: jpoeppel
"""

import numpy as np
import common
from common import NUMDEC
from itm import ITM
from sets import Set
import copy

USE_DYNS = False

class Object(object):
    
    def __init__(self):
        self.id = 0
        self.vec = np.array([])
        self.lastVec = np.array([])
        pass
    
    @classmethod
    def parse(cls, m):
        res = cls()
        res.id = m.id 
        if USE_DYNS:
            res.vec = np.zeros(6)
            res.vec[0] = np.round(m.pose.position.x, NUMDEC) #posX
            res.vec[1] = np.round(m.pose.position.y, NUMDEC) #posY
            res.vec[2] = np.round(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), NUMDEC)[2] #ori
            res.vec[3] = np.round(m.linVel.x, NUMDEC) #linVelX
            res.vec[4] = np.round(m.linVel.y, NUMDEC) #linVelY
            res.vec[5] = np.round(m.angVel.z, NUMDEC) #angVel
        else:
            res.vec = np.zeros(3)
            res.vec[0] = np.round(m.pose.position.x, NUMDEC) #posX
            res.vec[1] = np.round(m.pose.position.y, NUMDEC) #posY
            res.vec[2] = np.round(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), NUMDEC)[2] #ori
                                                
                                                
        res.lastVec = np.copy(res.vec)
        return res
        
    @classmethod
    def fromInteractionState(cls, intState):
        o1 = cls()
        o1.vec = np.zeros(3)
        o2 = cls()
        o2.vec = np.zeros(3)
        o1.id = intState.vec[0]
        o2.id = intState.vec[1]
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
    
class InteractionState(object):
    
    def __init__(self):
        self.id = ""
        self.vec = np.array([])
        self.lastVec = np.array([])
        self.trans = None
        self.invTrans = None
        self.ori = 0.0
        pass
    
    def update(self, newState):
        self.lastVec = np.copy(self.vec)
        self.vec = np.copy(newState.vec)
        self.ori = newState.ori
        self.trans = np.copy(newState.trans)
        self.invTrans = np.copy(newState.invTrans)
    
    @classmethod
    def fromObjectStates(cls, o1, o2):
        res = cls()
        res.id = str(o1.id) + "," + str(o2.id)
        
        res.vec = np.zeros(7)
        res.vec[0] = o1.id
        res.vec[1] = o2.id
#        res.vec[2:5] = 0.0 #Does not need to be set since it will be zero in local coordinate system
        res.vec[5:7] = common.relPos(o1.vec[0:2], o1.vec[2], o2.vec[0:2])
        """ Remove for now since this will most likely break itm performance...
            Can however only be removed under the assumption that the gripper always has 
            orientation 0
        """
#        res.vec[7] = o2.vec[2]-o1.vec[2] 
        res.lastVec = np.copy(res.vec)
        res.trans = common.eulerPosToTransformation(o1.vec[2],o1.vec[0:2])
        res.invTrans = common.invertTransMatrix(res.trans)
        res.ori = o1.vec[2]
        return res
    
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
            
            
        self.actuator = self.objectStates[8]
    
class Action(object):
    
    def __init__(self):
        self.vec = np.zeros(3)
        pass

class Episode(object):
    
    def __init__(self, pre, action, post):
        self.preState = pre
        self.action = action
        self.postState = post
        self.difs = post.vec-pre.vec
        #TODO Convert post into the coordinate system of pre first!!
        
    def getChangingFeatures(self):
        return np.where(self.difs>0.001)[0]
        
    
class AbstractCollection(object):
    
    def __init__(self, identifier, changingFeatures):
        self.id = identifier
        self.predictor = ITM()
        self.changingFeatures = np.copy(changingFeatures)
        self.storedEpisodes = []
    
    def update(self, episode):
        #Translation can be ignored since we are dealing with velocity
        transAction = np.dot(episode.preState.invTrans[:-1,:-1], episode.action) 
        vec = np.concatenate((episode.preState.vec, transAction))
        self.predictor.update(vec, episode.difs[self.changingFeatures])
        self.storedEpisodes.append(episode)
    
    def predict(self, intState, action):
        """
            Action needs to be transformed to local coordinate system!
        """
        res = InteractionState()
        res.id = intState.id
        res.trans = np.copy(intState.trans)
        res.invTrans = np.copy(intState.invTrans)
        res.ori = intState.ori
        res.lastVec = np.copy(intState.vec)
        res.vec = np.copy(intState.vec)
        transAction = np.dot(intState.invTrans[:-1,:-1], action)
        vec = np.concatenate((intState.vec, transAction))
        prediction = self.predictor.test(vec)
        res.vec[self.changingFeatures] += prediction
        return res
    
class ACSelector(object):
    
    def __init__(self):
        self.classifier = ITM()
        self.isTrained = False
        
    def update(self, intState, action, respACId):
        vec = np.concatenate((intState.vec, action))
        self.classifier.update(vec, np.array([respACId]))
        self.isTrained = True
    
    def test(self, intState, action):
        if self.isTrained:
            vec = np.concatenate((intState.vec, action))
            return int(self.classifier.test(vec))
#           return int(self.classifier.test(vec, testMode=0)) #Only use winner to make prediction
        else:
            return None


class ModelInteraction(object):

    def __init__(self):
        self.acSelector = ACSelector()
        self.abstractCollections = {}
        self.curInteractionStates = {}
        self.featuresACMapping = {}
        self.training = True
        pass    
    
    def update(self, curWorldState, usedAction):
        for intId, intState in curWorldState.interactionStates.items():
            if intId in self.curInteractionStates:
                #When training, update the ACs and the selector
                if self.training:
                    newEpisode = Episode(self.curInteractionStates[intId], usedAction, intState)
                    changingFeatures = newEpisode.getChangingFeatures()
                    featuresString = ",".join(map(str,changingFeatures))
                    #Create AC if not already known
                    if not featuresString in self.featuresACMapping:
                        newAC = AbstractCollection(len(self.abstractCollections),changingFeatures)
                        self.abstractCollections[newAC.id] = newAC
                        self.featuresACMapping[featuresString] = newAC.id
                        
                    self.abstractCollections[self.featuresACMapping[featuresString]].update(newEpisode)
                    self.acSelector.update(self.curInteractionStates[intId], usedAction, self.featuresACMapping[featuresString])
                
                self.curInteractionStates[intId].update(intState)
            else:
                self.curInteractionStates[intId] = intState
        
    
    def predict(self, curWorldState, curAction):
        #TODO is curWorldState even needed here?
        newWS = WorldState()
        for intId, intState in self.curInteractionStates.items():
            acID = self.acSelector.test(intState, curAction)
            print "acID: ", acID
            if acID == None:
                newIntState = copy.deepcopy(intState)
            else:
                print "changing features: ", self.abstractCollections[acID].changingFeatures
                newIntState = self.abstractCollections[acID].predict(intState, curAction)
            newWS.interactionStates[intId] = newIntState
        
        newWS.finalize()
        return newWS
        
        
        
    def resetObjects(self, curWS):
        for i in curWS.interactionStates.values():
            if i.id in self.curInteractionStates:
                self.curInteractionStates[i.id].update(i)
            else:
                self.curInteractionStates[i.id] = i