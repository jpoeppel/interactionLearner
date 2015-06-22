#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:41:59 2015

@author: jpoeppel
"""

from sets import Set
import numpy as np
from topoMaps import ITM
import copy
from operator import itemgetter
from metrics import similarities
from config import SINGLE_INTSTATE
from common import GAZEBOCMDS as GZCMD

if SINGLE_INTSTATE:
    from state3 import State, ObjectState, InteractionState, WorldState
else:
    from state2 import State, ObjectState, InteractionState, WorldState

from state2 import Action as GripperAction

THRESHOLD = 1

class BaseCase(object):
    
    def __init__(self, pre, post):
        assert isinstance(pre, State), "{} is not a State object.".format(pre)
        assert isinstance(post, State), "{} is not a State object.".format(post)
        
        self.preState = pre
        self.postState = post
        self.dif = {}
        for k in pre.relKeys:
            self.dif[k] = post[k] - pre[k]
            
            
    def getSetOfAttribs(self):
        
        r = Set()
        for k,v in self.dif.items():
            if np.linalg.norm(v) > 0.01:
                r.add(k)
        return r
        
    def __eq__(self, other):
        if not isinstance(other, BaseCase):
            return False
        return self.preState == other.preState and self.postState == other.postState
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
class Action(object):
    """
         'Action' object that modifies features of objects so they can
         predict locally
    """
    
    def __init__(self):
        self.preConditions = {}
        self.targets = Set()
        self.effect = {}
        self.refCases = []
        self.unusedFeatures = []
        
    def applyAction(self, state, worldState, strength = 1.0):
        for k,v in self.effect.items():
            if isinstance(v, ITM):
                state[k] += strength * v.predict(worldState.toVec(self.unusedFeatures))
            else:
                state[k] += strength * v
            
    def rate(self, objectState, worldState):
        if objectState["name"] not in self.targets:
            return 0
            
        s = 0.0
        for k,v in self.preConditions:
             s += similarities[k](v,worldState[k])
            
        if len(self.preConditions) != 0:
            return s/len(self.preConditions)
        else:
            return s
            
    def update(self, case, worldState):
        self.targets.add(case.preState["name"])
        for k,v in worldState.items():
            if k in self.preConditions:
                if np.linalg.norm(v-self.preConditions[k]) > 0.1:
                    del self.preConditions[k]
            else:
                if len(self.refCases) < 1:
                    self.preConditions[k] = v 
        for k in case.getSetOfAttribs():
            if not k in self.effect:
                self.effect[k] = ITM()
            self.effect[k].train(Node(0, wIn=worldState(self.unusedFeatures), wOut=case.dif[k]))
                
        self.refCases.append((case, worldState))
        
        
    @classmethod
    def getGripperAction(cls, cmd=GZCMD["NOTHING"] , direction=np.zeros(3)):
        res = GripperAction(cmd, direction)
        return res
            
            
    @classmethod            
    def getRandomGripperAction(cls):
        res = cls()
        res.targets = Set(["gripper"])
        res.effect["linVel"][:2] = np.random.rand(2)
        return res
        
class Predictor(object):
    """
        Wrapper class for the objectState prediction. Is used to predict
        the state in the next timestep.
    """
    def __init__(self):
        self.pred = ITM()
        self.targets = Set()
        self.refCases = []
        self.unusedFeatures = []
        
    def predict(self, state):
        res = copy.deepcopy(state)
        pred = self.pred[k].predict(state.toVec())
        i = 0
        for k in state.relKeys:
            if hasattr(res[k], "__len__"):
                l = len(res[k])
            else:
                l = 1
            res[k] += pred[i:i+l]
            i += l
            
        return res
        
    def update(self, case):
        if case in self.refCases:
            raise AttributeError("Case already present")
        self.refCases.append(case)
        self.pred.train(Node(0, wIn=case.preState.toVec(self.unusedFeatures), 
                             wOut = case.postState.toVec()))   

        
class ModelAction(object):
    
    def __init__(self):
        self.predictors = []
        self.actions = []
        self.cases = []

    def getAction(self):
        pass
    
    def applyMostSuitedAction(self, objectState, worldState, action):
        if objectState["name"] == "gripper":
            objectState["linVel"] = action["mvDir"]
            
        scoreList = [(a.rate(objectState, worldState), a) for a in self.actions]
        sortedList = sorted(scoreList, key=itemgetter(0), reverse=True) 
        totalScore = np.sum([s[0] for s in sortedList])
        res = copy.deepcopy(objectState)
        for s,a in sortedList:
            a.applyAction(res, worldState, s/totalScore)
        return res
        
    def predictObjectState(self, objectState):
        for pred in self.predictors:
            if objectState["name"] in pred.targets:
                return pred.predict(objectState)
                
        #Return the state itself if no predictor was found
        return copy.deepcopy(objectState)
    
    def predict(self, worldState, action):
        resultWS = WorldState()
        resultWS.transM = np.copy(worldState.transM)
        resultWS.invTrans = np.copy(worldState.invTrans)
        resultWS.ori = np.copy(worldState.ori)
        for objectState in worldState.objectStates.values():
            newOS = self.applyMostSuitedAction(objectState, worldState, action)
            result = self.predictObjectState(newOS)
            resultWS.addObjectState(result)
        return resultWS
        
    def checkForAction(self, case, worldState):
        for a in self.actions:
            if case.preState["name"] in a.targets:
                if Set(a.effect.keys()) == case.getSetOfAttribs():
                    a.update(case, worldState)
                    return
        #If no action was found
        newAction = Action()
        newAction.update(case, worldState)
        
        
    def updateState(self, objectState, worldState, action, resultingOS):
        case = BaseCase(worldState.getObjectState(objectState["name"]), resultingOS)
        predictionRatring = resultingOS.score(objectState)
        if predictionRatring < THRESHOLD:
            self.checkForAction(case, worldState)
            for pred in self.predictors:
                if objectState["name"] in pred.targets:
                    pred.update(case)
                    self.cases.append(case)
    
    def update(self, worldState, action, prediction, result):
        for os in prediction.objectStates.values():
            self.updateState(os, worldState, action, result.getObjectState(os["name"]))
