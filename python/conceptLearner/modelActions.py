#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:41:59 2015
TODO: When using more then 2 objects, the actions need to differentiate which interactionState
they primarily use!!
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
from network import Node


#if SINGLE_INTSTATE:
#    from state3 import State, ObjectState, InteractionState, WorldState
#else:
#    from state2 import State, ObjectState, InteractionState, WorldState

from state4 import State, ObjectState, InteractionState, WorldState

from state2 import Action as GripperAction

THRESHOLD = 0.98

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
        
    def getSetOfActionAttribs(self):
#        if self.preState["name"] == "blockA":
#            print "dif: ", self.dif
        r = Set()
        for k,v in self.dif.items():
            if k in self.preState.actionItems and np.linalg.norm(v) > 0.01:
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
    
    def __init__(self, variables):
        self.preConditions = {}
        self.targets = Set()
        self.effect = {}
        self.refCases = []
        self.vecs = []
        self.unusedFeatures = []
        for k in variables:
            self.effect[k] = ITM()
        
    def applyAction(self, state, intStates, strength = 1.0):
        for intState in intStates:
            for k,v in self.effect.items():
                if isinstance(v, ITM):
                    state[k] += strength * v.predict(intState.getVec())
                else:
                    state[k] += strength * v
            
    def rate(self, objectState, intStates):
#        print "rating action: {}, precons: {}".format(self.effect.keys(), self.preConditions)
        if objectState["name"] not in self.targets:
            return 0
        print "rate: ", intStates
        bestScore = 0.0
        for intState in intStates:
            r = {}
            s = 0.0
            for k,v in self.preConditions.items():
                 s += similarities[k](v,intState[k])
                 r[k] = similarities[k](v,intState[k])
            if s > bestScore:
                bestScore = s
             
        print "action: {} got rating: {} for {}".format(self.effect.keys(), r, objectState["name"])
        if len(self.preConditions) != 0:
            return s/len(self.preConditions)
        else:
            return 1.0
            
    def rate2(self, objectState, intStates):
        if objectState["name"] not in self.targets:
            return 0
        bestScore = 0.0
        print "intStates: ", intStates
        for intState in intStates:
            
            for vec in self.vecs:
                s = np.exp(-0.5*np.linalg.norm(vec-intState.getVec()))
                if s > bestScore:
                    bestScore = s
                    
        return bestScore
            
    def update(self, case, intStates):
        self.targets.add(case.preState["name"])
        for intState in intStates:
            self.vecs.append(intState.getVec())
#            for k,v in intState.relItems():
#                if k in self.preConditions:
#                    if np.linalg.norm(v-self.preConditions[k]) > 0.1:
#                        del self.preConditions[k]
#                else:
#                    if len(self.refCases) < 1:
#                        self.preConditions[k] = v 
        for k,v in self.effect.items():
#            if not k in self.effect:
#                self.effect[k] = ITM()
            v.train(Node(0, wIn=intState.getVec(), wOut=case.dif[k]))
                
        self.refCases.append((case, intStates))
        

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
        
    def __repr__(self):
        return "effect: {}".format(self.effect.keys())
        
class Predictor(object):
    """
        Wrapper class for the objectState prediction. Is used to predict
        the state in the next timestep.
    """
    def __init__(self):
#        self.pred = ITM()
        self.pred = {}
        self.targets = Set()
        self.refCases = []
        self.unusedFeatures = []
        
    def predict(self, state):
        res = ObjectState.clone(state)
#        print "predict state: {} \n res: {}".format(state, res)
#        prediction = self.pred.predict(state.toVec())
#        print "prediction for state: {}: {}".format(state["name"], prediction)
#        if state["name"] == "blockA":
#            print "predicting block A with: ", state
        i = 0
        for k in state.relKeys:
#            if hasattr(res[k], "__len__"):
#                l = len(res[k])
#            else:
#                l = 1
#            res[k] += prediction[i:i+l]
#            i += l
            res[k] += self.pred[k].predict(state.getVec())
            
#        if state["name"] == "blockA":
#            print "Prediction: ", res
        return res
        
    def update(self, case, action, worldState):
#        print "updating predictor for ", self.targets
        preState = ObjectState.clone(case.preState)
        action.applyAction(preState, worldState.getInteractionStates(preState["name"]))
#        if case.preState["name"] == "blockA":
#            print "updating with: pre: {} \n post: {}".format(preState, case.postState)
        if len(self.refCases) == 0:
            for k in case.preState.relKeys:
                self.pred[k] = ITM()
#        if case in self.refCases:
#            raise AttributeError("Case already present")
        self.refCases.append(case)
        for k in case.preState.relKeys:
            self.pred[k].train(Node(0, wIn=preState.getVec(), 
                             wOut=case.dif[k]))
#        self.pred.train(Node(0, wIn=case.preState.toVec(self.unusedFeatures), 
#                             wOut = case.postState.toVec()-case.preState.toVec()))   

        
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
#        print "rating for: ", objectState["name"]
        scoreList = [(a.rate2(objectState, worldState.getInteractionStates(objectState["name"])), a) for a in self.actions]
        sortedList = sorted(scoreList, key=itemgetter(0), reverse=True) 
        print "scorelist for {}: {}".format(objectState["name"], sortedList)
        filteredList = filter(lambda x: x[0] > 0.75, sortedList)
#        print "filteredList for {}: {}".format(objectState["name"], filteredList)
        totalScore = np.sum([s[0] for s in filteredList])
        if totalScore == 0:
            totalScore = 1
        res = ObjectState.clone(objectState)
        for s,a in filteredList:
            a.applyAction(res, worldState.getInteractionStates(res["name"]), s/totalScore)
            
        return res
        
    def predictObjectState(self, objectState):
#        print "predict: ", objectState
        for pred in self.predictors:
            if objectState["name"] in pred.targets:
                return pred.predict(objectState)
                
        #Return the state itself if no predictor was found
        return ObjectState.clone(objectState)
    
    def predict(self, worldState, action):
        print "predict"
#        print "Actions: ", [(a.effect.keys(), a.targets) for a in self.actions]
        resultWS = WorldState()
#        resultWS.transM = np.copy(worldState.transM)
#        resultWS.invTrans = np.copy(worldState.invTrans)
#        resultWS.ori = np.copy(worldState.ori)
        for objectState in worldState.objectStates.values():
            newOS = self.applyMostSuitedAction(objectState, worldState, action)
            newOS = self.predictObjectState(newOS)
            objectState = newOS
            resultWS.addObjectState(newOS)
            print "prediction interactions"
            resultWS.parseInteractions()
            
        return resultWS
        
    def checkForAction(self, case, worldState):
        for a in self.actions:
            if case.preState["name"] in a.targets:
                if Set(a.effect.keys()) == case.getSetOfActionAttribs():
                    a.update(case, worldState.getInteractionStates(case.preState["name"]))
                    return a

        #If no action was found
        print "creating new action for {}: {}".format(case.preState["name"], case.getSetOfActionAttribs())
        newAction = Action(case.getSetOfActionAttribs())
        newAction.update(case, worldState.getInteractionStates(case.preState["name"]))
        self.actions.append(newAction)
        return newAction
        
        
    def updateState(self, objectState, worldState, action, resultingOS):
        case = BaseCase(worldState.getObjectState(objectState["name"]), resultingOS)
        predictionRating = resultingOS.score(objectState)
        print "Prediction rating: ", predictionRating
        if predictionRating < THRESHOLD:            
            responsibleAction = self.checkForAction(case, worldState)
            predFound = False
            for pred in self.predictors:
                if objectState["name"] in pred.targets:
                    pred.update(case, responsibleAction, worldState)
                    
                    predFound = True
            if not predFound:
                pred = Predictor()
                pred.targets.add(objectState["name"])
                pred.update(case, responsibleAction, worldState)
                self.predictors.append(pred)
            self.cases.append(case)
    
    def update(self, worldState, action, prediction, result):
        for os in prediction.objectStates.values():
            self.updateState(os, worldState, action, result.getObjectState(os["name"]))
