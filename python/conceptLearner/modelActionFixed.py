#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:53:10 2015
A fixed adaptation to the modelAction, where the number of actions
and action selection is hardcoded to test the general idea.
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
import common
from network import Node
from network import LVQNeuron, LVQNeuralNet
import random
#if SINGLE_INTSTATE:
#    from state3 import State, ObjectState, InteractionState, WorldState
#else:
#    from state2 import State, ObjectState, InteractionState, WorldState

from state4 import State, ObjectState, InteractionState, WorldState

from state2 import Action as GripperAction

from sklearn import tree

THRESHOLD = 0.995



class BaseCase(object):
    
    def __init__(self, pre, post):
        assert isinstance(pre, State), "{} is not a State object.".format(pre)
        assert isinstance(post, State), "{} is not a State object.".format(post)
        
        self.preState = pre
        self.postState = post
        self.dif = {}
        for k in pre.relKeys + pre.actionItems:
            self.dif[k] = np.round(post[k] - pre[k], common.NUMDEC)
            
            
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
#        for k,v in self.dif.items():
#            if k in self.preState.actionItems and np.linalg.norm(v) > 0.01:
#                r.add(k)
        for k in self.preState.actionItems:
            if np.linalg.norm(self.dif[k]) > 0.01:
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
        self.id = -1
        self.preConditions = {}
        self.targets = Set()
        self.effect = {}
        self.refCases = []
        self.vecs = []
        self.changeSet = Set()
        self.unusedFeatures = []
        self.weights = np.ones(len(InteractionState().getVec()))
        self.weights /= sum(self.weights)
        for k in variables:
            self.effect[k] = ITM()
        
    def applyAction(self, state, intStates, strength = 1.0):
        for intState in intStates:
            for k,v in self.effect.items():
                if isinstance(v, ITM):
                    state[k] += strength * v.predict(intState.getVec())
                else:
                    state[k] += strength * v
            
            
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
#        if len(self.refCases) < 10:
        for k,v in self.effect.items():
#            if not k in self.effect:
#                self.effect[k] = ITM()
            v.train(Node(0, wIn=intState.getVec(), wOut=case.dif[k]))
#            v.train(Node(0, wIn=intState.getVec(), wOut=case.postState[k]))
                
        self.refCases.append((case, intStates))
        
    def getPrototype(self):
        if len(self.refCases) > 0:
            ref, refInts = random.choice(self.refCases)
            vec = np.copy(random.choice(refInts).getVec())
            vec += (np.random.rand(len(vec))- 0.5) * 0.01
        else:
            print "prototype without ref!!!!"
            raise NotImplementedError
            vec = np.random.rand(len(InteractionState().getVec()))-0.5
        return vec
        
    def getReferenzes(self):
        if len(self.refCases) > 0:
#            return [(random.choice(vec[1]).vec, self.id) for vec in self.refCases]
            return [(v, self.id, self.weights) for v in self.vecs]
        else:
            return [(self.getPrototype(), self.id)]
        

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
#            print "prediction OS: ", state
#            p = np.round(self.pred[k].predict(state.getVec(np.array([5,6,7,8]))), 3)
            p = np.round(self.pred[k].predict(state.getVec()), 3)
#            if state["name"] == "blockA":
#                print "Prediction for {}: {}".format(k, p)
            res[k] += p #self.pred[k].predict(state.getVec())
            
#        if state["name"] == "blockA":
#            print "Prediction: ", res
        return res
        
    def updateFix(self, case):
        if len(self.refCases) == 0:
            for k in case.preState.relKeys:
                self.pred[k] = ITM()
        self.refCases.append(case)
        for k in case.preState.relKeys:
            self.pred[k].train(Node(0,wIn=case.preState.getVec(), wOut=case.dif[k]))
        
    def update(self, case, action, worldAction, worldState):
#        print "updating predictor for ", self.targets
#        print "case preState: ", case.preState.vec
        preState = ObjectState.clone(case.preState)
        worldAction.applyAction(preState, worldState.getInteractionStates(preState["name"]))
        if preState["name"] == "gripper":
            preState["linVel"][:3] = action["mvDir"]
#        if case.preState["name"] == "blockA":
#            print "updating with: pre: {} \n post: {}".format(preState, case.postState)
        if len(self.refCases) == 0:
            for k in case.preState.relKeys:
                self.pred[k] = ITM()
#        if case in self.refCases:
#            raise AttributeError("Case already present")
        self.refCases.append(case)
        for k in case.preState.relKeys:
#            print "updating feature: ", k
#            print "inputVec: ", preState.getVec(np.array([5,6,7,8]))
            self.pred[k].train(Node(0, wIn=preState.getVec(), 
                             wOut=case.dif[k]))

        
class ModelAction(object):
    
    def __init__(self):
        self.predictors = []
        self.actions = {}
        self.cases = []
        self.lvq = LVQNeuralNet(len(InteractionState().getVec()))
        self.tree = None
        #Action0: No change
        #Action1: GetContact
        #Action2: LooseContact
        for i in xrange(3):
            action = Action(ObjectState().relKeys)
            action.id = i
            self.actions[i] = action
        
    def getAction(self):
        pass
    
    def applyActions(self, objectState, worldState, action):
        res = ObjectState.clone(objectState)
        intStates = worldState.getInteractionStates(res["name"])
        for intState in intStates:
            print "closing: ", intState["closing"]
            print "dist: ", intState["dist"]
            print "contact: ", intState["contact"]
            if intState["contact"]:
                if intState["closing"] > 0: #TODO not sufficient!!!
                    print "Applying loose contact action"
                    self.actions[2].applyAction(res, intStates)
                else:
                    print "Apply no change"
                    self.actions[0].applyAction(res, intStates)
            else:
                if intState["closing"] < -10*intState["dist"]:
                    print "Apply get contact"
                    self.actions[1].applyAction(res, intStates)
                else:
                    print "Apply no change"
                    self.actions[0].applyAction(res, intStates)
                    
        
        if objectState["name"] == "gripper":
            res["linVel"][:3] = action["mvDir"]
            
        return res
        
    def predictObjectState(self, objectState):
#        print "predict: ", objectState
        for pred in self.predictors:
            if objectState["name"] in pred.targets:
                return pred.predict(objectState)
                
        #Return the state itself if no predictor was found
        return ObjectState.clone(objectState)
    
    def predict(self, worldState, action):
#        print "predict"
#        print "Actions: ", [(a.effect.keys(), a.targets) for a in self.actions]
        resultWS = WorldState()
#        resultWS.transM = np.copy(worldState.transM)
#        resultWS.invTrans = np.copy(worldState.invTrans)
#        resultWS.ori = np.copy(worldState.ori)
        for objectState in worldState.objectStates.values():
#            print "OS before action: ", objectState
            newOS = self.applyActions(objectState, worldState, action)# self.applyMostSuitedAction2(objectState, worldState, action)
#            if objectState["name"] == "blockA": 
#                print "newOS after action: ", newOS
            newOS = self.predictObjectState(newOS)
#            objectState = newOS
            resultWS.addObjectState(newOS)
#            print "prediction interactions"
            resultWS.parseInteractions()
            
        return resultWS
        
        
    def updateTree(self):
        self.tree = tree.DecisionTreeClassifier(criterion="gini", class_weight='auto')
        c = 0
        for a in self.actions.values():
            c += len(a.vecs)
            l = len(a.vecs[0])
        i=0
        X = np.zeros((c, l))
        Y = np.zeros(c)
        for l, a in self.actions.items():
            for v in a.vecs:
                X[i,:] = v
                Y[i] = l
                i +=1
        self.tree.fit(X,Y)
        
    def updateLVQ(self):
        self.lvq = LVQNeuralNet(len(InteractionState().getVec()))
        for l,a in self.actions.items():
            for i in xrange(NUM_PROTOTYPES):
                self.lvq.addNeuron(a.getPrototype(), l, a.weights)
        trainData = []
        for a in self.actions.values():
            trainData.extend(a.getReferenzes())
        np.random.shuffle(trainData)
#        print "traindata: ", trainData
        for v, l, w in trainData:
            self.lvq.train(v, l, w)
            
    def updateActions(self, case, worldState):
        if case.preState["contact"] == case.postState["contact"]:
            print "updating no Change"
            self.actions[0].update(case, worldState.getInteractionStates(case.preState["name"]))
        else:
            if case.preState["contact"] == None:
                print "updating get contact"
                self.actions[1].update(case, worldState.getInteractionStates(case.preState["name"]))
            else:
                print "updating loose contact"
                self.actions[2].update(case, worldState.getInteractionStates(case.preState["name"]))
        
    def updateState(self, predictedOS, worldState, action, resultingOS):
        
        preState = worldState.getObjectState(predictedOS["name"])
        adaptedPreState = self.applyActions(preState, worldState, action)
        adaptedCase = BaseCase(adaptedPreState, resultingOS)
        
        case = BaseCase(worldState.getObjectState(predictedOS["name"]), resultingOS)
        predictionRating = resultingOS.score(predictedOS)
        print "Prediction rating for {}: {}".format(predictedOS["name"], predictionRating)
#        print "update casePrestate: ", case.preState.vec
#        print "prediction: ", predictedOS
#        print "result: ", {k: resultingOS[k] for k in resultingOS.actionItems}
#        responsibleAction = self.checkForAction3(case, worldState)
        self.updateActions(case, worldState)
        if predictionRating < THRESHOLD:            
#            responsibleAction = self.checkForAction2(case, worldState)
#            print "Responsible action for {}: {}".format(objectState["name"], responsibleAction)
            predFound = False
            for pred in self.predictors:
                if predictedOS["name"] in pred.targets:
                    pred.updateFix(adaptedCase)
                    
                    predFound = True
            if not predFound:
                pred = Predictor()
                pred.targets.add(predictedOS["name"])
                pred.updateFix(adaptedCase)
#                pred.update(case, action, responsibleAction, worldState)
                self.predictors.append(pred)
            self.cases.append(case)
    
    def update(self, worldState, action, prediction, result):
        for os in prediction.objectStates.values():
            self.updateState(os, worldState, action, result.getObjectState(os["name"]))

    def getGraphViz(self, dot_data):
        if self.tree != None:
            IS = InteractionState()
            tree.export_graphviz(self.tree, out_file=dot_data, feature_names=IS.features[IS.mask])