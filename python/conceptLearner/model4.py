#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:34:43 2015
TODOS:
* Implement possibility to set target!!!
* Implement active learning!!!!
* Block has action like gripper -> wrong?
* action influence appears not to be learned too well for pos
* Check the case selection when finding best cases, it appears some are quite bad.
* The model should remove unnecessary attributes (like spos) itself
* Split relevantScoringKeys and relevantTrainingKeys?
@author: jpoeppel
"""

import numpy as np
from metrics import similarities
from metrics import differences
#from metrics import weights
from common import GAZEBOCMDS as GZCMD
from common import NUMDEC

from sklearn.gaussian_process import GaussianProcess
from topoMaps import ITM
from network import Node, Tree
import copy
import math
from sets import Set
from operator import methodcaller, itemgetter

#from state1 import State, ObjectState, Action, InteractionState, WorldState
from state3 import State, ObjectState, Action, InteractionState, WorldState

THRESHOLD = 0.01
BLOCK_BIAS = 0.2

MAXCASESCORE = 14
MAXSTATESCORE = 12
#PREDICTIONTHRESHOLD = 0.5
PREDICTIONTHRESHOLD = MAXSTATESCORE - 0.01
TARGETTHRESHOLD = MAXCASESCORE - 0.05


#
import logging
logging.basicConfig()


        
        

class BaseCase(object):
    
    def __init__(self, pre, action, post):
        assert isinstance(pre, State), "{} is not a State object.".format(pre)
        assert isinstance(post, State), "{} is not a State object.".format(post)
        assert isinstance(action, Action), "{} is not an Action object.".format(action)
        assert (pre.keys()==post.keys()), "Pre and post states have different keys: {}, {}.".format(pre.keys(), post.keys())
        self.preState = pre
        self.postState = post
        self.action = action
        self.dif = {}
        self.abstCase = None
        for k in pre.relKeys:
            self.dif[k] = post[k]-pre[k]
            
        
    def getSetOfAttribs(self):
        """
            Returns the list of attributes that changed more than THRESHOLD.
        """
        r = Set()
        for k in self.dif.keys():
            if np.linalg.norm(self.dif[k]) > 0.01:
                r.add(k)
        return r
        
        
    def getListOfConstants(self):
        #TODO make more efficient by storing these values
        r = []
        for k in self.dif.keys():
            if np.linalg.norm(self.dif[k]) <= 0.01:
                r.append((k,self.preState[k]))
        return r
    
    def predict(self, state,action, attrib):
        return self.dif[attrib]
        
    def score(self, state, action):#, weights):
        s = self.preState.score(state)#, weights)
        s += self.action.score(action)#, weights)
        return s
        
    def __eq__(self, other):
        if not isinstance(other, BaseCase):
            return False
        return self.preState == other.preState and self.action == other.action and self.postState == other.postState
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __repr__(self):
        return "Pre: {} \n Action: {} \n Post: {}".format(self.preState,self.action,self.postState)
        
class AbstractCase(object):
    
    def __init__(self, case):
        assert isinstance(case, BaseCase), "case is not a BaseCase object."
        self.id = 0
        self.refCases = []
        self.avgPrediction = 0.0
        self.numPredictions = 0
        self.name = ""
        self.variables = Set() #List of attributes that changed 
        self.attribs = {} # Dictionary holding the attributs:[values,] pairs for the not changing attribs of the references
        self.predictors = {}
        self.variables.update(case.getSetOfAttribs())
        self.preCons = {}
        self.constants = {}
        self.gaussians = {}
        self.errorGaussians = {}
        self.numErrorCases = 0
        self.weights= {}
        self.values = {}
        self.minima = {}
        self.maxima = {}
        for k in self.variables:
            self.predictors[k] = ITM()
        self.addRef(case)
        
    def predict(self, state, action):
        resultState = copy.deepcopy(state)
        if len(self.refCases) > 1:
#            print "resultState intId: ", resultState["intId"]
            for k in self.variables:
                prediction = self.predictors[k].predict(np.concatenate((state.toVec(self.constants),action.toVec(self.constants))))
#                if state["sname"] == "blockA":
#                print "variable: {}, prediction: {}".format(k, prediction)
                if prediction != None:
                    resultState[k] = state[k] + prediction
                else:
                    resultState[k] = state[k] + self.refCases[0].predict(state, action, k)

        else:
#            print "predicting with only one ref"
            for k in self.variables:
                resultState[k] = state[k] + self.refCases[0].predict(state, action, k)
        return resultState
        
    def getAction(self, pre, var, weights, dif):
        action = np.zeros(4)
#        numVariables = len(self.variables)
        norm = 0.0
        for k in var:
            norm += weights[k]
#            print "case: {}, predicts: {} for variable K: {}".format(self.variables,self.predictors[k].predictAction(pre.toVec(), dif[k]),k )
            action += weights[k] * self.predictors[k].predictAction(pre.toVec(self.constants), dif[k])
            
        
        action /= norm
        res = Action(cmd = action[0], direction=action[1:])
        return res
            
    def addErrorCase(self, case):
        self.numErrorCases += 1
#        self.updateGaussians(self.errorGaussians, self.numErrorCases, case)
        
                
        
            
    def score(self, state, action):
        s = 0.0
        
        # Only use ACs with at least 2 references
#        if len(self.refCases) < 2:
#            return 0
        
#        for k,v in self.constants.items():
#            for k2,v2 in state.relevantItems() + action.relevantItems():
#                if k == k2:
#                    if np.linalg.norm(v-v2) > 0.01:
##                        if state["sid"] == 15:
#                        print "AC: {} failed because of k: {}, constant: {}, actual: {}, with {} numRefs".format(self.variables, k, v, v2, len(self.refCases))
#                        return 0
##        
        for k,v in state.relevantItems() + action.relevantItems():
#            
            if not k in self.constants.keys():
                if hasattr(v, "__len__"):
                    ori = np.zeros(len(v))
                else:
                    ori = 0
                distToOri = np.linalg.norm(v-ori)
                if distToOri < self.minima[k]:
                    s += 0
                elif distToOri > self.maxima[k]:
                    s += 0
                else:
                    score = 0.0                
                    bestScore = 0.0
                    for ref in self.refCases:
                        if ref.preState.has_key(k):
                            score = similarities[k](ref.preState[k], v)
                        else:
                            score = similarities[k](ref.action[k], v)
                        if score > bestScore:
                            bestScore = score
                    s += bestScore
            else:
                if np.linalg.norm(v-self.constants[k]) > 0.01:
#                    print "AC: {} failed because of k: {}, constant: {}, actual: {}, with {} numRefs".format(self.variables, k, self.constants[k], v, len(self.refCases))
                    return 0
                # Reward ACs with many constants that were met!
                s += 1
        
        return s
    
    def updatePredictionScore(self, score):
        self.numPredictions += 1
        self.avgPrediction += (score-self.avgPrediction)/(float(self.numPredictions))
#        if self.avgPrediction <= 0.0001:
#            raise Exception("something is going wrong when computing avgPrediction! score: {}, numPred: {}".format(score, self.numPredictions))
        
    def addRef(self, ref):
#        print "adding ref, old constants: ", self.constants
        if ref in self.refCases:
            raise TypeError("ref already in refCases")
        for k,v in ref.preState.relevantItems() + ref.action.relevantItems():
            if self.constants.has_key(k):
                if np.linalg.norm(v-self.constants[k]) > 0.001:
                    print "deleting constant {} in ac {}".format(k, self.variables)
                    del self.constants[k]
                    self.retrain()
            else:
                if len(self.refCases) == 0:
                    self.constants[k] = v
            if hasattr(v, "__len__"):
                ori = np.zeros(len(v))
            else:
                ori = 0
            distToOri = np.linalg.norm(v-ori)
            if not self.minima.has_key(k) or distToOri < self.minima[k]:
                self.minima[k] = distToOri
            if not self.maxima.has_key(k) or distToOri > self.maxima[k]:
                self.maxima[k] = distToOri
                
         
        self.refCases.append(ref)
        ref.abstCase = self
        self.updatePredictorsITM(ref)
        
        self.updateGaussians(self.gaussians, len(self.refCases), ref)        
#        self.updatePredictorsGP()
        
    def updateGaussians(self, gaussians, numData, ref):
        for k,v in ref.preState.relevantItems() + ref.action.relevantItems():
            if hasattr(v, "__len__"):
                dim = len(v)
            else:
                dim = 1
                
            if not gaussians.has_key(k):
                gaussians[k] = (np.array(v)[np.newaxis], np.identity(dim), 1, np.identity(dim))
            else:
                muO, covO, detO, invO = gaussians[k]
                mu = (1-1.0/numData)*muO + 1.0/numData*v
                cov = (1-1.0/numData)*(covO+1.0/numData*np.dot((v-muO).T,(v-muO)))
                inv = np.linalg.inv(cov)
                gaussians[k] = (mu, cov, np.linalg.det(cov), inv )

            
    def getData(self, attrib):
        
        numCases = len(self.refCases)
        if self.refCases[0].preState.has_key(attrib):
            if hasattr(self.refCases[0].preState[attrib], "__len__"):
                dim = len(self.refCases[0].preState[attrib])
            else:
                dim = 1
            data = np.zeros((numCases,dim))
            for i in range(numCases):
                data[i,:] = self.refCases[i].preState[attrib]
        elif self.refCases[0].action.has_key(attrib):
            if hasattr(self.refCases[0].action[attrib], "__len__"):
                dim = len(self.refCases[0].action[attrib])
            else:
                dim = 1
            data = np.zeros((numCases,dim))
            for i in range(numCases):
                data[i,:] = self.refCases[i].action[attrib]
        else:
            raise TypeError("Invalid attribute: ", attrib)
            
        return data
        
        
    def retrain(self):
        for k in self.variables:
            self.predictors[k] = ITM()
        for c in self.refCases:
            self.updatePredictorsITM(c)
#        self.updatePredictorsGP()
        
    def updatePredictorsITM(self, case):
        for k in self.variables:
            self.predictors[k].train(self.toNode(case, k))
            
    def toNode(self, case, attrib):
        node = Node(0, wIn=case.preState.toVec(self.constants), action=case.action.toVec(self.constants),
                    wOut=case.postState[attrib]-case.preState[attrib])
        return node
        
    def updatePredictorsGP(self):
        if len(self.refCases) > 1:
            for k in self.variables:
                self.predictors[k] = GaussianProcess(corr='cubic')
                data, labels = self.getTrainingData(k)
                self.predictors[k].fit(data, labels)
                
    def getTrainingData(self, attrib):
        inputs = []
        outputs = []
        for c in self.refCases:
            inputs.append(np.concatenate((c.preState.toVec(self.preCons),c.action.toVec(self.preCons))))
            outputs.append(c.postState[attrib]- c.preState[attrib])
        return inputs, outputs
    
    def createTarget(self, worldState):
        if self.constants.has_key("sname"):
            intState = worldState.getInteractionState(self.constants["sname"])
        else:
            intState = worldState.getInteractionState("gripper")
            
#        print "choosing state with name: ", intState["sname"]
        target = copy.deepcopy(intState)
        for k in self.variables:
            if hasattr(target[k], "__len__"):
                target[k] += 2*np.random.rand(len(target[k]))
            else:
                target[k] += 2*np.random.rand()
                
        target.relKeys = self.variables
        return target
    
    def getBestRef(self, state, action, weights):
        bestCase = None
        bestScore = 0.0
        for c in self.refCases:
            s = c.score(state,action, weights)
            if s > bestScore:
                bestScore = s
                bestCase = c
        return bestCase
        
    
    
class ModelCBR(object):
    
    def __init__(self):
        self.cases = []
        self.abstractCases = []
        self.numZeroCase = 0
        self.numCorrectCase = 0
        self.numPredictions = 0
        self.target = None
        self.weights = {}
        self.lastAC = None
        self.avgCorrectPrediction = 0.0
        self.correctPredictions = 0
        
    def getAction(self, state):

        print "target: ", self.target
        if self.target != None:
            gripperInt = state.getInteractionState(self.target["sname"])
            if gripperInt != None:
                variables = Set()
                dif = {}
                for k in gripperInt.relevantKeys():
                    dif[k] = (self.target[k] - gripperInt[k])/10.0
                    if k in self.target.relKeys and np.linalg.norm(dif[k]) > THRESHOLD:
                        variables.add(k)           
                bestAction = None
                bestScore = 0
                for ac in self.abstractCases:
                    if variables.issubset(ac.variables):
                        action = ac.getAction(gripperInt,variables, self.target.weights, dif)
                        prediction = ac.predict(gripperInt, action)
                        score = self.target.score(prediction)#*ac.avgPrediction
#                        print "abstract case: ", ac.variables
#                        print "ac avgPrediction: ", ac.avgPrediction
#                        print "numPredictions: ", ac.numPredictions
#                        print "predicted pos: ", prediction["spos"]
#                        print "score to target: {}, for action: {}".format(score, action)                    
                        if score > bestScore:
                            bestScore = score
                            bestAction = action
    #                        bestAction["cmd"] = 1
            if bestAction != None:
                print "using Action: ", bestAction
                return bestAction

        return self.getRandomAction(state, BLOCK_BIAS)
            
    def getRandomAction(self, state, blockbias = 0):
        print "getting random action"
        rnd = np.random.rand()
        a = Action()
        if rnd < 0.4:
            a["cmd"] = GZCMD["MOVE"]
            if np.random.rand() < blockbias:
                gripperInt = state.getInteractionState("gripper")
                a["mvDir"] = gripperInt["dir"]/np.linalg.norm(gripperInt["dir"]) + (np.random.rand(3)-0.5)
            else:
#            a["dir"] = np.array([1.2,0,0])
                a["mvDir"] = np.random.rand(3)*2-1
        elif rnd < 0.5:
            a["cmd"] = GZCMD["MOVE"]
            a["mvDir"] = np.array([0,0,0])
        else:
            a["cmd"] = GZCMD["NOTHING"]
#        a["mvDir"] *= 2.0
        a["mvDir"][2] = 0
        norm = np.linalg.norm(a["mvDir"])
        if norm > 0.25:
            a["mvDir"] /= 5*np.linalg.norm(a["mvDir"])
        return a
    
    def setTarget(self, postState = None):
        self.target = postState
    
    def getBestCase(self, state, action):
        
#        print "getBestCase with state: {} \n action: {}".format(state, action)
        
        bestCase = None
        scoreList = [(c,c.score(state,action)) for c in self.abstractCases]
#        scoreList = [(c.abstCase,c.score(state,action)) for c in self.cases]
        
#        sortedList = sorted(self.abstractCases, key=methodcaller('score', state, action), reverse= True)
        sortedList = sorted(scoreList, key=itemgetter(1), reverse=True) 
#        self.lastScorelist = [(s, sorted(c.variables), len(c.refCases)) for c,s in sortedList]
#        if state["sid"] == 15:
#        print "ScoreList: ", [(s, sorted(c.variables), len(c.refCases)) for c,s in sortedList]
        if len(sortedList) > 0:
#            if sortedList[0][1] == 0 and self.lastAC != None:
#                bestCase = self.lastAC
#            else:
            bestCase = sortedList[0][0]
            
            print "selected AC: ", bestCase.variables
        if isinstance(bestCase, AbstractCase):
            if bestCase.variables == []:
                self.numZeroCase += 1

            return bestCase
        else:
            return None
    
    def predictIntState(self, state, action):
        print "predict: ", state["sname"]
        bestCase = self.getBestCase(state, action)
        if bestCase != None:
            self.lastAC = bestCase
            return bestCase.predict(state, action), bestCase
        else:
            return state, bestCase
    
    def predict(self, worldState, action):
        
        predictionWs = WorldState()
        predictionWs.transM = np.copy(worldState.transM)
        predictionWs.invTrans = np.copy(worldState.invTrans)
        predictionWs.ori = np.copy(worldState.ori)
        transformedAction = copy.deepcopy(action)
        transformedAction.transform(worldState.invTrans)
        for intState in worldState.interactionStates.values():
            self.numPredictions += 1
            
            prediction, usedCase = self.predictIntState(intState, transformedAction)
            predictionWs.addInteractionState(prediction, usedCase)
#        print "resulting prediction: ", predictionWs.interactionStates
        return predictionWs
        
    def updateState(self, state, action, prediction, result, usedCase):
        """
        Parameters
        
        state: InteractionState
        Action: Action
        prediction: InteractionState
        result: Interaction
        usedCase: AbstractCase
        """

        
        newCase = BaseCase(state, action, result)
#        print "New case difs: ", newCase.dif
        attribSet = newCase.getSetOfAttribs()
        predictionRating = result.rate(prediction)#, self.weights)
        predictionScore = sum(predictionRating.values())
        print "Correct attribSet for state: {} : {}".format(result["sname"], sorted(attribSet))
        print "predictionScore: ", predictionScore
        abstractCase = None
        for ac in self.abstractCases:
            if ac.variables == attribSet:
                abstractCase = ac
                #TODO consider search for all of them in case we distinguis by certain features
                break
            
        if abstractCase != None:    
            correctPrediction = abstractCase.predict(state, action)
            correctRating = result.rate(correctPrediction)
            worstRating = min(correctRating.items(), key=itemgetter(1))
            self.correctPredictions += 1
            self.avgCorrectPrediction += (sum(correctRating.values())-self.avgCorrectPrediction)/(float(self.correctPredictions))
#            print "Prediction Score of correctCase prediction: {}, worst attrib: {} ({})".format(sum(correctRating.values()), worstRating[0], worstRating[1]) 
        if usedCase != None:
            if usedCase.variables == attribSet:
                print "correct case selected!!!!!!!!!!!!!!!!!"
                usedCase.updatePredictionScore(predictionScore)
                self.numCorrectCase += 1
                
        if predictionScore < PREDICTIONTHRESHOLD:
            if abstractCase != None:
                    try:
                        abstractCase.addRef(newCase)
                    except TypeError, e:
                        print "case was already present"
                    else:
                        self.addBaseCase(newCase)
                    
            else:
                #Create a new abstract case
                print "new Abstract case: ", attribSet
                self.abstractCases.append(AbstractCase(newCase))
                self.addBaseCase(newCase)

    def addBaseCase(self, newCase):
        self.cases.append(newCase)
        for k in newCase.preState.relKeys + newCase.action.relKeys:
            if not self.weights.has_key(k):
                self.weights[k] = 1.0
        self.normaliseWeights()
        
    def normaliseWeights(self):
        norm = sum(self.weights.values())
        for k in self.weights.keys():
            self.weights[k] /= norm
    
    def update(self, worldState, action, prediction, result):
        transformedAction = copy.deepcopy(action)
        transformedAction.transform(worldState.invTrans)
        if np.all(worldState.transM != result.transM):
            raise TypeError("Wrong coordinate system!")
        for intState in worldState.interactionStates.keys():
            self.updateState(worldState.interactionStates[intState], transformedAction, prediction.interactionStates[intState], 
                             result.interactionStates[intState], prediction.predictionCases[intState])
        