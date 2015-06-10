#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:34:43 2015
TODOS:
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
import heapq
from sets import Set
from operator import methodcaller, itemgetter

from sklearn import svm
from sklearn import preprocessing
from sklearn import tree

from config import SINGLE_INTSTATE


#from state1 import State, ObjectState, Action, InteractionState, WorldState
if SINGLE_INTSTATE:
    from state3 import State, ObjectState, Action, InteractionState, WorldState
else:
    from state2 import State, ObjectState, Action, InteractionState, WorldState

THRESHOLD = 0.01
BLOCK_BIAS = 0.2

MAXCASESCORE = 14-5
MAXSTATESCORE = 12-5
#PREDICTIONTHRESHOLD = 0.5
PREDICTIONTHRESHOLD = MAXSTATESCORE - 0.01
TARGETTHRESHOLD = MAXCASESCORE - 0.05

NUM_SAMPLES = 10

USE_CONSTANTS = True
#
import logging
logging.basicConfig()


MAX_DEPTH = 5
        

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
    
    def __init__(self, case, acId = 0):
        assert isinstance(case, BaseCase), "case is not a BaseCase object."
        self.id = acId
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
                if USE_CONSTANTS:
                    prediction = self.predictors[k].predict(np.concatenate((state.toVec(self.constants),action.toVec(self.constants))))
                else:
                    prediction = self.predictors[k].predict(np.concatenate((state.toVec(),action.toVec())))
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
        
    def getAction2(self, var, dif):
        action = np.zeros(4)
        inputs = np.zeros(len(self.refCases[0].preState.toVec(self.constants)))
        difChange = 0.0
        norm = 0.0
        for k in var:
            norm += 1.0
            partialAction, partialInputs, expectedDif = self.predictors[k].getAction(dif[k])
            action += partialAction
            inputs += partialInputs
            print "expectedDif for k: {}, {}".format(k, expectedDif)
            difChange += np.linalg.norm(dif[k] - expectedDif) - np.linalg.norm(dif[k])
        print "Dif change: ", difChange
        if difChange > -0.01:
            print "Best Action: {} is too bad {}".format(action/norm, difChange)
            return None,None
        action /= norm
        inputs /= norm
        res = Action(cmd = action[0], direction=action[1:])
        preCons = InteractionState.fromVector(inputs, self.constants)
        res = Action(cmd = action[0], direction=action[1:])
        return res, preCons
        
    def getAction(self, pre, var, dif, weights = None):
#        print "getAction from ac: ", self.variables
        action = np.zeros(4)
        norm = 0.0
        if weights != None:
            for k in var:
                norm += weights[k]
                if USE_CONSTANTS:
                    action += weights[k] * self.predictors[k].predictAction(pre.toVec(self.constants), dif[k])
                else:
                    action += weights[k] * self.predictors[k].predictAction(pre.toVec(), dif[k])
        else:
            for k in var:
                norm += 1.0
                if USE_CONSTANTS:
                    partialAction = self.predictors[k].predictAction(pre.toVec(self.constants), dif[k])
                else:
                    partialAction = self.predictors[k].predictAction(pre.toVec(), dif[k])
                action += partialAction
#                print "partial Action: ", partialAction
        action /= norm
        res = Action(cmd = action[0], direction=action[1:])
        return res
            
    def addErrorCase(self, case):
        self.numErrorCases += 1
#        self.updateGaussians(self.errorGaussians, self.numErrorCases, case)
        
    def checkPreCons(self, preCons, state):
        for k,v in state.relevantItems():
            if k in self.constants:
                if np.linalg.norm(v-self.constants[k]) > 0.05:
                    print "Failed at constant: {}".format(k)
                    return False
            else:
                if np.any(preCons[k]*v < 0):
                    print "Failed at side of k: {}, (pre {}, given: {})".format(k, preCons[k], v)
                    return False
        
        return True
        
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
        for k,v in ref.preState.relevantItems():# + ref.action.relevantItems():
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
        if USE_CONSTANTS:
            node = Node(0, wIn=case.preState.toVec(self.constants), action=case.action.toVec(self.constants),
                    wOut=case.postState[attrib]-case.preState[attrib])
        else:
            node = Node(0, wIn=case.preState.toVec(), action=case.action.toVec(),
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
        self.abstractCases = {}
        self.numZeroCase = 0
        self.numCorrectCase = 0
        self.numPredictions = 0
        self.target = None
        self.weights = {}
        self.lastAC = None
        self.avgCorrectPrediction = 0.0
        self.correctPredictions = 0
        self.aCClassifier = None
        self.scaler = None
        
    def createRelativeTargetInteraction(self, worldState, target):
        
        relTarget = copy.deepcopy(target)
        relTarget.transform(worldState.invTrans, -worldState.ori)
        print "relTarget: ", relTarget
        if target["name"] == "blockA":
            targetInt = copy.deepcopy(worldState.getInteractionState("gripper"))
            targetInt["intId"] = -1            
            targetInt.fill(relTarget)
            targetInt.relKeys = ["opos", "oeuler"]
#            targetInt.weights = {"opos":20, "oeuler":1}
        elif target["name"] == "gripper":
            targetInt = InteractionState(-1, relTarget)
            targetInt.relKeys = ["spos"]#, "oeuler"] #TODO Change so that is not hardcoded anymore
#        targetInt.weights = {"opos":20, "oeuler":1}
        return targetInt        
        
    def getAction3(self, worldState):
        bestAction = None
        if isinstance(self.target, ObjectState):
#            relTargetOs = copy.deepcopy(self.target)
#            relTargetOs.transform(worldState.invTrans, -worldState.ori)
#            relTargetInt = self.createRelativeTargetInteraction(worldState, self.target)
#            givenInteraction = worldState.getInteractionState("gripper")
#            curOs = givenInteraction.getObjectState(self.target["name"])
#            curScore = self.target.score(curOs)
#            curOsGlobal = copy.deepcopy(curOs)
#            curOsGlobal.transform(worldState.transM, worldState.ori)
            bestAction = self.findBestAction2(worldState, self.target)

                        
        elif isinstance(self.target, InteractionState):
            raise NotImplementedError
        print "bestAction: ", bestAction
        if bestAction != None:
            if np.random.rand() < 0.1:
                bestAction["mvDir"][:2] += (np.random.rand(2)-0.5)*0.1
            return bestAction
        else:
            return self.getRandomAction(worldState, BLOCK_BIAS)
            
    def findBestAction2(self, worldState, target, depth = 0):
        """
        Parameters
        ----------
        worldState: state.WorldState
        target: state.ObjectState
        depth: Int
        """
        if depth > MAX_DEPTH:
            return None
            
        givenInteraction = worldState.getInteractionState("gripper")
        relTargetInt = self.createRelativeTargetInteraction(worldState, target)
        difs = {}
        for k in relTargetInt.relKeys:
            difs[k] = relTargetInt[k] - givenInteraction[k]
            
        print "Difs: ", difs
        difSet = Set(difs.keys())
        action = preCons = None
        for ac in self.abstractCases.values():
            if ac.variables.issuperset(difSet):
                action, preCons = ac.getAction2(difSet, difs)
                if action != None:
                    if ac.checkPreCons(preCons, givenInteraction):
                        return action
                    else:
                        #create subtarget
                    
                        target = preCons.getTarget(givenInteraction)
                        print "!!!!!!!!!! Creating suptarget: ", target
                        target.transform(worldState.transM, worldState.ori)
                        
                        return self.findBestAction2(worldState, target, depth +1)
                    
        return None
        
    def createTargetInteraction(self, worldState, target):
        
        relTarget = copy.deepcopy(target)
        print "relTarget: ", relTarget
        if target["name"] == "blockA":
            targetInt = copy.deepcopy(worldState.getInteractionState("gripper"))
            targetInt["intId"] = -1            
            targetInt.fill(relTarget)
            targetInt.relKeys = ["opos", "oeuler"]
#            targetInt.weights = {"opos":20, "oeuler":1}
        elif target["name"] == "gripper":
            targetInt = InteractionState(-1, relTarget)
            targetInt.relKeys = ["spos"]#, "oeuler"] #TODO Change so that is not hardcoded anymore
#        targetInt.weights = {"opos":20, "oeuler":1}
        return targetInt     
        
    def getAction(self, worldState):
        worldTarget = copy.deepcopy(self.target)
        worldTargetInt = self.createTargetInteraction(worldState, worldTarget)
        givenInteraction = copy.deepcopy(worldState.getInteractionState("gripper"))
        givenInteraction.transform(worldState.transM, worldState.ori)
        statePlan = self.plan(givenInteraction, worldTargetInt)
#        print "statePlan: ", statePlan
        return self.getRandomAction(worldState, BLOCK_BIAS)
        
    def plan(self, start, goal):
        """
            Adaptation of the A*-algorithm that searches the interactionStateSpace 
            heuristically. Neighbours are generated by drawing samples from the possible
            actions and predicting the next state.
            
            Parameters
            ----------
            start: state.InteractionState
                Start state in world coordinates
            goal: state.InteractionState
                Goal state in world coordinates
        """
        frontier = []
        heapq.heappush(frontier, (0, 0, start))
        nodeCounter = 1
        came_from = {start: None}
        cost_so_far = {start: 0}
        iterations = 0
        while not len(frontier) == 0 and iterations < 10:
            iterations += 1
            current = heapq.heappop(frontier)[2]
            
            if current == goal:
                break
            localCurrent, transM, ori = current.getLocalTransformed() 
            for nextState in [self.predictIntState(localCurrent, action)[0] for action in Action.sample(NUM_SAMPLES)]:
                nextState.transform(transM, ori)
                new_cost = cost_so_far[current] + 1
                if nextState not in cost_so_far or new_cost < cost_so_far[nextState]:
                    cost_so_far[nextState] = new_cost
                    priority = new_cost + goal.score(nextState)
                    heapq.heappush(frontier, (priority, nodeCounter, nextState))
                    nodeCounter+=1
                    came_from[nextState] = current
                    
        return came_from, cost_so_far
            
    def getAction2(self, worldState):
        
        bestAction = None
        if isinstance(self.target, ObjectState):
            relTargetOs = copy.deepcopy(self.target)
            relTargetOs.transform(worldState.invTrans, -worldState.ori)
            #Transform target to relative coordinate system
            givenInteraction = worldState.getInteractionState("gripper")
            curOs = givenInteraction.getObjectState(self.target["name"])
#            curScore = self.target.score(curOs)
            curOsGlobal = copy.deepcopy(curOs)
            curOsGlobal.transform(worldState.transM, worldState.ori)
            print "curOSglobal: ", curOsGlobal
            print "relTargetOS: ", relTargetOs
#            curDif = 0.0
#            for k in self.target.relKeys:
#                curDif += self.target.weights[k]*np.linalg.norm(curOsGlobal[k]-self.target[k])
            curDif = self.target.score(curOsGlobal)    
            print "ScoreToBeat: ", curDif
            bestAction = self.findBestAction(copy.deepcopy(worldState), self.target, curDif, 0)

            
#            actions = {}
#            for dK, dV in difs.items():
#                actions[dK] = []
#                for ac in self.abstractCases.values():
#                    if dK in ac.variables:
#                        actions[dK].append(ac.getAction(givenInteraction, [dK], difs, weights=relTarget.weights))
#            for d, acts in actions.items():
#                for a in acts:
#                    intPrediction, ac = self.predictIntState(givenInteraction, a)
#                    osPrediction = intPrediction.getObjectState(self.target["name"])
#                    osPrediction.transform(state.transM, state.ori)   
#                    s = 0.0
#                    for k in self.target.relKeys:
#                        s += np.linalg.norm(osPrediction[k]-self.target[k])
#                    if s < worstDif:
#                        bestAction = a
#                        worstDif = s

                    
        elif isinstance(self.target, InteractionState):
            raise NotImplementedError        
        
        if bestAction != None:
#            if np.random.rand() < 0.1:
#                bestAction["mvDir"] *= 2
#            print "curScore: {}, bestScore: {}".format(curScore, bestScore)
#            if bestScore < curScore:
#                bestAction["mvDir"] *= -1
#                print "swapping movedir"
            print "BestAction: ", bestAction
            return bestAction
        else:
            return self.getRandomAction(worldState, BLOCK_BIAS)
            
    def findBestAction(self, worldState, target, scoreToBeat, depth):
        
        if depth > MAX_DEPTH:
            return None
        
        bestAction = None
        bestPrediction = None
        bestScore = 0.0
        bestDif = float('inf')
        givenInteraction = worldState.getInteractionState("gripper")
        relTarget = self.createRelativeTargetInteraction(worldState, target)
        difs = {}
        for k in relTarget.relKeys:
            difs[k] = relTarget[k] - givenInteraction[k]
            
        print "Difs: ", difs
        difSet = Set(difs.keys())
        
        actions = []
#             Problem: How to translate differences between target and given OS (e.g pos) 
#             into differences in relative interaction states???
        for ac in self.abstractCases.values():
            if ac.variables.issuperset(difSet):
                action = ac.getAction(givenInteraction, difSet, difs, weights=None)
                #if isApplicable(action)
                actions.append(action)
#                print "ac: {} selected action: {}".format(ac.variables, action)          
                
        

#        actions.extend(genericActions)
        
        for a in actions:
            intPrediction, ac = self.predictIntState(givenInteraction, a)
            osPrediction = intPrediction.getObjectState(self.target["name"])
#            print "Action: ", a
#            
            osPrediction.transform(worldState.transM, worldState.ori)
#            print "predictedOS: ", osPrediction
            s = target.score(osPrediction)
#            s = 0.0
#            for k in target.relKeys:
#                s += target.weights[k]* np.linalg.norm(osPrediction[k]-target[k])
            if s > bestScore:
                bestAction = a
                bestScore = s
                bestPrediction = intPrediction
            bestDif = bestScore
#            if s < bestDif:
#                bestAction = a
#                bestDif = s
#                bestPrediction = intPrediction
                
        print "bestDif at depth: {} is {}".format(depth, bestDif)#, bestAction)
        if bestDif - scoreToBeat > 0.01:
            return bestAction
        elif bestAction != None:
            ws = WorldState()
            ws.transM = np.copy(worldState.transM)
            ws.invTrans = np.copy(worldState.invTrans)
            ws.ori = np.copy(worldState.ori)
            ws.addInteractionState(bestPrediction)
            if self.findBestAction(ws, target, scoreToBeat, depth+1) != None:
                return bestAction
                
        return None
        
    def getRandomAction4(self,state,blockbias = 0):
        a = copy.deepcopy(np.random.choice(genericActions))
        print "getting Random action: ", a
        return a
            
            
    def getRandomAction2(self, state, blockbias = 0):
        a = Action()
        a["cmd"] = GZCMD["MOVE"]
        a["mvDir"] = np.ones(3)*0.01
        a["mvDir"][2] = 0
        rnd = np.random.rand()
        if rnd < 0.3:
            a["mvDir"][0] = -0.2
        elif rnd < 0.6:
            a["mvDir"][0] = 0.2
        else:
            a["mvDir"][1] = 0.2
        return a
            
    def getRandomAction3(self, state, blockbias = 0):
        gripperInt = state.getInteractionState("gripper")
        a = Action()
        a["cmd"] = GZCMD["MOVE"]
        a["mvDir"] = np.zeros(3)
        if gripperInt["spos"][0] < -0.2:
            a["mvDir"][0] = 0.5*np.random.rand()
        elif gripperInt["spos"][0] > 0.2:
            a["mvDir"][0] = -0.5*np.random.rand()
        else:
#            a["mvDir"][0] = np.random.rand()-0.5
            a["mvDir"][1] = 0.2
            if gripperInt["spos"][1] < 0:
                a["mvDir"][1] *= 1.0
            else:
                a["mvDir"][1] *= -1.0
#        else:
#            a["mvDir"][1] = np.random.rand()-0.5
        return a
        
            
    def getRandomAction(self, state, blockbias = 0):
        print "getting random action"
        rnd = np.random.rand()
        a = Action()
        if rnd < 0.7:
            a["cmd"] = GZCMD["MOVE"]
            if np.random.rand() < blockbias:
                gripperInt = state.getInteractionState("gripper")
                a["mvDir"] = (gripperInt["opos"]-gripperInt["spos"]) #+ (np.random.rand(3)-0.5)
            else:
#            a["dir"] = np.array([1.2,0,0])
                a["mvDir"] = np.random.rand(3)*2-1
        elif rnd < 0.8:
            a["cmd"] = GZCMD["MOVE"]
            a["mvDir"] = np.array([0,0,0])
        else:
#            a["cmd"] = GZCMD["NOTHING"]
            a["cmd"] = GZCMD["MOVE"]
#        a["mvDir"] *= 2.0
        a["mvDir"][2] = 0
        norm = np.linalg.norm(a["mvDir"])
        if norm > 0.25:
            a["mvDir"] /= 3*np.linalg.norm(a["mvDir"])
        return a
    
    def setTarget(self, postState = None):
        self.target = postState
    
    def getBestCase(self, state, action):
        
#        print "getBestCase with state: {} \n action: {}".format(state, action)
        bestCase = None
        if self.aCClassifier != None:
            x = [np.concatenate((state.toSelVec(),action.toSelVec()))]
#            print "X before scaling: ", x
            if self.scaler != None:
                x = self.scaler.transform(x)
#            print "X after sclaing: ", x
            caseID = int(self.aCClassifier.predict(x)[0])
#            print "CaseID: ", caseID
#            print "Case prob: ", self.aCClassifier.predict_proba(x)
            bestCase = self.abstractCases[caseID]
        else:
#            scoreList = [(c,c.score(state,action)) for c in self.abstractCases]
            scoreList = [(c.abstCase,c.score(state,action)) for c in self.cases]
            
    #        sortedList = sorted(self.abstractCases, key=methodcaller('score', state, action), reverse= True)
            sortedList = sorted(scoreList, key=itemgetter(1), reverse=True) 
    #        self.lastScorelist = [(s, sorted(c.variables), len(c.refCases)) for c,s in sortedList]
    #        if state["sid"] == 15:
#            print "ScoreList: ", [(s, sorted(c.variables), len(c.refCases)) for c,s in sortedList]
            if len(sortedList) > 0:
    #            if sortedList[0][1] == 0 and self.lastAC != None:
    #                bestCase = self.lastAC
    #            else:
                bestCase = sortedList[0][0]
                
        
        if isinstance(bestCase, AbstractCase):
#            print "selected AC: ", bestCase.variables
            if bestCase.variables == []:
                self.numZeroCase += 1

            return bestCase
        else:
            return None
    
    def predictIntState(self, state, action):
#        print "predict: ", state["sname"]
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
        
        
#        if isinstance(self.target, ObjectState):
#            resultingOS = result.getObjectState(self.target["name"])
#            prevOS = state.getObjectState(self.target["name"])
#            
            
#        if state["sid"] != 8:
#            raise TypeError("Wrong sID: ", state["sid"])
        newCase = BaseCase(state, action, result)
#        print "New case difs: ", newCase.dif
        attribSet = newCase.getSetOfAttribs()
#        predictionRating = result.rate(prediction)#, self.weights)
#        predictionScore = sum(predictionRating.values())
        predictionScore = result.score(prediction)
#        print "Correct attribSet for state: {} : {}".format(result["sname"], sorted(attribSet))
#        print "predictionScore: ", predictionScore
        abstractCase = None
        retrain = False
        for ac in self.abstractCases.values():
            if ac.variables == attribSet:
                abstractCase = ac
#                print "Correct AC_ID: ", abstractCase.id
                #TODO consider search for all of them in case we distinguish by certain features
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
            else:
                retrain = True
                
        if predictionScore < PREDICTIONTHRESHOLD:
            if abstractCase != None:
                    try:
                        abstractCase.addRef(newCase)
#                        retrain = True
                    except TypeError, e:
                        print "case was already present"
                    else:
                        self.addBaseCase(newCase)
                    
            else:
                #Create a new abstract case
#                print "new Abstract case: ", attribSet
                newAC = AbstractCase(newCase, len(self.abstractCases))
                self.abstractCases[newAC.id] = newAC
                self.addBaseCase(newCase)
                retrain = True
        if retrain:
            self.retrainACClassifier()


    def retrainACClassifier(self):
        print "Retraining!"
        if len(self.abstractCases) > 1:
            nFeature = np.size(np.concatenate((self.cases[0].preState.toSelVec(),self.cases[0].action.toSelVec())))
            X = np.zeros((len(self.cases),nFeature))
            Y = np.zeros(len(self.cases))
            for i in range(len(self.cases)):
                X[i,:] = np.concatenate((self.cases[i].preState.toSelVec(),self.cases[i].action.toSelVec()))
                Y[i] = self.cases[i].abstCase.id
#            self.scaler = preprocessing.StandardScaler(with_mean = False, with_std=True).fit(X)
#            self.scaler = preprocessing.MinMaxScaler().fit(X)
#            self.scaler = preprocessing.Normalizer().fit(X)
#            self.aCClassifier = svm.SVC(kernel='rbf', C=1, gamma=0.1)
#            self.aCClassifier = SGDClassifier(loss='log', penalty="l2")
            self.aCClassifier = tree.DecisionTreeClassifier(criterion="gini", class_weight='auto')#, min_samples_leaf=5) max_leaf_nodes=len(self.abstractCases))#, max_features='auto')
#            self.aCClassifier = RandomForestClassifier()
#            self.aCClassifier = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4), n_estimators=50)
#            self.aCClassifier.fit(self.scaler.transform(X),Y)
            self.aCClassifier.fit(X,Y)
            
    def getGraphViz(self, dot_data):
        if self.aCClassifier != None:
            tree.export_graphviz(self.aCClassifier, out_file=dot_data)

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
        