# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:22:29 2015
Try to implement a top-down model...
@author: jpoeppel
"""

from model4 import State, ObjectState, InteractionState, WorldState, BaseCase, Action
from model4 import THRESHOLD, MAXCASESCORE, MAXSTATESCORE, PREDICTIONTHRESHOLD, TARGETTHRESHOLD
from common import GAZEBOCMDS as GZCMD
import model4
import numpy as np
from operator import methodcaller
from sets import Set
import copy
from topoMaps import ITM
from operator import methodcaller, itemgetter
from network import Node
from network import Tree

from metrics import similarities


class AbstractCase(model4.AbstractCase):
    
    def __init__(self, case):
        assert isinstance(case, BaseCase), "case is not a BaseCase object."
        self.id = 0
        self.refCases = []
        self.avgPrediction = 0.0
        self.numPredictions = 0
        self.name = ""
        self.variables = Set() #List of attributes that changed 
        self.variables.update(case.getSetOfAttribs())
        self.constants = {}
        self.gaussians = {}
        self.weights= {}
        self.minima = {}
        self.maxima = {}
        for k in self.variables:
            self.weights[k] = 1.0
        self.addRef(case)
    
    def predict(self, state, action):
        return self.weights
    
    def updateWeights(self, prediction, result):
        
        for k in self.variables:
            error = result[k] - prediction[k]
            self.weights[k] += 0.05 * error
            
#        print "AC: {}, weights: {}".format(self.variables, self.weights)

    def addRef(self, ref):
        
        if ref in self.refCases:
            raise TypeError("ref already in refCases")
        constChanged = False
#        
#        for k,v in ref.getListOfConstants() + ref.action.relevantItems():
        for k,v in ref.preState.relevantItems() + ref.action.relevantItems():
            if self.constants.has_key(k):
                if np.linalg.norm(v-self.constants[k]) > 0.001:
                    del self.constants[k]
                    constsChanged = True
            else:
                if len(self.refCases) == 0:
                    self.constants[k] = v
            if not self.minima.has_key(k) or np.all(v < self.minima[k]):
                self.minima[k] = v
            if not self.maxima.has_key(k) or np.all(v > self.maxima[k]):
                self.maxima[k] = v
        
        self.refCases.append(ref)
        ref.abstCase = self
        self.updateGaussians(self.gaussians, len(self.refCases), ref)     
        print "addRef has constChanged: ", constChanged
        return constChanged

class ModelCBR(object):
    
    def __init__(self):
        self.predictors = {}
        self.abstractCases = {}
        self.cases = []
        self.numACs = 0
        self.numPredictions = 0
        self.numCorrectCase = 0
        self.aCTree = Tree()
        #TODO try building a decision tree for AC selection
        
    def getBestCase(self, state, action):
#        
#        values = dict(state.relevantItems()+action.relevantItems())
#        bestCases = self.aCTree.getElements(values)
#        
#        bestCase = None
#        if bestCases != None:
#            print "len pockets: ", len(bestCases)
#            print "consts of all elements in pocket: ", [c.constants for c in bestCases]
#            scoreList = [(c,c.score(state,action)) for c in bestCases]
#            sortedList = sorted(scoreList, key=itemgetter(1), reverse=True) 
#            if len(sortedList) > 0:
#                bestCase = sortedList[0][0]
        bestCase = None
        scoreList = [(c,c.score(state,action)) for c in self.abstractCases.values()]
#        scoreList = [(c.abstCase,c.score(state,action)) for c in self.cases]
        
#        sortedList = sorted(self.abstractCases, key=methodcaller('score', state, action), reverse= True)
        sortedList = sorted(scoreList, key=itemgetter(1), reverse=True) 
        print "ScoreList: ", [(s, sorted(c.variables), len(c.refCases)) for c,s in sortedList]
        if len(sortedList) > 0:
            bestCase = sortedList[0][0]
        if isinstance(bestCase, AbstractCase):
#            print "bestCase variables: ", bestCase.variables
            return bestCase
        elif isinstance(bestCase, BaseCase):
#            print "bestCase variables: ", bestCase.abstCase.variables
            return bestCase.abstCase
        else:
            return None
            
    def predictIntState(self, state, action):
        
        bestCase = self.getBestCase(state, action)
        resultState = copy.deepcopy(state)
        if bestCase != None:
            weightDic = bestCase.predict(state, action)
            for k in weightDic:
                prediction = self.predictors[k].predict(np.concatenate((state.toVec(),action.toVec())))
                if prediction != None:
                    resultState[k] = state[k] + weightDic[k] * prediction
                else:
                    resultState[k] = state[k] + bestCase.refCases[0].predict(state, action, k)

        return resultState, bestCase
    
    def predict(self, worldState, action):
        
        predictionWs = WorldState()
        for intState in worldState.interactionStates.values():
            self.numPredictions += 1
#            print "predicting for ", intState["intId"]
            prediction, usedCase = self.predictIntState(intState, action)
#            print "predicted intId: ", prediction["intId"]
            predictionWs.addInteractionState(prediction, usedCase)
#        print "resulting prediction: ", predictionWs.interactionStates
        return predictionWs
        
    def updateState(self, state, action, prediction, result, usedCase):
        newCase = BaseCase(state, action, result)
        attribSet = newCase.getSetOfAttribs()
        for k in attribSet:
            if not self.predictors.has_key(k):
                self.predictors[k] = ITM()
            if similarities[k](result[k], prediction[k]) < 0.95:
                self.predictors[k].train(self.toNode(state,action,result, k))
                
        abstractCase = None
        for ac in self.abstractCases.values():
            if ac.variables == attribSet:
                abstractCase = ac
                #TODO consider search for all of them in case we distinguis by certain features
                break
        predictionScore = result.score(prediction)
        print "Prediction score: ", predictionScore
        if abstractCase != None:
            if predictionScore < PREDICTIONTHRESHOLD:
                abstractCase.updateWeights(prediction, result)
                
            if usedCase != None:
                if usedCase.variables != attribSet:
                    constChanged = False
                    #TODO improve scoring
                    try:
                        print "ADDING REF"
                        constChanged = abstractCase.addRef(newCase)
                        print "adding new ref to AC: {}, new constants: {}".format(abstractCase.variables, abstractCase.constants)
                    except TypeError:
                        print "Case already present."
                    else:
                        self.cases.append(newCase)
                        
                    print "has constChanged? ", constChanged
                    if constChanged:
                        print "updating Tree"
#                        self.aCTree = Tree()
#                        self.aCTree.addElements(self.abstractCases.values())                        
#                        self.aCTree.removeElement(abstractCase)
#                        self.aCTree.addElement(abstractCase, abstractCase.constants, abstractCase.minima, abstractCase.maxima)
                    pass      
                else:
                    print "Correct case was selected!!!!"
                    self.numCorrectCase += 1
            
        else:
            print "Create new AC: ", attribSet
            abstractCase = AbstractCase(newCase)
            abstractCase.id = self.numACs
            self.cases.append(newCase)
            self.numACs += 1
            self.abstractCases[abstractCase.id] = abstractCase
#            self.aCTree = Tree()
#            self.aCTree.addElements(self.abstractCases.values())
#            
        

#        s = "\n".join([str(ac.variables) + ", " + str(ac.minima) + ", " + str(ac.maxima) for ac in self.abstractCases.values()])
#            
#        with open("ACs.txt", 'a') as f:
#            f.write("_____________________________________\n")
#            f.write(s)
#                

            
    def toNode(self, state, action, result, attrib):
        node = Node(0, wIn=state.toVec(), action=action.toVec(),
                    wOut=result[attrib]-state[attrib])
        return node
            
    def update(self, state, action, prediction, result):
        for intState in state.interactionStates.keys():
            self.updateState(state.interactionStates[intState], action, prediction.interactionStates[intState], 
                             result.interactionStates[intState], prediction.predictionCases[intState])