#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:00:15 2015
Try connected ACs with transition probabilities to hopefully increase selection rate
Update: Simply counting relative transitions does not improve selection rate.
@author: jpoeppel
"""

from model4 import State, ObjectState, InteractionState, WorldState, BaseCase, AbstractCase, Action
from model4 import THRESHOLD, NUMDEC, MAXCASESCORE, MAXSTATESCORE, PREDICTIONTHRESHOLD, TARGETTHRESHOLD
from common import GAZEBOCMDS as GZCMD
import numpy as np
from operator import methodcaller
from sets import Set


class ModelCBR(object):
    
    def __init__(self):
        self.cases = []
        self.abstractCases = {}
        self.numAbstractCases = 0
        self.lastAc = None
        self.acTransistions = {}
        self.transCounter = {}
        self.numPredictions= 0
        self.numCorrectCase= 0
        
    def getBestCase(self, state, action):
#        bestCases = {}
#        bestScores = {}
        bestCase = None
        bestScore = 0.0
#        print "acTransitions: ", self.acTransistions
#        for acId, ac in self.abstractCases.items():
        
        sortedList = sorted(self.abstractCases.values(), key=methodcaller('score', state, action), reverse= True)
        if len(sortedList) > 0:
            bestCase = sortedList[0]

        
        if bestCase != None:
            print "bestCase ID: {}, {} ".format(bestCase.id, bestCase.variables)
#            if bestCase2 != None:
#                print "bestCase2 ID: {}, {} ".format(bestCase2.id, bestCase2.variables)
            return bestCase
        else:
            print "return None"
            return None
        
    def predictIntState(self, state, action):
        
        bestCase = self.getBestCase(state, action)
        
        if bestCase != None:
            return bestCase.predict(state, action), bestCase
        else:
            return state, None
    
    def predict(self, worldState, action):
        
        predictionWs = WorldState()
        for intState in worldState.interactionStates.values():
            self.numPredictions += 1
            prediction, usedCase = self.predictIntState(intState, action)
            predictionWs.addInteractionState(prediction, usedCase)
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
        attribSet = newCase.getSetOfAttribs()
#        predictionScore = sum(result.score(prediction).values())
        predictionScore = result.score(prediction)
        if usedCase != None and usedCase.variables == attribSet:
            self.numCorrectCase += 1
            print "correct AC with predictionscore: ", predictionScore
            if predictionScore < PREDICTIONTHRESHOLD:
                try:
                    usedCase.addRef(newCase)
                except Exception, e:
                    print "case was already present"
                else:
                    self.cases.append(newCase)
        
        
        abstractCase = None        
        for acId, ac in self.abstractCases.items():
            if ac.variables == attribSet:
                abstractCase = ac
                #TODO consider search for all of them in case we distinguis by certain features
                break
        if abstractCase == None:
            #Create a new abstract case
            abstractCase = AbstractCase(newCase)
            abstractCase.id = self.numAbstractCases
            self.numAbstractCases += 1
            self.abstractCases[abstractCase.id] = abstractCase
            self.cases.append(newCase)

        self.updateTransitions(self.lastAc, abstractCase.id)
        self.lastAc = abstractCase.id
        
            
            
    def updateTransitions(self, oldAc, newAc):
        if oldAc == None:
            self.transCounter[newAc] = {newAc: 0}
        else:
            if not self.transCounter.has_key(newAc):
                for k in self.transCounter.keys():
                    self.transCounter[k][newAc] = 0
                self.transCounter[newAc] = {}
                for k in self.transCounter.keys():
                    self.transCounter[newAc][k] = 0
            self.transCounter[oldAc][newAc] += 1

                    
                
            
        for fk in self.transCounter.keys():
            self.acTransistions[fk] = {}
            for tk in self.transCounter[fk].keys():
                s = sum(self.transCounter[fk].values())
                if s > 0.0:
                    self.acTransistions[fk][tk] = float(self.transCounter[fk][tk])/s
                else:
                    self.acTransistions[fk][tk] = 1.0/len(self.transCounter[fk])
            
            
    

    def update(self, state, action, prediction, result):
        for intState in state.interactionStates.keys():
            if not prediction.interactionStates.has_key(intState):
                print "prediction: ", prediction.interactionStates
            if not result.interactionStates.has_key(intState):
                print "result: ", result.interactionStates
            if not prediction.predictionCases.has_key(intState):
                print "prediction.predictionCases: ", prediction.predictionCases
            self.updateState(state.interactionStates[intState], action, prediction.interactionStates[intState], 
                             result.interactionStates[intState], prediction.predictionCases[intState])


    def getAction(self, state):
        return self.getRandomAction()
        
    def getRandomAction(self):
        print "getting random action"
        rnd = np.random.rand()
        a = Action()
        if rnd < 0.3:
            a["cmd"] = GZCMD["MOVE"]
            a["mvDir"] = np.random.rand(3)*2-1
        elif rnd < 0.4:
            a["cmd"] = GZCMD["MOVE"]
            a["mvDir"] = np.array([0,0,0])
        else:
            a["cmd"] = GZCMD["NOTHING"]
        a["mvDir"][2] = 0
        return a
        

