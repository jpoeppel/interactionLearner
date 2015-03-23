# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:29:49 2015

@author: jpoeppel
"""

import numpy as np
#import gazeboInterface as gi
import math
import copy
from sklearn.gaussian_process import GaussianProcess

BESTCASESCORE = 8
BESTWORLDSCORE = 6
MARGIN = 0.5
           
def cosD(a, b):
    if np.linalg.norm(a) == 0:
        if np.linalg.norm(b) == 0:
            return 1
        else:
            return 0
    else:
        return abs(a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    
def expD(a,b):
    return math.exp(-(np.linalg.norm(a-b)))
    
def equalD(a,b):
    if a == b:
        return 1
    else:
        return 0
        
metrics = {"dir": cosD, "linVel": expD, "orientation": cosD, "pos": expD, 
           "angVel": expD, "name": equalD, "id": equalD, "cmd":equalD}

class WorldState(object):
    
    def __init__(self):
        self.gripperState = {}
        self.objectStates = []
     
    def score(self, state):
        s = 0.0
        for k in self.gripperState.keys():
            s += metrics[k](self.gripperState[k], state.gripperState[k])
        return s
        
    def toVec(self):
        r = np.array([])
        for v in self.gripperState.values():
            if isinstance(v, np.ndarray):
                r = np.concatenate((r,v))
            elif not isinstance(v, unicode):
                r = np.concatenate((r,[v]))
        return r #TODO objects!!!
        
    def parse(self, models):
        for m in models:
            if m.name == "ground_plane":
                continue
            else:
                tmpDict = {}
                tmpDict["pos"] = np.array(m.pose.position._fields.values())
                tmpDict["orientation"]  = np.array(m.pose.orientation._fields.values())
                tmpDict["linVel"] = np.array(m.linVel._fields.values())
                tmpDict["angVel"] = np.array(m.angVel._fields.values())
                tmpDict["name"] = m.name
                tmpDict["id"] = m.id
                if m.name == "gripper":
                    self.gripperState = tmpDict
                else:
                    self.objectStates.append(tmpDict)
    
class Action(dict):
    
    def score(self, action):
        s = 0.0
        for k in self.keys():
            s += metrics[k](self[k], action[k])
        return s
        
    def toVec(self):
        r = np.array([])
        for v in self.values():
            if isinstance(v, np.ndarray):
                r = np.concatenate((r,v))
            else:
                r = np.concatenate((r,[v]))
        return r
        
class predictionNode(object):
    def __init__(self):
        pass
    
    def predict(self, state, action):
        pass
    
    def score(self, state, action):
        pass
        
class Case(predictionNode):
    
    def __init__(self, pre, post, action):
        self.preState = pre
        self.postState = post
        self.action = action
        self.abstractCase = None
        
    def score(self, state, action):
        return self.preState.score(state) + self.action.score(action)
        
    def predict(self, state, action):
        return self.postState
        
class AbstractCase(predictionNode):
    
    def __init__(self, caseA, caseB):
        self.gripperAttribs = []
        self.predictors = {}  
        self.refCases = []
        self.refCases.append(caseA)
        self.refCases.append(caseB)
        caseA.abstractCase = self
        caseB.abstractCase = self
        for k in caseA.preState.gripperState.keys():
            if not (np.array_equal(caseA.preState.gripperState[k], caseA.postState.gripperState[k]) and 
                np.array_equal(caseB.preState.gripperState[k], caseB.postState.gripperState[k])):
                    self.gripperAttribs.append(k)
        self.updatePredictions()            
        
    def score(self, state, action):
        raise(NotImplementedError("AbstractCase.score should not be called"))
        pass
        
        
    def predict(self, state, action):
        resultState = copy.deepcopy(state)
        for k in self.gripperAttribs:
            resultState.gripperState[k] = self.predictors[k].predict(np.concatenate((state.toVec(),action.toVec())))
        return resultState
        
    def updatePredictions(self):
        for k in self.gripperAttribs:
            self.predictors[k] = GaussianProcess(corr='cubic')
            data, labels = self.getTrainingData(k)
#            print "data: " + str(data)
#            print "labels: " + str(labels)
            self.predictors[k].fit(data, labels)
        pass
    
    def getTrainingData(self, attrib):
        inputs = []
        outputs = []
        for c in self.refCases:
            inputs.append(np.concatenate((c.preState.toVec(),c.action.toVec())))
            outputs.append(c.postState.gripperState[attrib])
        return inputs, outputs
        
class ModelCBR(object):
    
    def __init__(self):
        self.cases = []
        self.abstractCases = []
        
    def searchBestCase(self, state, action):
        bestCase = None
        bestScore = 0.0
        for c in self.cases:
            tmpScore = c.score(state, action)
            if tmpScore > bestScore:
                bestScore = tmpScore
                bestCase = c
        print "best score: " + str(bestScore)
        if bestCase != None and bestCase.abstractCase != None and bestScore < BESTCASESCORE:
            print "using abstract case."
            return bestCase.abstractCase
        else:
            return bestCase
    
    def buildFeatures(self, state,action):
        return state.gripperState.values() + action.values()
        
    def predict(self, state, action):
        similarCase = self.searchBestCase(state, action)
        if similarCase != None:
            return similarCase.predict(state,action), similarCase
        else:
            return state, similarCase
    
    def update(self, state, action, prediction, result, usedCase):
        predictionScore = prediction.score(result)
        print "prediction score is: " + str(predictionScore)
        if predictionScore < BESTWORLDSCORE - MARGIN:
            newCase = Case(state, result, action)
#            print type(usedCase)
            if isinstance(usedCase, AbstractCase):
                #Prediction was bad, add new case and retrain predictors
                print "is abstract"
                attribList = []
                for k in state.gripperState.keys():
                    if not (np.array_equal(state.gripperState[k], result.gripperState[k])):
                        attribList.append(k)
                print "usedCased attribts: "+ str(usedCase.gripperAttribs)
                print "attribList: " + str(attribList)
                if attribList == usedCase.gripperAttribs:
                    print "update refCases"
                    usedCase.refCases.append(newCase)
                    newCase.abstractCase = usedCase
                    usedCase.updatePredictions()
            else:
                if usedCase != None and usedCase.abstractCase == None:
                    #Create abstractCase
                    self.abstractCases.append(AbstractCase(usedCase, newCase))
            self.cases.append(newCase)
                
        pass