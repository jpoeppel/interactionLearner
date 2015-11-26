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
from metrics import similarities

#BESTCASESCORE = 8
BESTWORLDSCORE = 5
MARGIN = 0.5
NUMDEC = 5
#PREDICTIONTHRESHOLD = BESTWORLDSCORE - MARGIN
PREDICTIONTHRESHOLD = 0.95
BESTCASESCORE = 3

class WorldState(object):
    """
        WorldState object representing all the objects in the world at a
        given moment.
    """
    
    def __init__(self):
        self.gripperState = {}
        self.objectStates = []
     
    def score(self, state):
        s = 0.0
        for k in self.gripperState.keys():
            s += similarities[k](self.gripperState[k], state.gripperState[k])
        s = similarities["pos"](self.gripperState["pos"], state.gripperState["pos"])
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
                tmpDict["pos"] = np.round(np.array([m.pose.position.x,m.pose.position.y,m.pose.position.z]), NUMDEC)
                tmpDict["orientation"]  = np.round(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                            m.pose.orientation.z,m.pose.orientation.w]), NUMDEC)
                tmpDict["linVel"] = np.round(np.array([m.linVel.x,m.linVel.y,m.linVel.z]), NUMDEC)
                tmpDict["angVel"] = np.round(np.array([m.angVel.x,m.angVel.y,m.angVel.z]), NUMDEC)
                tmpDict["name"] = m.name
                tmpDict["id"] = m.id
#                print "object: {}, type: {}".format(m.name, m.type)
                if m.name == "gripper":
                    self.gripperState = tmpDict
                else:
                    self.objectStates.append(tmpDict)
                    

    def __repr__(self):
        return str(self.gripperState)
    
class Action(dict):
    
    def __init__(self, cmd=3, direction=np.array([0.0,0.0,0.0])):
        self["cmd"] = cmd
        self["dir"] = direction
    
    def score(self, action):
        s = 0.0
        for k in self.keys():
            s += similarities[k](self[k], action[k])
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
        
    def getInterestingGripperAttribs(self):
        r = []
        for k in self.preState.gripperState.keys():
            if not np.array_equal(self.preState.gripperState[k], self.postState.gripperState[k]):
                r.append(k)
        return r
        
class AbstractCase(predictionNode):
        
    def __init__(self, caseA, caseB=None):
        #self.fixedAttribs= {} # Attributes for which this case was trained?
        self.gripperAttribs = []
        self.predictors = {}  
        self.refCases = []
        self.refCases.append(caseA)   
        self.gripperAttribs = caseA.getInterestingGripperAttribs()
        if len(self.gripperAttribs) == 0:
            raise(ValueError("No change in any attribute."))
        caseA.abstractCase = self
        if caseB != None:
            if self.gripperAttribs == caseB.getInterestingGripperAttribs():
                self.refCases.append(caseB)
                caseB.abstractCase = self
            else:
                print "WARNING caseB will be ignored because of different gripperAttribList!"
            
        
        self.updatePredictions()            
        
    def score(self, state, action):
        raise(NotImplementedError("AbstractCase.score should not be called"))
        pass
        
        
    def predict(self, state, action):
        if len(self.refCases) > 1:
            resultState = copy.deepcopy(state)
            for k in self.gripperAttribs:
                resultState.gripperState[k] = state.gripperState[k] + self.predictors[k].predict(np.concatenate((state.toVec(),action.toVec())))[0]
            return resultState
        else:
            return self.refCases[0].predict(state,action)
            
    def addRef(self, case):
        case.abstractCase = self
        self.refCases.append(case)
        self.updatePredictions()
        
    def updatePredictions(self):
        if len(self.refCases) > 1:
            for k in self.gripperAttribs:
                self.predictors[k] = GaussianProcess(corr='cubic')
                data, labels = self.getTrainingData(k)
#                print "data: " + str(data)
#                print "labels: " + str(labels)
                self.predictors[k].fit(data, labels)
        pass
    
    def getTrainingData(self, attrib):
        inputs = []
        outputs = []
        for c in self.refCases:
            inputs.append(np.concatenate((c.preState.toVec(),c.action.toVec())))
            outputs.append(c.postState.gripperState[attrib]- c.preState.gripperState[attrib])
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
        if bestScore == BESTCASESCORE:
            self.printUpdate = True
        else:
            self.printUpdate = False
        if bestCase != None and bestCase.abstractCase != None and bestScore < BESTCASESCORE:
#            print "using abstract case."
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
        """
            Function to update the model according to the last case.
            
        """
        predictionScore = result.score(prediction)
        print "prediction score is: " + str(predictionScore)
        if predictionScore < PREDICTIONTHRESHOLD:
            #Prediction was not good enough -> improve
            newCase = Case(state, result, action)
            attribList = newCase.getInterestingGripperAttribs()
            print "appending new case"
            self.cases.append(newCase)
            #Search for abstract Case with the same attribList:
            abstractCase = None
            for ac in self.abstractCases:
                if ac.gripperAttribs == attribList:
                    abstractCase = ac
                    #TODO consider search for all of them in case we distinguis by certain features
                    break
            if abstractCase != None:
                #If an abstract case is found add the reference
                abstractCase.addRef(newCase)
            else:
                #Create a new abstract case
                if isinstance(usedCase, Case) and usedCase != None and \
                    usedCase.getInterestingGripperAttribs() == attribList:
                        #Combine usedCase and newCase
                    try:
                        self.abstractCases.append(AbstractCase(newCase, usedCase))
                    except ValueError:
                        #Simply do nothing in this case
                        pass
                else:
                    #Create a new abstractCase only based on the newCase
                    try:
                        self.abstractCases.append(AbstractCase(newCase))
                    except ValueError:
                        #Simply do nothing in this case
                        pass

    
    def update2(self, state, action, prediction, result, usedCase):
        predictionScore = prediction.score(result)
        print "prediction score is: " + str(predictionScore)
        if predictionScore < BESTWORLDSCORE - MARGIN:
            if self.printUpdate:
                print "prediction: " + str(prediction)
                print "result: " + str(result)
                print "action: " + str(action)
                print "usedCase action: " + str(usedCase.action)
            newCase = Case(state, result, action)
            attribList = newCase.getInterestingGripperAttribs()
            if len(attribList) != 0:
    #            for k in state.gripperState.keys():
    #                if not (np.array_equal(state.gripperState[k], result.gripperState[k])):
    #                    attribList.append(k)
    #            print type(usedCase)
                if isinstance(usedCase, AbstractCase):
                    #Prediction was bad, add new case and retrain predictors
    #                print "is abstract"
    #                print "usedCased attribts: "+ str(usedCase.gripperAttribs)
    #                print "attribList: " + str(attribList)
                    if attribList == usedCase.gripperAttribs:
    #                    print "update refCases"
                        usedCase.refCases.append(newCase)
                        newCase.abstractCase = usedCase
                        usedCase.updatePredictions()
                    else:
                        self.abstractCases.append(AbstractCase(newCase))
                else:
                    if usedCase != None and usedCase.abstractCase == None:
                        #Search for applicable abstractCase
                        abstractCase = None
                        for ac in self.abstractCases:
                            if ac.gripperAttribs == attribList:
    #                            print "useable abstract case found"
                                abstractCase = ac
                                break
                        if abstractCase != None:
                            abstractCase.refCases.append(newCase)
                            newCase.abstractCase = abstractCase
                            abstractCase.updatePredictions()
                        else:
                            #Create abstractCase
                            if usedCase.getInterestingGripperAttribs() == attribList:
                                self.abstractCases.append(AbstractCase(usedCase, newCase))
                            else:
                                if len(usedCase.getInterestingGripperAttribs()) != 0:
                                    self.abstractCases.append(AbstractCase(usedCase))
                                self.abstractCases.append(AbstractCase(newCase))
            self.cases.append(newCase)
                
        pass