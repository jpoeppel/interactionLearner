#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:34:43 2015

@author: jpoeppel
"""

import numpy as np
from metrics import metrics2 as metrics
from common import GAZEBOCMDS as GZCMD
import copy

THRESHOLD = 0.1
NUMDEC = 5
MAXCASESCORE = 16
MAXSTATESCORE = 14
PREDICTIONTHRESHOLD = 0.1
PREDICTIONTHRESHOLD = MAXSTATESCORE - PREDICTIONTHRESHOLD

class State(dict):
    """
        Base class representing the state of something. Will be used to
        derive specific state classes from.
    """
    
    def __init__(self):
        pass
    
    def score(self, otherState):
        assert isinstance(otherState, State), "{} is not a State object".format(otherState)
        s = 0.0
        for k in self.relevantKeys():
            s += metrics[k](self[k], otherState[k]) #* weights[k] 
        return s
        
    def relevantKeys(self):
        return self.keys()
        
    def toVec(self):
        r = np.array([])
        for k in self.relevantKeys():
            if isinstance(self[k], np.ndarray):
                r = np.concatenate((r,self[k]))
            elif not isinstance(self[k], unicode):
                r = np.concatenate((r,[self[k]]))
        return r
        
#    def __repr__(self):
#        return str(self)
        
class ObjectState(State):
    """
        State class used to represent object states.
        Holds information about object position, orientation, type, id, name 
        and other properties.
    """
    
    def __init__(self):
        self.update({"id": -1, "name": "", "type": -1, "pos": np.zeros(3), 
                         "orientation": np.zeros(4), "linVel": np.zeros(3), 
                         "angVel": np.zeros(3)})
        
class InteractionState(State):
    
    def __init__(self, intId, o1):
        self.update({"intId": intId, "sid":o1["id"], "sname": o1["name"], 
                     "stype": o1["type"], "spos":o1["pos"], 
                     "sori": o1["orientation"], "slinVel": o1["linVel"], 
                     "sangVel": o1["angVel"], "dist": 0, "dir": np.zeros(3),
                     "contact": 0, "oid": -1, "oname": "", "otype": 0, 
                     "dori": np.zeros(4), "dlinVel": np.zeros(3), "dangVel":np.zeros(3)})
                     
    def relevantKeys(self):
        keys = self.keys()
        keys.remove("intId")
        keys.remove("sname")
        keys.remove("oname")
        return keys
                     
    def fill(self, o2):
        assert isinstance(o2, ObjectState), "{} (o2) is not an ObjectState!".format(o2)
        self["dist"] = np.linalg.norm(self["spos"]-o2["pos"])
        self["dir"] = o2["pos"]-self["spos"]
        self["oid"] = o2["id"]
        self["oname"] = o2["name"]
        self["otype"] = o2["type"]
        self["dori"] = o2["orientation"]-self["sori"] # TODO Fix THIS IS WRONG!!!
        self["dlinVel"] = o2["linVel"] - self["slinVel"]
        self["dangVel"] = o2["angVel"] - self["sangVel"]
    
class Action(State):
    
    def __init__(self, cmd=GZCMD["NOTHING"], direction=np.array([0.0,0.0,0.0])):
        self.update({"cmd":cmd, "mvDir": direction})
        
class WorldState(object):
    
    def __init__(self):
        self.objectStates = {}
        self.interactionStates = {}
        self.numIntStates = 0
        self.predictionCases = {}

    def addInteractionState(self, intState, usedCase = None):
#        print "adding interactionState: ", intState["intId"]
        assert isinstance(intState, InteractionState), "{} (intState) is not an InteractionState object.".format(intState)
        self.interactionStates[intState["intId"]] = intState
        self.numIntStates += 1        
        self.predictionCases[intState["intId"]] = usedCase
    
    def parseModels(self, models):
        for m in models:
            if m.name == "ground_plane" or "wall" in m.name or m.name == "gripperShadow":
                #Do not model the ground plane or walls for now
                continue
            else:
                tmp = ObjectState()
                tmp["pos"] = np.round(np.array([m.pose.position.x,m.pose.position.y,m.pose.position.z]), NUMDEC)
                tmp["orientation"]  = np.round(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                            m.pose.orientation.z,m.pose.orientation.w]), NUMDEC)
                tmp["linVel"] = np.round(np.array([m.linVel.x,m.linVel.y,m.linVel.z]), NUMDEC)
                tmp["angVel"] = np.round(np.array([m.angVel.x,m.angVel.y,m.angVel.z]), NUMDEC)
                tmp["name"] = m.name
                tmp["id"] = m.id
                tmp["type"] = m.type
                self.objectStates[m.id] = tmp
                
    def parseInteractions(self, ws):
        tmpList = self.objectStates.values()
        for o1 in self.objectStates.values():
            intState = InteractionState(self.numIntStates, o1)
            for o2 in tmpList:
                if np.array_equal(o1,o2):                    
                    intState.fill(o2)
                    self.addInteractionState(intState)
                
#    def parseContacts(self, contacts):
#        for c in contacts:
#            if self.objectStates.has_key(c.wrench[0].body_1_id):
#                self.objectStates[c.wrench[0].body_1_id].addContact(c.wrench[0].body_2_id)
    
    def parse(self, gzWS):
        self.parseModels(gzWS.model_v.models)
        self.parseInteractions(gzWS)
#        self.parseContacts(ws.contacts.contact)
    

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
        for k in pre.relevantKeys():
            self.dif[k] = post[k]-pre[k]
        
    def getListOfAttribs(self):
        """
            Returns the list of attributes that changed more than THRESHOLD.
        """
        r = []
        for k in self.dif.keys():
            if np.linalg.norm(self.dif[k]) > THRESHOLD:
                r.append(k)
        return r
        
        
    def getListOfConstants(self):
        #TODO make more efficient by storing these values
        r = []
        for k in self.dif.keys():
            if np.linalg.norm(self.dif[k]) <= THRESHOLD:
                r.append((k,self.preState[k]))
        return r
    
    def predict(self, state,action):
        return self.postState
        
class AbstractCase(object):
    
    def __init__(self, case):
        assert isinstance(case, BaseCase), "case is not a BaseCase object."
        self.refCases = []
        self.avgPrediction = 0.0
        self.numPredictions = 0
        self.name = ""
        self.variables = [] #List of attributes that changed 
        self.attribs = {} # Dictionary holding the attributs:[values,] pairs for the not changing attribs of the references
        self.predictors = {}
        self.addRef(case)
        
    def predict(self, state, action):
        if len(self.refCases) > 1:
            resultState = copy.deepcopy(state)
#            print "resultState intId: ", resultState["intId"]
            for k in self.variables:
                resultState[k] = state[k] + self.predictors[k].predict(np.concatenate((state.toVec(),action.toVec())))[0]
#            print "resultState intId after: ", resultState["intId"]
            return resultState
        else:
            prediction= self.refCases[0].predict(state,action)
            prediction["intId"] = state["intId"]
            return prediction
            
            
    def score(self, state, action):
        s = MAXCASESCORE
        for k in state.relevantKeys()+ action.relevantKeys():
            if self.attribs.has_key(k):
                bestScore = 1
                for v in self.attribs[k]:
                    tmpScore = metrics[k](state[k], v)
                    if tmpScore < bestScore:
                        bestScore = tmpScore
                s -= bestScore
        return s
    
    def updatePredictionScore(self, score):
        self.numPredictions += 1
        self.avgPrediction += (score-self.avgPrediction)/self.numPredictions
        
    def addRef(self, ref):
        for k,v in ref.getListOfConstants():
            if self.attribs.has_key(k):
                
                if any(np.array_equal(v,x) for x in self.attribs[k]):
                    self.attribs[k].append(v)
            else:
                self.attribs[k] = [v]
            
        self.refCases.append(ref)
        self.updatePredictors()
        
    def updatePredictors(self):
        if len(self.refCases) > 1:
            for k in self.variables:
                self.predictors[k] = GaussianProcess(corr='cubic')
                data, labels = self.getTrainingData(k)
                self.predictors[k].fit(data, labels)
                
    def getTrainingData(self, attrib):
        inputs = []
        outputs = []
        for c in self.refCases:
            inputs.append(np.concatenate((c.preState.toVec(),c.action.toVec())))
            outputs.append(c.postState[attrib]- c.preState[attrib])
        return inputs, outputs
    
class ModelCBR(object):
    
    def __init__(self):
#        self.cases = []
        self.abstractCases = []
        
    def getAction(self, state):
        pass
    
    def getBestCase(self, state, action):
        bestCase = None
        bestScore = 0.0
        for c in self.abstractCases:
            s = c.score(state, action)
            if s >= bestScore:
                bestCase = c
                bestScore = s
        return bestCase
    
    def predictIntState(self, state, action):
        bestCase = self.getBestCase(state, action)
        if bestCase != None:
            return bestCase.predict(state, action), bestCase
        else:
            print "using old state with id: ", state["intId"]
            return state, bestCase
    
    def predict(self, worldState, action):
        predictionWs = WorldState()
        for intState in worldState.interactionStates.values():
            
#            print "predicting for ", intState["intId"]
            prediction, usedCase = self.predictIntState(intState, action)
#            print "predicted intId: ", prediction["intId"]
            predictionWs.addInteractionState(prediction, usedCase)
#        print "resulting prediction: ", predictionWs.interactionStates
        return predictionWs
        
    def updateState(self, state, action, prediction, result, usedCase):
        newCase = BaseCase(state, action, result)
        attribList = newCase.getListOfAttribs()
        predictionScore = result.score(prediction)
        if usedCase != None and usedCase.variables == attribList:
            usedCase.updatePredictionScore(predictionScore)
        if predictionScore < PREDICTIONTHRESHOLD:
            abstractCase = None
            for ac in self.abstractCases:
                if ac.variables == attribList:
                    abstractCase = ac
                    #TODO consider search for all of them in case we distinguis by certain features
                    break
            if abstractCase != None:
                #If an abstract case is found add the reference
                abstractCase.addRef(newCase)
            else:
                #Create a new abstract case
                self.abstractCases.append(AbstractCase(newCase))
                
            
            
    
    def update(self, state, action, prediction, result):
        for intState in state.interactionStates.keys():
            if not prediction.interactionStates.has_key(intState):
                print "prediction: ", prediction.interactionStates
            if not result.interactionStates.has_key(intState):
                print "result: ", result.interactionStates
            self.updateState(state.interactionStates[intState], action, prediction.interactionStates[intState], 
                             result.interactionStates[intState], prediction.predictionCases[intState])
        