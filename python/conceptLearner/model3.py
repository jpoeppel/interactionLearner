#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:20:05 2015
#TODO: Consider differentiation between gripper and other objects
@author: jpoeppel
"""

from metrics import metrics
import numpy as np
import copy

THRESHOLD = 0.1
NUMDEC = 5

class BaseCase(object):
    
    def __init__(self):
        self.preState
        self.postState
        self.action
        self.dif = {}
        
    def getListOfAttribs(self):
        """
            Returns the list of attributes that changed more than THRESHOLD.
        """
        r = []
        for k in self.dif.keys():
            if np.linalg.norm(self.dif[k]) > THRESHOLD:
                r.append(k)
        return r
        
class AbstractCase(object):
    
    def __init__(self):
        self.refCases = []
        self.avgPrediction = 0.0
        self.numPredictions = 0
        self.name = ""
        self.attribs = []
        self.predictors = {}
        
    def predict(self, state, action):
        pass
    
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
        for k in self.keys():
            s += metrics[k](self[k], otherState[k]) #* weights[k] 
        return s
        
    def toVec(self):
        r = np.array([])
        for v in self.values():
            if isinstance(v, np.ndarray):
                r = np.concatenate((r,v))
            elif isinstance(v, ObjectState):
                r = np.concatenate((r,v.toVec()))
            elif not isinstance(v, unicode):
                r = np.concatenate((r,[v]))
        return r
        
    def __repr__(self):
        return str(self.features)
    
class ObjectState(State):
    """
        State class used to represent object states.
        Holds information about object position, orientation, type, id, name 
        and other properties.
    """
    
    def __init__(self):
        self.update({"id": 0, "name": "", "type":0, "pos":np.zeros(3), 
                         "orientation": np.zeros(4), "linVel": np.zeros(3), 
                         "angVel": np.zeros(3)})
        
        
class InteractionState(State):
    """
        State class used to represent interaction states.
        Holds information about object pairs: Participants A and B, distance, 
        direction, contact yes/no etc.
    """
    
    def __init__(self, objA, objB):
        assert isinstance(objA, ObjectState), "{}(A) is not an ObjectState object".format(objA)
        assert isinstance(objB, ObjectState), "{}(B) is not an ObjectState object".format(objB)
        self.update({"id": "{:d}{:d}".format(objA["id"], objB["id"]), "A": objA, "B": objB, "dir": objB["pos"]-objA["pos"],
                         "dist": np.linalg.norm(objB["pos"]-objA["pos"]), "contact": 0})
        pass
    
class WorldState(object):
    """
        State class used to represent a current world situation.
        Holds object and interaction states for all objects and interactions.
    """
    
    def __init__(self):
        self.objectStates = {}
        self.interactionStates = {}
        pass
    
    def addObjectState(self, state):
        assert isinstance(state, ObjectState), "{} is not an ObjectState object".format(state)
        self.objectStates[state["id"]] = state

    def addInteractionState(self, state):
        assert isinstance(state, InteractionState), "{} is not an InteractionState object".format(state)
        self.interactionStates[state["id"]] = state   
        
    def parseModels(self, models):
        for m in models:
            if m.name == "ground_plane":
                #Do not model the ground plane for now
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
                self.addObjectState(tmp)
                
    def parseInteractions(self, ws):
        tmpList = self.objectStates.values()
        for o1 in self.objectStates.values():
            tmpList.remove(o1)
            for o2 in tmpList:
                intState = InteractionState(o1,o2)
                self.addInteractionState(intState)
        for c in ws.contacts.contact:
            idS = "{:d}{:d}".format(c.wrench[0].body_1_id,c.wrench[0].body_2_id)
            if self.interactionStates.has_key(idS):
                self.interactionStates[idS]["contact"] = 1
                
    
    def parse(self, ws):
        self.parseModels(ws.model_v.models)
        self.parseInteractions(ws)
    
class CBRModel(object):

    def __init__(self):
        self.cases = []
        pass
    
    def update(self, initial, action, prediction, result):
        pass
    
    def predict(self, state, action):
        resultWorldState = copy.deepcopy(state)
        for intS in resultWorldState.interactionStates.values():
            bestCase = self.findBestObjectCase(intS, action)
            if bestCase != None:
                rIntS = bestCase.predict(intS, action)
                resultWorldState.interactionStates[rIntS["id"]] = rIntS
        
            
        pass
    
        
if __name__ == "__main__":
    a = ObjectState()
    b = ObjectState()
    print a.score(b)
    print metrics["dir"](1,2)