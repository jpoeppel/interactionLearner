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
    
    def __init__(self, pre, action, post):
        assert isinstance(pre, State), "{} is not a State object.".format(pre)
        assert isinstance(post, State), "{} is not a State object.".format(post)
        assert isinstance(action, Action), "{} is not an Action object.".format(action)
        assert (pre.keys()==post.keys()), "Pre and post states have different keys: {}, {}.".format(pre.keys(), post.keys())
        self.preState = pre
        self.postState = posr
        self.action = action
        self.dif = {}
        for k in pre.keys():
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
        
class AbstractCase(object):
    
    def __init__(self):
        self.refCases = []
        self.avgPrediction = 0.0
        self.numPredictions = 0
        self.name = ""
        self.attribs = [] #List of attributes that changed 
        self.constants = {} # Dictionary holding the attributs:values whose values are important
        self.predictors = {}
        
    def predict(self, state, action):
        pass
    
class Action(dict):
    
    def __init__(self):
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
        self.update({"id": -1, "name": "", "type": -1, "pos": np.zeros(3), 
                         "orientation": np.zeros(4), "linVel": np.zeros(3), 
                         "angVel": np.zeros(3)})
        
#    def addContact(self, otherId):
#        self["contacts"].append(otherId)
#        
#    def addInteraction(self, isId):
#        self["interaction"].append(isId)
        
class InteractionState(State):
    """
        State class used to represent interaction states.
        Holds information about object pairs: Participants A and B, distance, 
        direction, contact yes/no etc.
    """
    
    def __init__(self, isId, objA, objB=ObjectState()):
        assert isinstance(objA, ObjectState), "{}(A) is not an ObjectState object".format(objA)
        assert isinstance(objB, ObjectState), "{}(B) is not an ObjectState object".format(objB)
        self.update({"id": "{:d}{:d}".format(objA["id"], objB["id"]), "self": objA, "other": objB, "dir": objB["pos"]-objA["pos"],
                         "dist": np.linalg.norm(objB["pos"]-objA["pos"]), "contact": 0})
        objA.addInteraction(self["id"])
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
            if m.name == "ground_plane" or "wall" in m.name:
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
                
#    def parseContacts(self, contacts):
#        for c in contacts:
#            if self.objectStates.has_key(c.wrench[0].body_1_id):
#                self.objectStates[c.wrench[0].body_1_id].addContact(c.wrench[0].body_2_id)
    
    def parse(self, ws):
        self.parseModels(ws.model_v.models)
        self.parseInteractions(ws)
#        self.parseContacts(ws.contacts.contact)
    
class CBRModel(object):

    def __init__(self):
        self.abstractCases = []
        pass
    
    def getAction(self):
        pass
    
    def updateCase(self, initial, action, prediction, result, usedCase):
        predictionScore = result.score(prediction)
        newCase = BaseCase(initial, action, result)
        
    
    def update(self, initial, action, prediction, result):
        for intState in initial.interactionStates.values():
            updateCase(intState, action, prediction.interactionStates[intState.id], result.interactionStates[intState.id])
            
    
    def predict(self, state, action):
        """
            Predicts each interaction state for the next timestep based on the current
            world state and the current action.
        """
        resultWorldState = copy.deepcopy(state)
        for intState in resultWorldState.interactionStates.values():
            bestIntCase = self.getBestInteractionCase(intState, action)
            if bestIntCase != None:
                resultWorldState[intState.id] = bestIntCase.predict(intState,action)
                        
        return resultWorldState
    
        
if __name__ == "__main__":
    a = ObjectState()
    b = ObjectState()
    print a.score(b)
    print metrics["dir"](1,2)