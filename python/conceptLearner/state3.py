#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:12:06 2015
Similar to state2, but worldstate will only consist out of 1 InteractionState!
@author: jpoeppel
"""

import common
from common import NUMDEC
from common import GAZEBOCMDS as GZCMD
from metrics import similarities
from metrics import differences
import numpy as np
import math
import copy
from operator import methodcaller, itemgetter
from state2 import State, InteractionState, Action
import state2

class ObjectState(state2.ObjectState):
    """
        State class used to represent object states.
        Holds information about object position, orientation, type, id, name 
        and other properties.
    """
            
    def fromInteractionState2(self, intState):
        super(state2.ObjectState, self).__init__()
        self.update({"id": intState["oid"], "name": intState["oname"], "pos":np.copy(intState["spos"]+intState["dir"]),
                     "euler": np.copy(intState["seuler"]+intState["deuler"]), "linVel":np.copy(intState["slinVel"]+intState["dlinVel"]),
                     "angVel": np.copy(intState["sangVel"]+intState["dangVel"])})
        if intState["contact"]:
            self["contact"] = intState["sname"]
            
class WorldState(object):
    
    def __init__(self, transM = None, invTrans = None, ori = None):
        self.objectStates = {}
        self.interactionStates = {}
        self.numIntStates = 0
        self.predictionCases = {}
        self.transM = transM
        self.invTrans = invTrans
        self.ori = ori

    def addInteractionState(self, intState, usedCase = None):
#        print "adding interactionState: ", intState["intId"]
        assert isinstance(intState, InteractionState), "{} (intState) is not an InteractionState object.".format(intState)
        self.interactionStates[intState["intId"]] = intState
        self.numIntStates += 1        
        self.predictionCases[intState["intId"]] = usedCase
    
    def parseModels(self, models):
        
        for m in models:
            if m.name == "ground_plane" or "wall" in m.name or "Shadow" in m.name:
                #Do not model the ground plane or walls for now
                continue
            else:
#                print "parsing: ", m.name
                tmp = ObjectState()
                tmp["pos"] = np.round(np.array([m.pose.position.x,m.pose.position.y,m.pose.position.z]), NUMDEC) #/ 2.0
#                print "parsing model: {}, pos: {}".format(m.name, tmp["pos"])
                tmp["euler"]  = np.round(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                            m.pose.orientation.z,m.pose.orientation.w])), NUMDEC)
                tmp["euler"][:2] = 0
#                tmp["euler"]  = common.quaternionToEuler(np.array([0.0,0.0,m.pose.orientation.z,m.pose.orientation.w]))
#                print "quaternion: {}, euler: {}".format(np.array([m.pose.orientation.x,m.pose.orientation.y,
#                                            m.pose.orientation.z,m.pose.orientation.w]), tmp["euler"])
                tmp["linVel"] = np.round(np.array([m.linVel.x,m.linVel.y,m.linVel.z]), NUMDEC)
                if np.linalg.norm(tmp["linVel"]) < 0.01:
                    tmp["linVel"] = np.array([0.0,0.0,0.0])
#                print "name: {}, linVel: {}".format(m.name, tmp["linVel"])
                tmp["angVel"] = np.round(np.array([m.angVel.x,m.angVel.y,m.angVel.z]), 1)
#                print "Raw AngVel: ", tmp["angVel"]
                tmp["name"] = m.name
                tmp["id"] = m.id
                tmp["type"] = m.type
                self.objectStates[m.name] = tmp
                
                if m.name == "blockA" and self.transM == None:
                    self.transM = common.eulerPosToTransformation(tmp["euler"],tmp["pos"])
                    self.invTrans = common.invertTransMatrix(self.transM)
#                    print "invTrans: ", self.invTrans
                    self.ori = np.copy(tmp["euler"])

                
    def parseInteractions(self):
        tmpList = self.objectStates.values()
        gripperO = None
        for o in tmpList:
            #Transform to local block coordinate system
            o.transform(self.invTrans, -self.ori)
            if o["name"] == "gripper":
                gripperO = o
        for o1 in self.objectStates.values():
            intState = InteractionState(self.numIntStates, gripperO)
            if not np.array_equal(o1,gripperO):             
                intState.fill(o1)
                self.addInteractionState(intState)
#                    

                
    def parseContacts(self, contacts):
        for c in contacts:
            o1Name = c.wrench[0].body_1_name.split(':')[0]
            o2Name = c.wrench[0].body_2_name.split(':')[0]
            if self.objectStates.has_key(o1Name):
                self.objectStates[o1Name]["contact"] = o2Name
            if self.objectStates.has_key(o2Name):
                self.objectStates[o2Name]["contact"] = o1Name
    
    def parse(self, gzWS):
        self.parseModels(gzWS.model_v.models)
        self.parseContacts(gzWS.contacts.contact)
        self.parseInteractions()
        
    def reset(self, worldState):
        self.objectStates = {}
        self.interactionStates = {}
        ws = copy.deepcopy(worldState)
        for intState in ws.interactionStates.values():
            tmp = ObjectState()
            tmp.fromInteractionState(intState)
            #Transform back to world coordinate system first
            tmp.transform(ws.transM, ws.ori)
            tmp2 = ObjectState()
            tmp2.fromInteractionState2(intState)
            tmp2.transform(ws.transM, ws.ori)
#            print "Tmp after back transformation: ", tmp
            if not self.objectStates.has_key(tmp["name"]):
                self.objectStates[tmp["name"]] = tmp
            if not self.objectStates.has_key(tmp2["name"]):
                self.objectStates[tmp2["name"]] = tmp2
            if tmp["name"] == "blockA":
                self.transM = common.eulerPosToTransformation(tmp["euler"],tmp["pos"])
                self.invTrans = common.invertTransMatrix(self.transM)
                self.ori = np.copy(tmp["euler"])
            if tmp2["name"] == "blockA":
                self.transM = common.eulerPosToTransformation(tmp2["euler"],tmp2["pos"])
                self.invTrans = common.invertTransMatrix(self.transM)
                self.ori = np.copy(tmp2["euler"])
        self.parseInteractions()
#        
#        print "InteractionStates: ", self.interactionStates.values()

    def getInteractionState(self, sname):
        for i in self.interactionStates.values():
            if i["sname"] == sname:
                return i
        return None    