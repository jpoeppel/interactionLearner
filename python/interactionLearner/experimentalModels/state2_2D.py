#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:39:06 2015
Try a minimal state, with only 2D postion, 1D euler
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

class State(dict):
    """
        Base class representing the state of something. Will be used to
        derive specific state classes from.
    """
    
    def __init__(self):
        self.relKeys = self.keys()
        self.weights = {}
        for k in self.relKeys:
            self.weights[k] = 1.0
        pass
    
    def score(self, otherState):#, gWeights):
        assert isinstance(otherState, State), "{} is not a State object".format(otherState)
#        s = MAXSTATESCORE
        s = 0.0
#        s = {}
        for k in self.relKeys:
#            if gWeights.has_key(k):
#                w = gWeights[k]
#            else:
#                w = 1.0/MAXCASESCORE
#            s -= self.weights[k]*differences[k](self[k], otherState[k]) 
            s += self.weights[k]* similarities[k](self[k], otherState[k]) #* w
#            s[k] = self.weights[k]* similarities[k](self[k], otherState[k])
        return s
        
    def score2(self, otherState):
        a = 0.0
        for k,v in self.relevantItems():
            if hasattr(v, "__len__"):
                norm = len(v)
            else:
                norm = 1.0
            a += math.exp(-1.0/float(norm) * np.linalg.norm(v-otherState[k])**2)#*1.0/variances[k]
        return a
        
    def rate(self, otherState):
        s = {}
        for k in self.relKeys:
            s[k] = similarities[k](self[k], otherState[k])
        return s
        
    def relevantKeys(self):
        return self.relKeys
        
    def relevantItems(self):
        r = []
        for k in self.relKeys:
            r.append((k,self[k]))
        return r
        
    def toVec(self, const = {}):
        r = np.array([])
        keyOrder = []
        for k in self.relKeys:
            keyOrder.append(k)
            if k not in const.keys():
#            if k != "spos":
                if isinstance(self[k], np.ndarray):
                    r = np.concatenate((r,self[k]))
                elif not isinstance(self[k], unicode):
                    r = np.concatenate((r,[self[k]]))
#        print "KeyOrder: ", keyOrder
        return r
        
    def updateWeights(self, curState):
        print "updating weights"
        minAttrib = None
        minDif = float('inf')
        maxAttrib = None
        maxDif = 0.0
        for k in self.relKeys:
            d = np.linalg.norm(self[k] - curState[k])
            if d < minDif:
                minAttrib = k
                minDif = d
            if d > maxDif:
                maxAttrib = k
                maxDif = d
        print "maxAttrib: ", maxAttrib
        print "minAttrib: ", minAttrib
        if maxAttrib != None:
            if minAttrib != maxAttrib:
                self.weights[minAttrib] /= 2.0
                
            self.weights[maxAttrib] *= 2
        
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
            
        for k, v in self.relevantItems():
            if np.linalg.norm(v-other[k]) > 0.001:
                return False
        
        return True
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __repr__(self):
        s = "\n"
        for k,v in sorted(self.items(), key=itemgetter(0)):
            s+= "{}: {} \n".format(k,v)
        return s

class ObjectState(State):
    """
        State class used to represent object states.
        Holds information about object position, orientation, type, id, name 
        and other properties.
    """
    
    def __init__(self):
        State.__init__(self)
        self.update({"id": -1, "name": "", "type": -1, "pos": np.zeros(2), 
                         "euler": 0, "linVel": np.zeros(2), 
                         "angVel": 0, "contact": None})
        self.relKeys = self.keys()    
                          
    def transform(self, matrix, euler):
#        print "calling transform for: ", self["name"]
        tmpPos = np.matrix(np.concatenate((self["pos"],[1])))
#        print "tmpPos: {}, matrix: {}".format(tmpPos, matrix)
        self["pos"] = np.round(np.array((matrix*tmpPos.T)[:2]).flatten(), NUMDEC)
#        print "original pos: {}, result pos: {}".format(tmpPos, self["pos"])
        if self["name"] == "gripper":
            self["euler"] = 0#np.array([0.0,0.0,0.0])
        else:
            self["euler"] += euler
            

#        print "original ori: {}, resulting ori: {}".format(q2, self["orientation"])
        tmplV = np.matrix(np.concatenate((self["linVel"],[0])))
        self["linVel"] = np.round(np.array((matrix*tmplV.T)[:2]).flatten(), NUMDEC)
#        tmpaV = np.matrix(np.concatenate((self["angVel"],[0])))
#        self["angVel"] = np.round(np.array((matrix*tmpaV.T)[:2]).flatten(), NUMDEC)
        
    def fromInteractionState(self, intState):
        self.update({"id": intState["sid"], "name":intState["sname"], "pos":np.copy(intState["spos"]), 
                     "euler":np.copy(intState["seuler"]), "linVel":np.copy(intState["slinVel"]), 
                     "angVel": copy.deepcopy(intState["sangVel"])})
        if intState["contact"]:
            self["contact"] = intState["oname"]
            

class Action(State):
    
    def __init__(self, cmd=GZCMD["NOTHING"], direction=np.array([0.0,0.0,0.0])):
        
        self.update({"cmd":int(round(cmd)), "mvDir": direction})
        State.__init__(self)
        self.relKeys = self.keys()
        
    def transform(self, matrix):
        tmpMVDir = np.matrix(np.concatenate((self["mvDir"][:2],[0])))
        self["mvDir"] = np.round(np.array((matrix*tmpMVDir.T)[:2]).flatten(), NUMDEC)            
            
class InteractionState(State):
    
    def __init__(self, intId, o1):
        assert isinstance(o1, ObjectState), "{} (o1) is not an ObjectState!".format(o1)
        self.update({"intId": intId, "sid":o1["id"], "sname": o1["name"], 
                     "stype": o1["type"], "spos":o1["pos"], 
                     "seuler": o1["euler"], "slinVel": o1["linVel"], 
                     "sangVel": o1["angVel"], "dist": 0, "dir": np.zeros(2),
                     "contact": 0, "oid": -1, "oname": "", "otype": 0, 
                     "deuler": 0, "dlinVel": np.zeros(2), "dangVel": 0})
        #Do not move from here because the keys need to be set before State.init and the relKeys need to be changed afterwards             
        State.__init__(self) 
#        self.relKeys = ["spos", "slinVel"]
        self.relKeys = self.keys()
        self.relKeys.remove("intId")
        self.relKeys.remove("sname")
        self.relKeys.remove("oname")
        self.relKeys.remove("stype")
        self.relKeys.remove("otype")
    
#        self.relKeys.remove("sangVel")
#        self.relKeys.remove("dangVel")
#        self.relKeys.remove("contact")
#        self.relKeys.remove("sid")
#        self.relKeys.remove("oid")
#        self.relKeys.remove("spos")
#        self.weights["slinVel"] = 2
#        self.weights["spos"] = 0.5
#        self.weights["dist"] = 2
#        self.weights["dir"] = 2
                   
                    
                     
    def fill(self, o2):
        assert isinstance(o2, ObjectState), "{} (o2) is not an ObjectState!".format(o2)
        self["dir"] = o2["pos"]-self["spos"]
        self["dist"] = self.computeDistance(o2)
#        print "distance from {} to {}: {}".format(self["sid"], self["oid"], self["dist"])
        
        self["oid"] = o2["id"]
        self["oname"] = o2["name"]
        self["otype"] = o2["type"]
        self["deuler"] = o2["euler"]-self["seuler"] 
        self["dlinVel"] = o2["linVel"] - self["slinVel"]
        self["dangVel"] = o2["angVel"] - self["sangVel"]
        if o2["contact"] == self["sname"]:
            self["contact"] = 1
            
    def computeDistance(self, o2):
        if self["sid"] == 8:
            mp = np.copy(o2["pos"])
#            mp[2] = self["spos"][2]
            ang = o2["euler"]
            
            x0x,x0y = self["spos"]
        elif self["sid"] == 15:
            mp = np.copy(self["spos"])
#            mp[2] = o2["pos"][2]
            ang = self["seuler"]
            x0x,x0y = o2["pos"]
            
        c = math.cos(ang)
        s = math.sin(ang)
        x1x = -0.25
        x1y = -0.05
        x2x = 0.25
        x2y = -0.05
        x1xn = x1x*c - x1y*s
        x1yn = x1x*s + x1y*c
        x2xn = x2x*c - x2y*s
        x2yn = x2x*s + x2y*c
        x1n = np.array([x1xn,x1yn]) + mp
        x2n = np.array([x2xn,x2yn]) + mp
        if x0x <= x2x and x0x >= x1x: 
            d1 = abs((x2n[0]-x1n[0])*(x1n[1]-x0y)-(x1n[0]-x0x)*(x2n[1]-x1n[1]))/math.sqrt((x2n[0]-x1n[0])**2+(x2n[1]-x1n[1])**2) - 0.025
        else:
            d1 = min(np.linalg.norm(x1n-np.array([x0x,x0y])), np.linalg.norm(x2n-np.array([x0x,x0y])))
        x1x = -0.25
        x1y = 0.05
        x2x = 0.25
        x2y = 0.05
        x1xn = x1x*c - x1y*s
        x1yn = x1x*s + x1y*c
        x2xn = x2x*c - x2y*s
        x2yn = x2x*s + x2y*c
        x1n = np.array([x1xn,x1yn]) + mp
        x2n = np.array([x2xn,x2yn]) + mp
        if x0x <= x2x and x0x >= x1x: 
            d2 = abs((x2n[0]-x1n[0])*(x1n[1]-x0y)-(x1n[0]-x0x)*(x2n[1]-x1n[1]))/math.sqrt((x2n[0]-x1n[0])**2+(x2n[1]-x1n[1])**2) - 0.025
        else:
            d2 = min(np.linalg.norm(x1n-np.array([x0x,x0y])), np.linalg.norm(x2n-np.array([x0x,x0y])))
        x1x = -0.25
        x1y = 0.05
        x2x = -0.25
        x2y = -0.05
        x1xn = x1x*c - x1y*s
        x1yn = x1x*s + x1y*c
        x2xn = x2x*c - x2y*s
        x2yn = x2x*s + x2y*c
        x1n = np.array([x1xn,x1yn]) + mp
        x2n = np.array([x2xn,x2yn]) + mp
        if x0y <= x1y and x0y >= x2y: 
            d3 = abs((x2n[0]-x1n[0])*(x1n[1]-x0y)-(x1n[0]-x0x)*(x2n[1]-x1n[1]))/math.sqrt((x2n[0]-x1n[0])**2+(x2n[1]-x1n[1])**2) - 0.025
        else:
            d3 = min(np.linalg.norm(x1n-np.array([x0x,x0y])), np.linalg.norm(x2n-np.array([x0x,x0y])))
        x1x = 0.25
        x1y = 0.05
        x2x = 0.25
        x2y = -0.05
        x1xn = x1x*c - x1y*s
        x1yn = x1x*s + x1y*c
        x2xn = x2x*c - x2y*s
        x2yn = x2x*s + x2y*c
        x1n = np.array([x1xn,x1yn]) + mp
        x2n = np.array([x2xn,x2yn]) + mp
        if x0y <= x1y and x0y >= x2y: 
            d4 = abs((x2n[0]-x1n[0])*(x1n[1]-x0y)-(x1n[0]-x0x)*(x2n[1]-x1n[1]))/math.sqrt((x2n[0]-x1n[0])**2+(x2n[1]-x1n[1])**2) - 0.025
        else:
            d4 = min(np.linalg.norm(x1n-np.array([x0x,x0y])), np.linalg.norm(x2n-np.array([x0x,x0y])))
        return max(0.0,min((d1,d2,d3,d4)))
            
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
                tmp["pos"] = np.round(np.array([m.pose.position.x,m.pose.position.y]), NUMDEC) #/ 2.0
#                print "parsing model: {}, pos: {}".format(m.name, tmp["pos"])
                tmp["euler"]  = np.round(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                            m.pose.orientation.z,m.pose.orientation.w])), NUMDEC)[2]
#                tmp["euler"][:2] = 0
#                tmp["euler"]  = common.quaternionToEuler(np.array([0.0,0.0,m.pose.orientation.z,m.pose.orientation.w]))
#                print "quaternion: {}, euler: {}".format(np.array([m.pose.orientation.x,m.pose.orientation.y,
#                                            m.pose.orientation.z,m.pose.orientation.w]), tmp["euler"])
                tmp["linVel"] = np.round(np.array([m.linVel.x,m.linVel.y]), NUMDEC)
                if np.linalg.norm(tmp["linVel"]) < 0.01:
                    tmp["linVel"] = np.array([0.0,0.0])
#                print "name: {}, linVel: {}".format(m.name, tmp["linVel"])
                tmp["angVel"] = np.round(m.angVel.z, NUMDEC)
#                print "Raw AngVel: ", tmp["angVel"]
                tmp["name"] = m.name
                tmp["id"] = m.id
                tmp["type"] = m.type
                self.objectStates[m.name] = tmp
                
                if m.name == "blockA" and self.transM == None:
                    self.transM = common.eulerPosToTransformation2d(tmp["euler"],tmp["pos"])
                    self.invTrans = common.invertTransMatrix(self.transM)
#                    print "invTrans: ", self.invTrans
                    self.ori = np.copy(tmp["euler"])

                
    def parseInteractions(self):
        tmpList = self.objectStates.values()
        for o in tmpList:
            #Transform to local block coordinate system
            o.transform(self.invTrans, -self.ori)
        for o1 in self.objectStates.values():
#            print "interactionState for o1: ", o1
            intState = InteractionState(self.numIntStates, o1)
#            self.addInteractionState(intState)
            for o2 in tmpList:
                if not np.array_equal(o1,o2):             
                    intState.fill(o2)
#                    if intState["sname"] == "blockA":
#                        print "intState blockA spos: ", intState["spos"]
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
#            print "Tmp after back transformation: ", tmp
            self.objectStates[tmp["name"]] = tmp
            if tmp["name"] == "blockA":
                self.transM = common.eulerPosToTransformation(tmp["euler"],tmp["pos"])
                self.invTrans = common.invertTransMatrix(self.transM)
                self.ori = np.copy(tmp["euler"])
        self.parseInteractions()
#        
#        print "InteractionStates: ", self.interactionStates.values()

    def getInteractionState(self, sname):
        for i in self.interactionStates.values():
            if i["sname"] == sname:
                return i
        return None    