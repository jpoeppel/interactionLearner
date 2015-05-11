#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:31:07 2015

@author: jpoeppel
"""

import common
from common import NUMDEC
import common.GAZEBOCMDS as GZCMD
from metrics import similarities
from metrics import differences
import numpy as np
import math

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
        for k in self.relKeys:
            if k not in const.keys():
#            if k != "spos":
                if isinstance(self[k], np.ndarray):
                    r = np.concatenate((r,self[k]))
                elif not isinstance(self[k], unicode):
                    r = np.concatenate((r,[self[k]]))
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
        
        
class ObjectState(State):
    """
        State class used to represent object states.
        Holds information about object position, orientation, type, id, name 
        and other properties.
    """
    
    def __init__(self):
        State.__init__(self)
        self.update({"id": -1, "name": "", "type": -1, "pos": np.zeros(3), 
                         "orientation": np.zeros(4), "linVel": np.zeros(3), 
                         "angVel": np.zeros(3), "contact": None})
        self.relKeys = self.keys()
        
    def getTranformationMatrix(self):
        px,py,pz = self["pos"]
        x,y,z,w = self["orientation"]
        return np.matrix([[1-2*y*y-2*z*z, 2*x*y + 2*w*z, 2*x*z - 2*w*y, px],[2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x, py],
                          [2*x*z+2*w*y,2*y*z-2*w*x, 1-2*x*x-2*y*y, pz],[0.0,0.0,0.0,1.0]])
                          
    def transform(self, matrix, q1):
#        print "calling transform for: ", self["name"]
        tmpPos = np.matrix(np.concatenate((self["pos"],[1])))
#        print "tmpPos: {}, matrix: {}".format(tmpPos, matrix)
        self["pos"] = np.round(np.array((matrix*tmpPos.T)[:3]).flatten(), NUMDEC)
#        print "original pos: {}, result pos: {}".format(tmpPos, self["pos"])
        q2 = self["orientation"]
#        q1[:3] *= -1
#        self["orientation"] = np.array([q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0],
#                                -q1[0]*q2[2]+q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1],
#                                q1[0]*q2[1]-q1[1]*q2[0]+q1[2]*q2[3]+q1[3]*q2[2],
#                                -q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]+q1[3]*q2[3]])
#        self["orientation"] = np.array([q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2],
#                                q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1],
#                                q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0],
#                                q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]])
        newOri = np.zeros(4)
        newOri[0] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
        newOri[1] = -q1[0]*q2[2]+q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
        newOri[2] = q1[0]*q2[1]-q1[1]*q2[0]+q1[2]*q2[3]+q1[3]*q2[2]
        newOri[3] = -q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]+q1[3]*q2[3]
        newOri /= np.linalg.norm(newOri)
#        self["orientation"] /= np.linalg.norm(self["orientation"])
        self["orientation"] = np.round(newOri, NUMDEC)
#        print "original ori: {}, resulting ori: {}".format(q2, self["orientation"])
        tmplV = np.matrix(np.concatenate((self["linVel"],[0])))
        self["linVel"] = np.round(np.array((matrix*tmplV.T)[:3]).flatten(), NUMDEC)
        tmpaV = np.matrix(np.concatenate((self["angVel"],[0])))
        self["angVel"] = np.round(np.array((matrix*tmpaV.T)[:3]).flatten(), NUMDEC)
        
    def fromInteractionState(self, intState):
        self.update({"id": intState["sid"], "name":intState["sname"], "pos":intState["spos"], 
                     "orientation":intState["sori"], "linVel":intState["slinVel"], 
                     "angVel": intState["sangVel"]})
        if intState["contact"]:
            self["contact"] = intState["oname"]
        
        
class InteractionState(State):
    
    def __init__(self, intId, o1):
        assert isinstance(o1, ObjectState), "{} (o1) is not an ObjectState!".format(o1)
        self.update({"intId": intId, "sid":o1["id"], "sname": o1["name"], 
                     "stype": o1["type"], "spos":o1["pos"], 
                     "sori": o1["orientation"], "slinVel": o1["linVel"], 
                     "sangVel": o1["angVel"], "dist": 0, "dir": np.zeros(3),
                     "contact": 0, "oid": -1, "oname": "", "otype": 0, 
                     "dori": np.zeros(4), "dlinVel": np.zeros(3), "dangVel":np.zeros(3)})
        #Do not move from here because the keys need to be set before State.init and the relKeys need to be changed afterwards             
        State.__init__(self) 
#        self.relKeys = ["spos", "slinVel"]
        self.relKeys = self.keys()
        self.relKeys.remove("intId")
        self.relKeys.remove("sname")
        self.relKeys.remove("oname")
        self.relKeys.remove("stype")
        self.relKeys.remove("otype")
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
                    

#        self["dist"] = np.linalg.norm(self["spos"]-o2["pos"])
        self["dist"] = self.computeDistance(o2)
#        print "distance from {} to {}: {}".format(self["sid"], self["oid"], self["dist"])
        self["dir"] = o2["pos"]-self["spos"]
#        self["dir"] /= self["dist"] # make direction unit vector
#        self["dir"] /= np.linalg.norm(self["dir"])
        self["oid"] = o2["id"]
        self["oname"] = o2["name"]
        self["otype"] = o2["type"]
        self["dori"] = o2["orientation"]-self["sori"] # TODO Fix THIS IS WRONG!!! Although it works reasonably well
        self["dlinVel"] = o2["linVel"] - self["slinVel"]
        self["dangVel"] = o2["angVel"] - self["sangVel"]
        if o2["contact"] == self["sname"]:
            self["contact"] = 1
            
    def computeDistance(self, o2):
        if self["sid"] == 8:
            mp = o2["pos"]
            mp[2] = self["spos"][2]
            sign = o2["orientation"][2]
            ang = 2.0*math.acos(o2["orientation"][3])
            
            x0x,x0y,x0z = self["spos"]
        elif self["sid"] == 15:
            mp = self["spos"]
            mp[2] = o2["pos"][2]
            sign = self["sori"][2]
            ang = 2.0*math.acos(self["sori"][3])
            x0x,x0y,x0z = o2["pos"]
            
        c = math.cos(ang)
        s = math.sin(ang)
        if sign != 0:
            s *= sign/abs(sign)
        x1x = -0.25
        x1y = -0.05
        x2x = 0.25
        x2y = -0.05
        x1xn = x1x*c - x1y*s
        x1yn = x1x*s + x1y*c
        x2xn = x2x*c - x2y*s
        x2yn = x2x*s + x2y*c
        x1n = np.array([x1xn,x1yn,0]) + mp
        x2n = np.array([x2xn,x2yn,0]) + mp
#        print "mp: ", mp
#        print "x1n: {}, x2n: {}".format(x1n,x2n)
        d = abs((x2n[0]-x1n[0])*(x1n[1]-x0y)-(x1n[0]-x0x)*(x2n[1]-x1n[1]))/math.sqrt((x2n[0]-x1n[0])**2+(x2n[1]-x1n[1])**2) - 0.025
#            d = np.linalg.norm(np.cross(x2n-x1n, x1n-self["spos"]))/np.linalg.norm(x2n-x1n)
        if d < 0.002:
            return 0.0
        else:
            return d
            
class Action(State):
    
    def __init__(self, cmd=GZCMD["NOTHING"], direction=np.array([0.0,0.0,0.0])):
        
        self.update({"cmd":int(round(cmd)), "mvDir": direction})
        State.__init__(self)
        self.relKeys = self.keys()
        
    def transform(self, matrix):
        tmpMVDir = np.matrix(np.concatenate((self["mvDir"],[0])))
        self["mvDir"] = np.array((matrix*tmpMVDir.T)[:3]).flatten()

        
class WorldState(object):
    
    def __init__(self, transM = None, invTrans = None, quat = None):
        self.objectStates = {}
        self.interactionStates = {}
        self.numIntStates = 0
        self.predictionCases = {}
        self.transM = transM
        self.invTrans = invTrans
        self.quat = quat

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
                tmp = ObjectState()
                tmp["pos"] = np.round(np.array([m.pose.position.x,m.pose.position.y,m.pose.position.z]), NUMDEC) #/ 2.0
#                tmp["orientation"]  = np.round(np.array([m.pose.orientation.x,m.pose.orientation.y,
#                                            m.pose.orientation.z,m.pose.orientation.w]), NUMDEC)
                tmp["orientation"]  = np.round(np.array([0.0,0.0,m.pose.orientation.z,m.pose.orientation.w]), NUMDEC)
                tmp["orientation"] /= np.linalg.norm(tmp["orientation"])
                tmp["linVel"] = np.round(np.array([m.linVel.x,m.linVel.y,m.linVel.z]), NUMDEC)
#                print "name: {}, linVel: {}".format(m.name, tmp["linVel"])
                tmp["angVel"] = np.round(np.array([m.angVel.x,m.angVel.y,m.angVel.z]), 1)
                tmp["name"] = m.name
                tmp["id"] = m.id
                tmp["type"] = m.type
                self.objectStates[m.name] = tmp
                
                if m.name == "blockA" and self.transM == None:
                    self.transM = tmp.getTranformationMatrix()
                    self.invTrans = np.matrix(np.zeros((4,4)))
                    self.invTrans[:3,:3] = self.transM[:3,:3].T
                    self.invTrans[:3,3] = -self.transM[:3,:3].T*self.transM[:3,3]
                    self.invTrans[3,3] = 1.0
                    self.quat = np.copy(tmp["orientation"])
#                    self.quat[:3] *= -1
#                print "BlockA angVel: ", tmp["angVel"]
                
    def parseInteractions(self):
        tmpList = self.objectStates.values()
        for o in tmpList:
            q = self.quat
            q[:3] *= -1
            #Transform to local block coordinate system
            o.transform(self.invTrans, q)
        for o1 in self.objectStates.values():
#            print "interactionState for o1: ", o1
            intState = InteractionState(self.numIntStates, o1)
#            self.addInteractionState(intState)
            for o2 in tmpList:
                if not np.array_equal(o1,o2):             
                    intState.fill(o2)
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
        for intState in worldState.interactionStates.values():
            tmp = ObjectState()
            tmp.fromInteractionState(intState)
            
            invq = worldState.quat
            #Transform back to world coordinate system first
            tmp.transform(worldState.transM, invq)
            print "Tmp after back transformation: ", tmp
            self.objectStates[tmp["name"]] = tmp
            if tmp["name"] == "blockA":
                self.transM = tmp.getTranformationMatrix()
                self.invTrans = np.matrix(np.zeros((4,4)))
                self.invTrans[:3,:3] = self.transM[:3,:3].T
                self.invTrans[:3,3] = -self.transM[:3,:3].T*self.transM[:3,3]
                self.invTrans[3,3] = 1.0
                self.quat = np.copy(tmp["orientation"])
        self.parseInteractions()
#        
#        print "InteractionStates: ", self.interactionStates.values()

    def getInteractionState(self, sname):
        for i in self.interactionStates.values():
            if i["sname"] == sname:
                return i
        return None    