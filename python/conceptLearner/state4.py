#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:40:03 2015

@author: jpoeppel
"""

import numpy as np
import common
from common import NUMDEC
from common import GAZEBOCMDS as GZCMD
import math
import copy

from config import DIFFERENCES, INTERACTION_STATES, SINGLE_INTSTATE

WIDTH = {"blockA":0.25, "gripper": 0.025} #Width from the middle point
DEPTH = {"blockA":0.05, "gripper": 0.025} #Height from the middle point

CLOSING_REFERAL = False

class State(dict):
    
    def __init__(self):
        self.vec = np.array([]) #Np.array representing the state vector
        self.mask = np.array(range(len(self.vec))) #Np.array to mask the access to the vector
        self.relKeys = self.keys()
        
    def getVec(self, mask=None):
        if mask != None:
            return self.vec[mask]
        else:
            return self.vec[self.mask]
            
    def score(self, other):
        return np.exp(-0.5*np.linalg.norm(self.getVec()-other.getVec()))
        
    def relItems(self):
        for k in self.relKeys:
            yield k, self[k]
            
    def relevantItems(self):
        r = []
        for k in self.relKeys:
            r.append((k,self[k]))
        return r
        
    def toSelVec(self):
        return self.getVec(self.selMask)
        
        
        
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
            
        for k, v in self.relevantItems():
            if np.linalg.norm(v-other[k]) > 0.001:
                return False
        
        return True
        
    def __ne__(self, other):
        return not self.__eq__(other)
            
        
class Action(State):
    
    def __init__(self, cmd=GZCMD["NOTHING"], direction=np.array([0.0,0.0,0.0])):
        self.vec = np.zeros(4)
        self.update({"cmd":self.vec[0:1], "mvDir": self.vec[1:4]})
        self["mvDir"][:3] = direction
        self.relKeys = self.keys()
        self.mask = np.array(range(len(self.vec)))
        self.selMask = self.mask
        
    def transform(self, matrix):
        tmpMVDir = np.zeros(4)
        tmpMVDir[:3] = np.copy(self["mvDir"])
        self["mvDir"][:3] = np.round(np.array((np.dot(matrix,tmpMVDir))[:3]), NUMDEC)      
   
    @classmethod
    def sample(cls, number):
        return [cls(cmd=GZCMD["MOVE"], direction=np.array([0.5*math.cos(x), 0.5*math.sin(y),0.0])) for x in [0+i*math.pi/number for i in range(number)] for y in [0+i*math.pi/number for i in range(number)]]
         
    @classmethod
    def getGripperAction(cls, cmd=GZCMD["NOTHING"], direction = np.zeros(3)):
        res = cls()
        res["cmd"] = cmd
        res["mvDir"] = direction
        return res
    
class ObjectState(State):
    
    def __init__(self):
        self.vec = np.zeros(9)
        self.update({"name": "", "id": self.vec[0:1], "posX":self.vec[1:2],
                     "posY": self.vec[2:3], "posZ": self.vec[3:4], "ori":self.vec[4:5], 
                     "linVelX": self.vec[5:6], "linVelY": self.vec[6:7],
                     "linVelZ": self.vec[7:8], "angVel": self.vec[8:9],
                     "pos": self.vec[1:4], "linVel": self.vec[5:8], "contact":None})
        self.mask = np.array(range(len(self.vec)))
        self.relKeys = ["id", "posX", "posY", "posZ", "ori", "linVelX", "linVelY", "linVelZ", "angVel"]
        self.actionItems = ["linVelX", "linVelY", "linVelZ", "angVel"]
        self.mask = np.array([1,2,3,4,5,6,7,8])
        
    @classmethod
    def fromInteractionState(cls, intState):
        o1 = cls()
        o1["name"] = intState["sname"]
        o1["id"][0] = intState["sid"]
        o1["pos"][:3] = np.copy(intState["spos"])
        o1["linVel"][:3] = np.copy(intState["slinVel"])
        o1["angVel"][0] = np.copy(intState["sangVel"])
        o1["ori"][0] = np.copy(intState["sori"])
        o2 = cls()
        o2["name"] = intState["oname"]
        o2["id"] = intState["oid"]
        if DIFFERENCES:
            o2["pos"][:3], o2["linVel"][:3] = common.globalPosVel(o1["pos"], o1["ori"], intState["dir"], intState["dori"])
            o2["angVel"][0] = o1["angVel"] + intState["dangVel"]
            o2["ori"][0] = o1["ori"] + intState["dori"]
        else:
            o2["pos"][:3] = np.copy(intState["opos"])
            o2["linVel"][:3] = np.copy(intState["olinVel"])
            o2["angVel"][0] = np.copy(intState["oangVel"])
            o2["ori"][0] = np.copy(intState["oori"])
        
        return o1, o2
        
    @classmethod
    def clone(cls, other):
        assert isinstance(other, ObjectState), "{} is no ObjectState (but {})".format(other, type(other))
        res = cls()
        np.copyto(res.vec, other.vec)
        res["name"] = other["name"]
        res["contact"] = copy.deepcopy(other["contact"])
        return res
        
    def getKeyPoints(self):
        p1x = WIDTH[self["name"]]
        p2x = -p1x
        p1y = DEPTH[self["name"]]
        p2y = -p1y
        ang = self["ori"]
        c = math.cos(ang)
        s = math.sin(ang)
        p1xn = p1x*c -p1y*s + self["posX"][0]
        p1yn = p1x*s + p1y*c + self["posY"][0]
        p2xn = p2x*c - p2y*s + self["posX"][0]
        p2yn = p2x*s + p2y*c + self["posY"][0]
        return np.array([np.copy(self["pos"][:]), np.array([p1xn,p1yn,self["posZ"]]), np.array([p2xn,p2yn,self["posZ"]])])
        
    def compare(self, other):
        assert self["name"] == other["name"], "Should only compare the same objects not {} and {}".format(self["name"], other["name"])
        sKeyPoints = self.getKeyPoints()
        oKeyPoints = other.getKeyPoints()
        return sum(np.linalg.norm(sKeyPoints-oKeyPoints,axis=1))/3.0
        
    def transform(self, matrix, euler):
        res = ObjectState.clone(self)
        tmpPos = np.ones(4)
        tmpPos[:3] = np.copy(self["pos"])
        res["pos"] = np.round(np.array((np.dot(matrix,tmpPos))[:3]), NUMDEC)
        if res["name"] == "gripper":
            res["ori"][0] = 0.0
        else:
            res["ori"][0] += euler

        tmplV = np.zeros(4)
        tmplV[:3] = np.copy(self["linVel"])
        res["linVel"] = np.round(np.array((np.dot(matrix,tmplV))[:3]), NUMDEC)
#        tmpaV = np.zeros(4)
#        tmpaV[:3] = np.copy(self["angVel"])
#        res["angVel"] = np.round(np.array((np.dot(matrix,tmpaV))[:3]), NUMDEC)
#        print "res pos:", res["pos"]
        return res

        
class InteractionState(State):
    
    def __init__(self):
        
        if INTERACTION_STATES:
            
            if DIFFERENCES:
                self.vec = np.zeros(20)
                self.update({"sname": "", "oname": "", "sid":self.vec[0:1], 
                             "oid": self.vec[1:2], "spos":self.vec[2:5], "sori":self.vec[5:6],
                             "slinVel": self.vec[6:9],"sangVel": self.vec[9:10], "dist": self.vec[10:11], "dir": self.vec[11:14],
                             "contact": self.vec[14:15], "dori": self.vec[15:16], "dlinVel": self.vec[16:19], 
                             "dangVel": self.vec[19:20]})
                self.features = np.array(["sid", "oid", "dist", "dir", "spos", "sori", "slinVel", "sangVel", "contact",
                                          "dori", "dlinVel", "dangVel"])
            else:  
                self.vec = np.zeros(23)
                self.update({"sname": "", "oname": "", "sid":self.vec[0:1], 
                             "oid": self.vec[1:2], "spos":self.vec[2:5], "sori":self.vec[5:6],
                             "slinVel": self.vec[6:9],"sangVel": self.vec[9:10], "dist": self.vec[10:11], "opos": self.vec[11:14],
                             "contact": self.vec[14:15], "oori": self.vec[15:16], "olinVel": self.vec[16:19], 
                             "oangVel": self.vec[19:20], "dir":self.vec[20:23]})
                self.features = np.array(["sid", "oid", "dist", "dir", "spos", "sori", "slinVel", "sangVel", "contact",
                                          "opos", "oori", "olinVel", "oangVel"])
            self.relKeys = list(self.features)
            self.mask = np.array(range(len(self.vec)))
            self.selMask = self.mask
        else:
            self.vec = np.zeros(16)
            self.update({"name": "", "sname":"", "oname": "", "sid": self.vec[0:1], "oid":self.vec[1:2],
                         "dist": self.vec[2:3], "closing": self.vec[3:4], "contact":self.vec[4:5],
                         "relPosX": self.vec[5:6], "relPosY": self.vec[6:7], "relPosZ":self.vec[7:8],
                         "relVlX": self.vec[8:9], "relVlY": self.vec[9:10], "relVlZ": self.vec[10:11],
                         "closingDivDist":self.vec[11:12], "relPos": self.vec[5:8], "relVl": self.vec[8:11], 
                         "closing1":self.vec[12:13], "closing2":self.vec[13:14], "closing1DivDist":self.vec[14:15], 
                         "closing2DivDist":self.vec[15:16] })
            self.features = np.array(["sid","oid","dist","closing","contact","relPosX", "relPosY", "relPosZ", 
                                      "relVlX", "relVlY", "relVlZ", 
                                      "closingDivDist", "closing1", "closing2", "closing1DivDist", "closing2DivDist"])
            self.mask = np.array(range(len(self.vec)))
            self.selMask = self.mask
#        if CLOSING_REFERAL:
#            self.mask=[0,1,2,5,6,8,9,12,13]
#        else:            
#            self.mask = [0,1,2,3,4,5,6,8,9]
#            self.mask = np.array([0,1,4,5,6,8,9,12,13,14,15])
#        self.relKeys = ["sid", "oid", "dist", "closing", "contact", "relPosX", "relPosY", "relPosZ", "closingDivDist", "closing2"]
        
        
    @classmethod
    def parseInteraction(cls, os1, os2):
        intState = cls()
        intState["name"] = os1["name"]+os2["name"]
        intState["sname"] = os1["name"]
        intState["oname"] = os2["name"]
        intState["sid"][0] = os1["id"]
        intState["oid"][0] = os2["id"]
#                    if CLOSING_REFERAL:
#                        intState["dist"][0], intState["closing1"][0], intState["closing2"][0] = self.computeDistanceClosing(os1,os2)
#                    else:
#                        intState["dist"][0], intState["closing"][0] = self.computeDistanceClosing(os1,os2)
        intState["dist"][0], closing, closing1, closing2 = cls.computeDistanceClosing(os1,os2)
        if intState["dist"] < 0.0:
            intState["dist"] = 0.0
        if os1["contact"] == os2["name"]:
            intState["contact"][0] = 1
        if INTERACTION_STATES:
            intState["spos"][:3] = os1["pos"]
            intState["slinVel"][:3] = os1["linVel"]
            intState["sori"][0] = os1["ori"]
            intState["sangVel"][0] = os1["angVel"]
            if DIFFERENCES:
                intState["dir"][:3] = os2["pos"]-os1["pos"]
                intState["dori"][0] = os2["ori"]-os1["ori"] 
                intState["dlinVel"][:3] = os2["linVel"] - os1["linVel"]
                intState["dangVel"][0] = os2["angVel"] - os1["angVel"]
            else:
                intState["opos"][:3] = os2["pos"]
                intState["oori"][0] = os2["ori"]
                intState["olinVel"][:3] = os2["linVel"]
                intState["oangVel"][0] = os2["angVel"]
                intState["dir"][:3] = os2["pos"]-os1["pos"]
            
        else:
            intState["closing"][0], intState["closing1"][0], intState["closing2"][0] = closing, closing1, closing2
            intState["relPos"][:3], intState["relVl"][:3] = cls.calcRelPosVel(os1,os2)
            if intState["dist"] != 0:
                intState["closingDivDist"][0] = intState["closing"]/intState["dist"]
                intState["closing1DivDist"][0] = intState["closing1"]/intState["dist"]
                intState["closing2DivDist"][0] = intState["closing2"]/intState["dist"]
            else:
                intState["closingDivDist"][0] = intState["closing"]/0.001
                intState["closing1DivDist"][0] = intState["closing1"]/0.001
                intState["closing2DivDist"][0] = intState["closing2"]/0.001
    #                    intState["closingDivDist"][0] = np.round(math.tanh(1+0.1*intState["closingDivDist"]), NUMDEC)
        return intState
        
    @classmethod
    def calcRelPosVel(cls, os1, os2):
        euler = np.zeros(3)
        euler[2] = os1["ori"][0]
        trans = common.eulerPosToTransformation(euler, os1["pos"])
        invTrans = common.invertTransMatrix(trans)
        tmpPos = np.ones(4)
        tmpPos[:3] = np.copy(os2["pos"])
        newPos = np.dot(invTrans, tmpPos)[:3]
        tmpVel = np.zeros(4)
        tmpVel[:3] = np.copy(os2["linVel"])-np.copy(os1["linVel"])
        tmpVel = np.dot(invTrans, tmpVel)[:3]
        return np.round(newPos, NUMDEC), np.round(tmpVel, NUMDEC)
        
    @classmethod
    def calcRelPosition(cls, os1, os2):
        euler = np.zeros(3)
        euler[2] = os1["ori"][0]
        trans = common.eulerPosToTransformation(euler, os1["pos"])
        invTrans = common.invertTransMatrix(trans)
        tmpPos = np.ones(4)
        tmpPos[:3] = np.copy(os2["pos"])
        newPos = np.dot(invTrans, tmpPos)[:3]
        return np.round(newPos, NUMDEC)
        
    @classmethod
    def calcDist(cls, p, mp, x1x,x1y,x2x,x2y,c,s):
        x1xn = x1x*c - x1y*s
        x1yn = x1x*s + x1y*c
        x2xn = x2x*c - x2y*s
        x2yn = x2x*s + x2y*c
        v = np.array([x1xn,x1yn]) + mp
        w = np.array([x2xn,x2yn]) + mp
#        print "x1xn: {}, x1yn: {}, x2xn: {}, x2yn:{}".format(x1xn,x1yn,x2xn,x2yn)
#        print "x1x: {}, x1y: {}, x2x: {}, x2y: {}".format(x1x,x1y,x2x,x2y)
#        print "mp: ", mp
        l2 = np.dot(v-w, v-w)
#        print "l2: ", l2
        if l2 == 0.0:
            return np.sqrt(np.dot(v-p, v-p)), v
        t = np.dot(p-v, w-v) / l2
        if t < 0.0:
            return np.sqrt(np.dot(v-p,v-p)), v
        elif t > 1.0:
            return np.sqrt(np.dot(w-p,w-p)), w
        projection = v + t * (w - v)
        return np.sqrt(np.dot(p-projection, p-projection)), projection
        
    @classmethod
    def computeDistanceClosing(cls, os1, os2):
#        print "ComputeDistanceClosing: os1vel: {}, os2vel: {}".format(os1["linVel"], os2["linVel"])
        if os1["name"] == "gripper":
            p = os1["pos"][:2]
            mp = np.copy(os2["pos"][:2])
            ang = os2["ori"]
            blockN = os2["name"]
            vel = os1["linVel"]-os2["linVel"]
        elif os2["name"] == "gripper":
            p = os2["pos"][:2]
            mp = np.copy(os1["pos"][:2])
            ang = os1["ori"]
            blockN = os1["name"]
            vel = os2["linVel"]-os1["linVel"]
        else:
            raise NotImplementedError("Currently only distance between gripper and object is implemented.")
        c = math.cos(ang)
        s = math.sin(ang)
        
        #Right
        x1x = WIDTH[blockN]
        x1y = DEPTH[blockN]
        x2x = x1x
        x2y = -x1y
        d1, p1 = cls.calcDist(p,mp,x1x,x1y,x2x,x2y,c,s) 
        
        #Top
        x1x = WIDTH[blockN]
        x1y = DEPTH[blockN]
        x2x = -x1x
        x2y = x1y
        d2, p2 = cls.calcDist(p,mp,x1x,x1y,x2x,x2y,c,s)
        
        #Left
        x1x = -WIDTH[blockN]
        x1y = DEPTH[blockN]
        x2x = x1x
        x2y = -x1y
        d3, p3 = cls.calcDist(p,mp,x1x,x1y,x2x,x2y,c,s)
        
        #Bottom
        x1x = WIDTH[blockN]
        x1y = -DEPTH[blockN]
        x2x = -x1x
        x2y = x1y
        d4, p4 = cls.calcDist(p,mp,x1x,x1y,x2x,x2y,c,s)
        
        ds = np.array([d1,d2,d3,d4])
        ps = [p1,p2,p3,p4]
        di = np.argmin(ds)
        
        normal = np.zeros(3)
        normal[:2] = (p-ps[di])
        norm = np.linalg.norm(normal)
        if norm > 0.0:
            normal /= np.linalg.norm(normal)
        
        
#        print "normal: ", normal
#        print "vel: ", vel
#        print "norm vel: ", np.linalg.norm(vel)
#        if CLOSING_REFERAL:
#            normal1 = np.array([math.cos(di*math.pi/2.0+ang), math.sin(di*math.pi/2.0+ang),0.0])
#            if ds[di-1] < ds[(di+1) % len(ds)]:
#                normal2 = np.array([math.cos((di-1)*math.pi/2.0+ang), math.sin((di-1)*math.pi/2.0+ang),0.0])
#            else:
#                normal2 = np.array([math.cos((di+1)*math.pi/2.0+ang), math.sin((di+1)*math.pi/2.0+ang),0.0])
#            return np.round(ds[di]-0.025, NUMDEC), np.round(np.dot(normal1, vel), NUMDEC), np.round(np.dot(normal2, vel), NUMDEC)
#        else:        
#            return np.round(ds[di]-0.025, NUMDEC), np.round(np.dot(normal, vel), NUMDEC)
        normal1 = np.array([math.cos(di*math.pi/2.0+ang), math.sin(di*math.pi/2.0+ang),0.0])
        if ds[di-1] < ds[(di+1) % len(ds)]:
            normal2 = np.array([math.cos((di-1)*math.pi/2.0+ang), math.sin((di-1)*math.pi/2.0+ang),0.0])
        else:
            normal2 = np.array([math.cos((di+1)*math.pi/2.0+ang), math.sin((di+1)*math.pi/2.0+ang),0.0])
        return np.round(ds[di]-0.025, NUMDEC), np.round(np.dot(normal, vel), NUMDEC), np.round(np.dot(normal1, vel), NUMDEC), np.round(np.dot(normal2, vel), NUMDEC)
        
        
        
class WorldState(object):
    
    def __init__(self, transM = np.identity(4), invTrans = np.identity(4), ori = 0.0):
        self.objectStates = {}
        self.interactionStates = {}
        self.predictionCases = {}
        self.transM = transM
        self.invTrans = invTrans
        self.ori = 0.0
        pass        
            

    def parseModels(self, models):
        for m in models:
            if m.name == "ground_plane" or "wall" in m.name or "Shadow" in m.name:
                continue
            else:
                tmp = ObjectState()
                tmp["name"] = m.name
                tmp["id"][0] = m.id
                tmp["posX"][0] = np.round(m.pose.position.x, NUMDEC)
                tmp["posY"][0] = np.round(m.pose.position.y, NUMDEC)
                tmp["posZ"][0] = np.round(m.pose.position.z, 2)
                tmp["ori"][0] = np.round(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                            m.pose.orientation.z,m.pose.orientation.w])), NUMDEC)[2]
                tmp["linVelX"][0] = np.round(m.linVel.x, NUMDEC)
                tmp["linVelY"][0] = np.round(m.linVel.y, NUMDEC)
                tmp["linVelZ"][0] = 0.0 # np.round(m.linVel.z, NUMDEC)
#                print "linVelX of {}: {} ".format(m.name, tmp["linVelX"])
#                print "posX of {}: {} ".format(m.name, tmp["posX"])
#                print "norm linVel: ", np.linalg.norm(tmp["linVel"])
                if np.linalg.norm(tmp["linVel"]) < 0.01:
#                    print "setting linVel to 0"
                    tmp["linVelX"][0] = 0.0
                    tmp["linVelY"][0] = 0.0
                    tmp["linVelZ"][0] = 0.0
                if m.name == "blockA":
                    tmp["angVel"][0] = np.round(m.angVel.z, NUMDEC)
                    if self.transM == None:
                        if INTERACTION_STATES:
                            self.transM = common.eulerPosToTransformation(tmp["ori"],tmp["pos"])
                            self.invTrans = common.invertTransMatrix(self.transM)
                            self.ori = np.copy(tmp["ori"])
                        else:
                            self.transM = np.identity(4)
                            self.invTrans = np.identity(4)     
                            self.ori = 0
                self.objectStates[m.name] = tmp
                
    def parseContacts(self, contacts):
        curTime = (contacts.time.sec*1e9 + contacts.time.nsec) * 1e-9
        for c in contacts.contact[-1:]:
            o1Name = c.wrench[0].body_1_name.split(':')[0]
            o2Name = c.wrench[0].body_2_name.split(':')[0]
            cTime = (c.time.sec*1e9 + c.time.nsec) * 1e-9
            if np.abs(curTime-cTime) < 0.02:
                if self.objectStates.has_key(o1Name):
                    self.objectStates[o1Name]["contact"] = o2Name
                if self.objectStates.has_key(o2Name):
                    self.objectStates[o2Name]["contact"] = o1Name
    
                
    def parseInteractions(self):
        if SINGLE_INTSTATE:
            for n1, os1 in self.objectStates.items():
                if n1 != "gripper":
                    for n2, os2 in self.objectStates.items():
                        if n1 != n2:
                            intState = InteractionState.parseInteraction(os1,os2.transform(self.invTrans, -self.ori))
                            self.interactionStates[intState["name"]] = intState
        else:
            for n1, os1 in self.objectStates.items():
                for n2, os2 in self.objectStates.items():
                    if n1 != n2:
                        intState = InteractionState.parseInteraction(os1,os2.transform(self.invTrans, -self.ori))
                        self.interactionStates[intState["name"]] = intState
                    
                    
   
            
    def parse(self, gzWS):
        self.parseModels(gzWS.model_v.models)
        self.parseContacts(gzWS.contacts)
#        print "parsing"
        self.parseInteractions()


    def getInteractionState(self, name):
        return self.interactionStates[name]
        
    def getObjectState(self, name):
        if name in self.objectStates:
            return self.objectStates[name]
        else:
            return None 
                
    def updateObjectStates(self):
        for intState in self.interactionStates.values():
            o1,o2 = ObjectState.fromInteractionState(intState)
            if DIFFERENCES:
                #Transform back to global coordinate system
                o1 = o1.transform(self.transM, self.ori)
                o2 = o2.transform(self.transM, self.ori)
            self.objectStates[o1["name"]] = o1
            if SINGLE_INTSTATE:
                self.objectStates[o2["name"]] = o2
        
    def addInteractionState(self, intState, usedCase = None):
        self.interactionStates[intState["name"]] = intState
        self.predictionCases[intState["name"]] = usedCase
#        if not intState["sname"] in self.objectStates:
#            self.objectStates[intState["sname"]], self.objectStates[intState["oname"]] = ObjectState.fromInteractionState(intState)
        
    def addObjectState(self, os):
        self.objectStates[os["name"]] = os
        
    def getSelfInteractionStates(self, sname):
        return self.getInteractionStates(sname)
        
    def getOtherInteractionStates(self, oname):
        return [intState for intState in self.interactionStates.values() if intState["oname"] == oname]
        
    def getInteractionStates(self, sname):
        return [intState for intState in self.interactionStates.values() if intState["sname"] == sname]
            