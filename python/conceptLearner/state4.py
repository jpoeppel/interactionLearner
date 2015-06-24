#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:40:03 2015

@author: jpoeppel
"""

import numpy as np
import common
from common import NUMDEC
import math

WIDTH = {"blockA":0.25} #Width from the middle point
HEIGHT = {"blockA":0.05} #Height from the middle point

class State(dict):
    
    def __init__(self):
        self.vec = np.array([]) #Np.array representing the state vector
        self.mask = np.array(range(len(self.vec))) #Np.array to mask the access to the vector
        self.relKeys = self.keys()
        
    def getVec(self, mask=None):
        if mask:
            return self.vec[mask]
        else:
            return self.vec[self.mask]
            
    def score(self, other):
        return np.exp(-0.5*np.linalg.norm(self.getVec()-other.getVec()))
        
    
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
        self.actionItems = ["linVel", "angVel"]
        
class InteractionState(State):
    
    def __init__(self):
        self.vec = np.zeros(9)
        self.update({"name": "", "sname":"", "oname": "", "sid": self.vec[0:1], "oid":self.vec[1:2],
                     "dist": self.vec[2:3], "closing": self.vec[3:4], "contact":self.vec[4:5],
                     "relPosX": self.vec[5:6], "relPosY": self.vec[6:7], "relPosZ":self.vec[7:8],
                     "closingDivDist":self.vec[8:9], "relPos": self.vec[5:8] })
        self.mask = np.array(range(len(self.vec)))

        
class WorldState(object):
    
    def __init__(self):
        self.objectStates = {}
        self.interactionStates = {}
        self.transM = np.identity(4)
        self.invTrans = np.identity(4)
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
                tmp["posZ"][0] = np.round(m.pose.position.z, NUMDEC)
                tmp["ori"][0] = np.round(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                            m.pose.orientation.z,m.pose.orientation.w])), NUMDEC)[2]
                tmp["linVelX"][0] = np.round(m.linVel.x, NUMDEC)
                tmp["linVelY"][0] = np.round(m.linVel.y, NUMDEC)
                tmp["linVelZ"][0] = np.round(m.linVel.z, NUMDEC)
                tmp["angVel"][0] = np.round(m.angVel.z, NUMDEC)
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
        for n1, os1 in self.objectStates.items():
            for n2, os2 in self.objectStates.items():
                if n1 != n2:
                    intState = InteractionState()
                    intState["name"] = n1+n2
                    intState["sname"] = n1
                    intState["oname"] = n2
                    intState["sid"][0] = os1["id"]
                    intState["oid"][0] = os2["id"]
                    intState["dist"][0], intState["closing"][0] = self.computeDistanceClosing(os1,os2)
                    if os1["contact"] == n2:
                        intState["contact"][0] = 1
                    intState["relPosX"][0] = os1["posX"]-os2["posX"]
                    intState["relPosY"][0] = os1["posY"]-os2["posY"]
                    intState["relPosZ"][0] = os1["posZ"]-os2["posZ"]
                    intState["closingDivDist"][0] = intState["closing"]/intState["dist"]
                    self.interactionStates[intState["name"]] = intState
            
            
    def parse(self, gzWS):
        self.parseModels(gzWS.model_v.models)
        self.parseContacts(gzWS.contacts)
        self.parseInteractions()
        
    def calcDist(self, x0x,x0y,mp, x1x,x1y,x2x,x2y,c,s):
        x1xn = x1x*c - x1y*s
        x1yn = x1x*s + x1y*c
        x2xn = x2x*c - x2y*s
        x2yn = x2x*s + x2y*c
        x1n = np.array([x1xn,x1yn]) + mp
        x2n = np.array([x2xn,x2yn]) + mp
        if x0x <= x2x and x0x >= x1x: 
            d = abs((x2n[0]-x1n[0])*(x1n[1]-x0y)-(x1n[0]-x0x)*(x2n[1]-x1n[1]))/math.sqrt((x2n[0]-x1n[0])**2+(x2n[1]-x1n[1])**2) - 0.025
        else:
            d = min(np.linalg.norm(x1n-np.array([x0x,x0y])), np.linalg.norm(x2n-np.array([x0x,x0y])))
        return d
        
    def computeDistanceClosing(self, os1, os2):
        if os1["name"] == "gripper":
            x0x,x0y = os1["pos"][:2]
            mp = np.copy(os2["pos"][:2])
            ang = os2["ori"]
            blockN = os2["name"]
            vel = os2["linVel"]
        elif os2["name"] == "gripper":
            x0x,x0y = os2["pos"][:2]
            mp = np.copy(os1["pos"][:2])
            ang = os1["ori"]
            blockN = os1["name"]
            vel = os1["linVel"]
        else:
            raise NotImplementedError("Currently only distance between gripper and object is implemented.")
        c = np.cos(ang)
        s = np.sin(ang)
        
        x1x = WIDTH[blockN]
        x1y = HEIGHT[blockN]
        x2x = x1x
        x2y = -x1y
        d1 = self.calcDist(x0x,x0y,mp,x1x,x1y,x2x,x2y,c,s)
        
        x1x = WIDTH[blockN]
        x1y = HEIGHT[blockN]
        x2x = -x1x
        x2y = x1y
        d2 = self.calcDist(x0x,x0y,mp,x1x,x1y,x2x,x2y,c,s)
        
        x1x = -WIDTH[blockN]
        x1y = HEIGHT[blockN]
        x2x = x1x
        x2y = -x1y
        d3 = self.calcDist(x0x,x0y,mp,x1x,x1y,x2x,x2y,c,s)
        
        x1x = WIDTH[blockN]
        x1y = -HEIGHT[blockN]
        x2x = -x1x
        x2y = x1y
        d4 = self.calcDist(x0x,x0y,mp,x1x,x1y,x2x,x2y,c,s)
        
        ds = [d1,d2,d3,d4]
        di = np.argmin(ds)
        normal = np.array([np.cos(ang+di*(math.pi/2.0)), np.sin(ang+di*(math.pi/2.0)), 0])
        return ds[di], np.dot(normal, vel)
        
    def getInteractionState(self, name):
        return self.interactionStates[name]
        
    def getObjectState(self, name):
        return self.objectStates[name]
        
    def addObjectState(self, os):
        self.objectStates[os["name"]] = os
            