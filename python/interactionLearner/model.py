# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:24:53 2015

@author: jpoeppel
"""

import numpy as np
import gazeboInterface as gi


class State(object):
    
    def __init__(self, worldState):
        self.gripperState = worldState.gripper.toDict()
        self.objectStates = [o.toDict() for o in worldState.objects]
        self.importantGripperAttribs = self.gripperState.keys()
        if self.objectStates:
            self.importantObjectAttribs = self.objectStates[0].keys() 
        else:
            self.importantObjectAttribs = []
            
    def score(self, otherState):
        otherGripperState = otherState.gripper.toDict()
        otherObjectStates = [o.toDict() for o in otherState.objects]
        s = 0.0
        for k in self.importantGripperAttribs:
            tmp = self.gripperState[k]
            if k == "linVel":
                s += abs(tmp.dot(otherGripperState[k])/(np.linalg.norm(tmp)*np.linalg.norm(otherGripperState[k])))
            elif type(tmp) == int:
                s += 1 if tmp == otherGripperState[k] else 0
            #Todo compare                    

        #TODO other object scores
#        for k in self.importantObjectAttribs:
#            for o in otherObjectStates:
#                k
            #Todo compare objects
            pass
        return s

class Case(object):
    
    def __init__(self, name, initState, action, endState):
        self.id = name
        self.initialState = State(initState)
        self.action = action
        self.resultState = State(endState)
        #Remove attributes with no change
        toRem = []
        for k in self.initialState.importantGripperAttribs:
            if np.array_equal(self.initialState.gripperState[k], self.resultState.gripperState[k]):
                del self.initialState.gripperState[k]
                del self.resultState.gripperState[k]
                toRem.append(k)
        for k in toRem:
            self.initialState.importantGripperAttribs.remove(k)
            self.resultState.importantGripperAttribs.remove(k)
        
        
    def score(self, state, action):
        return self.initialState.score(state) + self.action.score(action)


class ModelCBR(object):
    """
        A simple case-base reasoning model. Creates cases and stores them for
        future generalisation.
    """
    
    
    def __init__(self):
        self.caseStore = []
        self.numCases = 0
        pass
    
    def searchCases(self, state,action):
        bestCase = None
        bestScore = 0.0
        for c in self.caseStore:
            tmpScore = c.score(state, action)
            if tmpScore > bestScore:
                bestCase = c
                bestScore = tmpScore
                
        return bestCase
    
    def predict(self, action, state):
        """
            Predict the next world state based on the current state of the world
            and the choosen action.
            
            Parameters
            ----------
            action: 
                The chosen action that the robot should execute
            state:
                The current world state.
            
            Returns
            -------
            
                The predicted world state.
        """
        similarCase = self.searchCases(state,action)
        if similarCase == None:
            #If no similar case was found, say the case does not change
            print "no similar case"
            return state
        else:
            state.gripper.pose[:3] += state.gripper.linVel
            print "prediction: " + str(state)
            return state
    
    def update(self, action, state, prediction, result):
        """
            Update the model according to the actual results.
            
            Parameters
            ----------
            action: 
                The command that was used last
            state:
                The state the world was in before the action.
            prediction:
                The predicted next world state.
            result:
                The state the world was in after the action.
        """
        print "prediction: "+ str(prediction)
        print "result: " + str(result)
        if prediction == result:
            #All good, we do not need to train
            pass
        else:
            self.caseStore.append(Case("Case" + str(self.numCases+1), state, action, result))
            self.numCases += 1
            
            
    def getAction(self, state):
        rnd = np.random.rand()
        if rnd < 0.5:
            return gi.Action(direction=np.random.rand(3)*2-1)
        elif rnd < 0.7:
            return None
        else:
            return gi.Action()
    
class ModelHMM(object):
    def __init__(self):
        
        pass
    
    def predict(self, action, state):
        """
            Predict the next world state based on the current state of the world
            and the choosen action.
            
            Parameters
            ----------
            action: 
                The chosen action that the robot should execute
            state:
                The current world state.
            
            Returns
            -------
            
                The predicted world state.
        """
        
        pass
    
    def update(self, action, state, result):
        """
            Update the model according to the actual results.
            
            Parameters
            ----------
            action: 
                The command that was used last
            state:
                The state the world was in before the action.
            result:
                The state the world was in after the action.
        """
        
        pass
   