#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:15:35 2015
Simple script to read collected data and produce suitable plots.
@author: jpoeppel
"""
#
import numpy as np
import os
import re

def natSort(s, _nsr=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsr, s)]

class Experiment(object):
    
    def __init__(self):
        self.name = ""
        self.testPosValues = {}
        self.trainRunOrder = []
        self.trainStartPosX = {}
        self.testStartPosX = {}
        self.testActValues = {}
        
        self.testPosValues2 = {}
        
        
class PushTaskExperiment(object):
    
    def __init__(self):
        self.name = ""
        self.numSteps = {}
        self.positionalErros = {}
        self.orientationErros = {}
        self.combinedError = {}

directory = "../evalData/"
#directory = "./"

fileList = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]

ysPos = []
ysOri = []
errorsPos = []
errorsOri = []

ABSOLUT_ORIS = True


#testPosValues = {}


def readMoveTaskData():
    print "read move to task"
    
    currentExperiment = None
    experiments = {}
    
    targets = {1: np.array([0.6,0.7, -2.1]), 2: np.array([-0.6,0.7,0.75]), 
               3:np.array([-0.6,-0.7,0]), 5:np.array([0.6,-0.7,np.pi])}
    
    for f in sorted(fileList, key=natSort):
        if f.startswith(".") or "_config" in f or "ITMInformation" in f or not "Mode4" in f:
            continue
        
        if not "interaction" in f:
            continue
#
        if not "5_TrainRuns" in f:
            continue
        if not "Sigma005" in f:
            continue
        if "V2" in f:
            continue
        
        nameOnly = f.replace('.txt','')
        parts = nameOnly.split("_")
        modelName = parts[0]
        targetConfig = int(parts[parts.index("TrainRuns")-1])
        currentConfig = int(parts[parts.index("Configuration")+1])
        if "_E" in f:
            experimentSuffix = parts[-1]
        else:
            experimentSuffix = ""
        experimentName = modelName+"_C_"+str(currentConfig)+"_Target_"+str(targetConfig)+experimentSuffix
        if currentExperiment == None or experimentName != currentExperiment.name:
            if experimentName in experiments:
                currentExperiment = experiments[experimentName]
            else:
                currentExperiment = PushTaskExperiment()
                currentExperiment.name = experimentName
                experiments[experimentName] = currentExperiment
            
        #load data
        print "filename: ", directory+f
        data = np.loadtxt(directory+f, delimiter = ';')
        
        testrows = data[:,1] == 0
        testData = data[testrows,:]
        firstFrames = testData[:,3] == 0
        
        lastFrames = np.roll(firstFrames, -1)
        currentExperiment.numSteps = {testData[lastFrames,0][i]:testData[lastFrames,3][i] for i in range(5)}
        currentExperiment.positionalErros = {testData[lastFrames,0][i]: testData[testData[:,0]==i,5:7]-targets[targetConfig][:2] for i in range(5)}
        currentExperiment.orientationErros = {testData[lastFrames,0][i]: testData[testData[:,0]==i,11]-targets[targetConfig][2] for i in range(5)}

        for k in currentExperiment.orientationErros.keys():
            currentExperiment.orientationErros[k][currentExperiment.orientationErros[k] > np.pi] -= 2*np.pi
            currentExperiment.orientationErros[k][currentExperiment.orientationErros[k] < -np.pi] += 2*np.pi
            
            combinedError = np.copy(currentExperiment.positionalErros[k])
            combinedError= np.insert(combinedError,np.shape(combinedError)[1],currentExperiment.orientationErros[k], axis=1)
            currentExperiment.combinedError[k] =  np.linalg.norm(combinedError,axis=1)
        
    return experiments



def readPushTaskData():
    
    currentExperiment = None
    experiments = {}
    
    for f in sorted(fileList, key=natSort):
        if f.startswith(".") or "_config" in f or "ITMInformation" in f or not "Mode2" in f:
            continue
        if not "gate" in f:
            continue
#        if not "Configuration_0_E20FoldsAll" in f:
#            continue
        if not "2Objects" in f:
            continue
#        if not "Sigma005" in f:
#            continue
#        if not "E20Folds" in f:
#            continue
    #    if "Start0" in f:
    #        continue
        nameOnly = f.replace('.txt','')
        parts = nameOnly.split("_")
        modelName = parts[0]
        numTrainRuns = int(parts[parts.index("TrainRuns")-1])
        currentConfig = int(parts[parts.index("Configuration")+1])
        if "_E" in f:
            experimentSuffix = parts[-1]
        else:
            experimentSuffix = ""
        experimentName = modelName+"_C_"+str(currentConfig)+experimentSuffix
        if currentExperiment == None or experimentName != currentExperiment.name:
            if experimentName in experiments:
                currentExperiment = experiments[experimentName]
            else:
                currentExperiment = Experiment()
                currentExperiment.name = experimentName
                experiments[experimentName] = currentExperiment
            
    #    trainRunOrder.append(numTrainRuns)
        currentExperiment.trainRunOrder.append(numTrainRuns)
        #load data
        print "filename: ", directory+f
        data = np.loadtxt(directory+f, delimiter = ';')
        
        trainrows = data[:,1] == 1
        
        trainData = data[trainrows,:]
        firstTrainFrames = trainData[:,3] == 0
    #    startingTrainPositionsX = trainData[firstTrainFrames,12]
#        currentExperiment.trainStartPosX[numTrainRuns] = np.copy(trainData[firstTrainFrames,12])
        currentExperiment.trainStartPosX[numTrainRuns] = np.copy(trainData[firstTrainFrames,20])
        
        testrows = np.invert(trainrows)
        testData = data[testrows,:]
#        testDifs = np.copy(testData[:,:14])
#        testDifs[:,5:14] -= testData[:,15:]
        
        testDifs = np.copy(testData[:,:22])
        testDifs[:,4:] -= testData[:,22:]
        
        firstFrames = testData[:,3] == 0
        
        lastFrames = np.roll(firstFrames, -1) #TODO Test if this is always correct!    
    
    #    startingPositionsX = testData[firstFrames,12]    
    #    currentExperiment.testStartPosX[numTrainRuns] = testData[firstFrames,12] 
        
        #Consider filtering, i.e. only last frame
#        actDifs = testDifs[lastFrames,12:14]
#        oriDifs = testDifs[lastFrames,11]
#        posDifs = testDifs[lastFrames,5:7]
        
        actDifs = testDifs[lastFrames,20:22]
        oriDifs = testDifs[lastFrames,11]
        posDifs = testDifs[lastFrames,5:7]
        oriDifs2 = testDifs[lastFrames,19]
        posDifs2 = testDifs[lastFrames,13:15]
        
        
        numTestRuns = int(np.max(testData[:,2]))+1
        testPosRes = {}
        testPosRes2 = {}
        tmpActDifs = {}
        startingTestPos = np.zeros(numTestRuns)
        #Consider each testpos separetly
        for i in xrange(numTestRuns):
            iMask = testData[lastFrames,2] == i
            posDifsTmp = posDifs[iMask]
            posDifsTmp2 = posDifs2[iMask]
            if ABSOLUT_ORIS:
                oriDifsTmp = np.abs(oriDifs[iMask])
                oriDifsTmp2 = np.abs(oriDifs2[iMask])
            else:
                oriDifsTmp = oriDifs[iMask]
                oriDifsTmp2 = oriDifs2[iMask]
            testPosRes[i] = (np.mean(np.linalg.norm(posDifsTmp, axis=1)), np.mean(oriDifsTmp), 
                            np.std(np.linalg.norm(posDifsTmp, axis=1)), np.std(oriDifsTmp))
            
            testPosRes2[i] = (np.mean(np.linalg.norm(posDifsTmp2, axis=1)), np.mean(oriDifsTmp2), 
                            np.std(np.linalg.norm(posDifsTmp2, axis=1)), np.std(oriDifsTmp2))
            tmpActDifs[i] = (np.mean(np.linalg.norm(actDifs[iMask],axis=1)), np.std(np.linalg.norm(actDifs[iMask],axis=1)))
            startPosMask = testData[:,2] == i
#            startingTestPos[i] = np.mean(testData[firstFrames*startPosMask,12])
            startingTestPos[i] = np.mean(testData[firstFrames*startPosMask,20])
            

        currentExperiment.testPosValues2[numTrainRuns] = testPosRes2       
        currentExperiment.testPosValues[numTrainRuns] = testPosRes    
        currentExperiment.testStartPosX[numTrainRuns] = startingTestPos
        currentExperiment.testActValues[numTrainRuns] = tmpActDifs
        
    return experiments
#    ysPos.append(np.mean(np.linalg.norm(posDifs, axis=1)))
#    ysOri.append(np.mean(oriDifs))
#    errorsPos.append(np.std(np.linalg.norm(posDifs, axis=1)))
#    errorsOri.append(np.std(oriDifs))
    
#    print "Starting pos: ", startingTrainPositionsX
    
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#pp = PdfPages("../gateEvaConfig12.pdf")
#
#
#fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(15,6))
#ax0.axhline(y=0, color ='lightgrey')
#ax0.errorbar(xs,ysOri, yerr=errorsOri, fmt='o')
#ax0.set_title('Mean orientational differences at the end of a run and their standard deviation.')
#ax1.axhline(y=0, color ='lightgrey')
#ax1.errorbar(xs,ysPos, yerr=errorsPos, fmt='o')
#ax1.set_title('Mean positional differences at the end of a run and their standard deviation.')
#ax1.set_xlim([0,35])

#pp.savefig()
#pp.close()


def plotRow(e, row, rowI):
    
    xs = e.testStartPosX[e.trainRunOrder[rowI]]
    resDict = e.testPosValues2[e.trainRunOrder[rowI]]
    
    ysPos = [resDict[j][0] for j in xrange(len(resDict))]
    errorsPos = [resDict[j][2] for j in xrange(len(resDict))]  
    ysOri = [resDict[j][1] for j in xrange(len(resDict))]
    errorsOri = [resDict[j][3] for j in xrange(len(resDict))]#
    row[0].axhline(y=0, color ='lightgrey')
    row[1].axhline(y=0, color ='lightgrey')
    for p in e.trainStartPosX[e.trainRunOrder[rowI]]:
        row[0].axvline(x=p, color='black')
        row[1].axvline(x=p, color='black')
    row[0].errorbar(xs, ysPos, yerr = errorsPos, fmt='o', elinewidth=2, label="Blue block")
    row[1].errorbar(xs, ysOri, yerr = errorsOri, fmt='o', elinewidth=2, label="Blue block")
    
    
    resDict = e.testPosValues[e.trainRunOrder[rowI]]
    ysPos2 = [resDict[j][0] for j in xrange(len(resDict))]
    errorsPos2 = [resDict[j][2] for j in xrange(len(resDict))]  
    ysOri2 = [resDict[j][1] for j in xrange(len(resDict))]
    errorsOri2 = [resDict[j][3] for j in xrange(len(resDict))]#
    row[0].errorbar(xs, ysPos2, yerr = errorsPos2, fmt='or', label="Red block")
    row[1].errorbar(xs, ysOri2, yerr = errorsOri2, fmt='or', label="Red block")
    
    row[0].set_title("Position")
    row[0].set_ylabel("Difference [m]")
    row[1].set_title("Orientation")
    row[1].set_ylabel("Difference [rad]")
    
    row[0].set_xlim(-1,0.6)
    row[1].set_xlim(-1,0.6)
    plt.legend(bbox_to_anchor=(0, 2.2,1, .0))
    print e.name
    print "ysPos: ", ysPos
    print "errorsPos: ", errorsPos
    print "ysOri: ", ysOri
    print "errorsOri: ", errorsOri
    

def eachTestPosSep(e):
    numSubPlotRows = len(e.testPosValues)
#    numSubPlotRows = 1
    fig, axes = plt.subplots(2, numSubPlotRows, sharex=True)
    if numSubPlotRows == 1:
        plotRow(e, axes, 0)
    else:
        i = 0
        for row in axes:
            plotRow(e, row, i)
            i+=1
    plt.xlabel("x position of testrun")
#    fig.subplots_adjust(hspace = 1)
#    fig.suptitle("Results for experiment " + e.name)
    
def learnCurve(e ,e2=None):
    fig, row = plt.subplots(3,1, sharex=True)
    xs = e.trainRunOrder
    ysPos = []
    ysOri = []
    ysAct = []
    errorsPos = []
    errorsOri = []
    errorsAct = []
    e2 = e
    if e2 != None:
        ysPos2 = []
        ysOri2 = []
        ysAct2 = []
        errorsPos2 = []
        errorsOri2 = []
        errorsAct2 = []
    for trainRun in e.trainRunOrder:
        resDict = e.testPosValues[trainRun]
#        print "resdict: ", resDict[20]
        actDict = e.testActValues[trainRun]
        testPos = range(len(resDict))
        testPos2 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#        testPos2 = [0,1,2,18,19,20]
#        testPos2 = range(len(resDict))
        ysPos.append(np.mean([resDict[j][0] for j in testPos]))
        errorsPos.append(np.mean([resDict[j][2] for j in testPos])) 
        ysAct.append(np.mean([actDict[j][0] for j in testPos]))
        errorsAct.append(np.mean([actDict[j][1] for j in testPos]))
        ysOri.append(np.mean([resDict[j][1] for j in testPos]))
        errorsOri.append(np.mean([resDict[j][3] for j in testPos]))
        if e2 != None:
            resDict = e2.testPosValues[trainRun]
            ysPos2.append(np.mean([resDict[j][0] for j in testPos2]))
            errorsPos2.append(np.mean([resDict[j][2] for j in testPos2])) 
            ysAct2.append(np.mean([actDict[j] for j in testPos2]))
            errorsAct2.append(np.mean([actDict[j][1] for j in testPos2]))
            ysOri2.append(np.mean([resDict[j][1] for j in testPos2]))
            errorsOri2.append(np.mean([resDict[j][3] for j in testPos2]))
    row[0].errorbar(xs, ysPos, yerr= errorsPos, fmt='ob', elinewidth=2, label="All 21 test positions")
    row[1].errorbar(xs, ysOri, yerr= errorsOri, fmt='ob', elinewidth=2, label="All 21 test positions")
    row[2].errorbar(xs, ysAct, yerr= errorsAct, fmt='ob', elinewidth=2, label="All 21 test positions" )
    if e2 != None:
        row[0].errorbar(xs, ysPos2, yerr= errorsPos2, fmt='or', label="Only interacting 15 test positions")
        row[1].errorbar(xs, ysOri2, yerr = errorsOri2, fmt='or', label="Only interacting 15 test positions")
        row[2].errorbar(xs, ysAct2, yerr= errorsAct2, fmt='or', label="Only interacting 15 test positions" )
    row[0].set_title("Block position")
    row[1].set_title("Block orientation")
    row[2].set_title("Actuator position")
    row[0].set_xlim(0,32)
    row[1].set_xlim(0,32)
#    row[0].set_xlabel("Number of training examples")
    row[0].set_ylabel("Difference [m]")
#    row[1].set_xlabel("Number of training examples")
    row[1].set_ylabel("Difference [rad]")
    row[2].set_ylabel("Difference [m]")
#    fig.suptitle("Learn curve for experiment " + e.name)
    plt.xlabel("Number of training examples")
    plt.legend(bbox_to_anchor=(0.0, 3.75,1, .0))
    print e.name
    print "ysPos: ", ysPos
    print "errorsPos: ", errorsPos
    print "ysOri: ", ysOri
    print "errorsOri: ", errorsOri
    print "ysAct: ", ysAct
    print "errorsAct: ", errorsAct
    print "ysPos2: ", ysPos2
    print "errorsPos2: ", errorsPos2
    print "ysOr2i: ", ysOri2
    print "errorsOri2: ", errorsOri2
    print "ysAct2: ", ysAct2
    print "errorsAct2: ", errorsAct2


def analysePushTask():
    experiments = readPushTaskData()
    for e in experiments.values():
    
#        learnCurve(e)
#    learnCurve(experiments.values()[0], experiments.values()[1])
        eachTestPosSep(e)
    
        plt.tight_layout()
        plt.savefig("../pdfs/EachPos" + e.name + ".pdf", 
                #This is simple recomendation for publication plots
                dpi=1000, 
                # Plot will be occupy a maximum of available space
                bbox_inches='tight', 
                )
    
    plt.show()
    
def analyseMoveToTarget():
    experiments = readMoveTaskData()
    
    for e in experiments.values():
        fig, rows = plt.subplots(3,1, sharex=True)
#        fig.suptitle(e.name.replace("_",""))
        i=0
#        for row in rows:
##            row.plot(e.combinedError[i])
#            row[0].plot(np.linalg.norm(e.positionalErros[i], axis=1))
#            row[1].plot(np.abs(e.orientationErros[i]))
#            i+= 1
        print "experiment name: ", e.name
        tmpSteps = np.zeros(5)
        tmpPos = np.zeros(5)
        tmpOri = np.zeros(5)
        tmpTotal = np.zeros(5)
        for k in e.numSteps.keys():
#            print "number of steps in fold {}: {}".format(k, e.numSteps[k])
            tmpSteps[k]=e.numSteps[k]
#            print "remaining error pos: {}".format(np.linalg.norm(e.positionalErros[k][-1]))
            tmpPos[k] =np.linalg.norm(e.positionalErros[k][-1])
#            print "remaining error ori: {}".format(np.linalg.norm(e.orientationErros[k][-1]))
            tmpOri[k] = np.abs(e.orientationErros[k][-1])
#            print "remaining total error: {}".format(e.combinedError[k][-1])
            tmpTotal[k] = e.combinedError[k][-1]
        print "Steps: ", tmpSteps
        print "pos errors: ", tmpPos
        print "ori erros: ", tmpOri
        print "total errors: ", tmpTotal
        print "mean stesp: ", np.mean(tmpSteps)
        print "mean steps reached: ", np.mean(tmpSteps[tmpSteps < 3000])
        print "mean Pos: ", np.mean(tmpPos)
        print "mean Pos not reached: ", np.mean(tmpPos[tmpSteps == 3000])
        print "mean Ori: ", np.mean(tmpOri)
        print "mean Ori not reached: ", np.mean(tmpOri[tmpSteps == 3000])
        print "mean total: ", np.mean(tmpTotal)
#        print len(e.positionalErros[0])
        labels = ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"]
        colors = ["b", "r", "#FF8000", 'g', 'k']
        cI = 0
        for i in [2,3,4]:
            rows[0].plot(np.linalg.norm(e.positionalErros[i], axis=1), label=labels[i], c=colors[cI])
            rows[1].plot(np.abs(e.orientationErros[i]), label=labels[i], c=colors[cI])
            rows[2].plot(e.combinedError[i], label=labels[i], c=colors[cI])
            cI += 1
        rows[0].set_ylabel("Position[m]")
        rows[1].set_ylabel("Orientation [rad]")
        rows[2].set_ylabel("Combined")
#        rows[0].plot(np.linalg.norm(e.positionalErros[2], axis=1))
#        rows[1].plot(np.abs(e.orientationErros[2]))
#        rows[2].plot(e.combinedError[2])
#        rows[0].plot(np.linalg.norm(e.positionalErros[3], axis=1))
#        rows[1].plot(np.abs(e.orientationErros[3]))
#        rows[2].plot(e.combinedError[3])
#        row[0].set_title("Position")
#        row[1].set_title("Orientation")
#        row[0].set_ylabel("Difference [m]")
#        row[1].set_ylabel("Difference [rad]")
#        plt.ylabel("Combined error of position and orientation")
        plt.xlabel("Number of interaction steps")
        plt.legend(bbox_to_anchor=(0.0, 3.6,1, .0), loc="center",
           ncol=5, mode="expand", borderaxespad=0.)
        
#        fig.text(0.05, 0.5, 'Combined error of position and orientation',
#            horizontalalignment='right',
#            verticalalignment='center',
#            rotation='vertical')
        
        plt.savefig("../pdfs/MoveToTargetInteractionT4Detail.pdf", 
                #This is simple recomendation for publication plots
                dpi=1000, 
                # Plot will be occupy a maximum of available space
                bbox_inches='tight', 
                )
    plt.show()
    
if __name__=="__main__":
    analysePushTask()
#    analyseMoveToTarget()