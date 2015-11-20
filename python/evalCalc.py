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

directory = "../evalData/"
#directory = "./"

fileList = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]

ysPos = []
ysOri = []
errorsPos = []
errorsOri = []

ABSOLUT_ORIS = True


#testPosValues = {}
experiments = {}

currentExperiment = None

for f in sorted(fileList, key=natSort):
    
    if f.startswith(".") or "_config" in f or "ITMInformation" in f or "old" in f:
        continue
#    if "Configuration_40" in f or "Configuration_8" in f:
#        continue
    if not "interaction" in f:
        continue
#    if not "8_E11Tests" in f:
#        continue
#    if "Symmetric" in f:
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
    currentExperiment.trainStartPosX[numTrainRuns] = np.copy(trainData[firstTrainFrames,12])
    
    testrows = np.invert(trainrows)
    testData = data[testrows,:]
    testDifs = np.copy(testData[:,:14])
    testDifs[:,5:14] -= testData[:,15:]
    
    firstFrames = testData[:,3] == 0
    
    lastFrames = np.roll(firstFrames, -1) #TODO Test if this is always correct!    

#    startingPositionsX = testData[firstFrames,12]    
#    currentExperiment.testStartPosX[numTrainRuns] = testData[firstFrames,12] 
    
    #Consider filtering, i.e. only last frame
    actDifs = testDifs[lastFrames,12:14]
    oriDifs = testDifs[lastFrames,11]
    posDifs = testDifs[lastFrames,5:7]
    keyPoint1Difs = testDifs[lastFrames,7:9]
    keyPoint2Difs = testDifs[lastFrames,9:11]
    
    numTestRuns = int(np.max(testData[:,2]))+1
    testPosRes = {}
    startingTestPos = np.zeros(numTestRuns)
    #Consider each testpos separetly
    for i in xrange(numTestRuns):
        iMask = testData[lastFrames,2] == i
        posDifsTmp = posDifs[iMask]
        if ABSOLUT_ORIS:
            oriDifsTmp = np.abs(oriDifs[iMask])
        else:
            oriDifsTmp = oriDifs[iMask]
        testPosRes[i] = (np.mean(np.linalg.norm(posDifsTmp, axis=1)), np.mean(oriDifsTmp), 
                        np.std(np.linalg.norm(posDifsTmp, axis=1)), np.std(oriDifsTmp))
        startPosMask = testData[:,2] == i
        startingTestPos[i] = np.mean(testData[firstFrames*startPosMask,12])
        
    currentExperiment.testPosValues[numTrainRuns] = testPosRes    
    currentExperiment.testStartPosX[numTrainRuns] = startingTestPos
    
    
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
    resDict = e.testPosValues[e.trainRunOrder[rowI]]
    xs = e.testStartPosX[e.trainRunOrder[rowI]]
    ysPos = [resDict[j][0] for j in xrange(len(resDict))]
    errorsPos = [resDict[j][2] for j in xrange(len(resDict))]  
    ysOri = [resDict[j][1] for j in xrange(len(resDict))]
    errorsOri = [resDict[j][3] for j in xrange(len(resDict))]#
    row[0].axhline(y=0, color ='lightgrey')
    row[1].axhline(y=0, color ='lightgrey')
    for p in e.trainStartPosX[e.trainRunOrder[rowI]]:
        row[0].axvline(x=p, color='black')
        row[1].axvline(x=p, color='black')
    row[0].errorbar(xs, ysPos, yerr = errorsPos, fmt='o')
    row[1].errorbar(xs, ysOri, yerr = errorsOri, fmt='o')
    row[0].set_title("Positional")
    row[0].set_ylabel("Difference [m]")
    row[1].set_title("Orientation")
    row[1].set_ylabel("Difference [rad]")
    

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
    
def learnCurve(e):
    fig, row = plt.subplots(2,1, sharex=True)
    xs = e.trainRunOrder
    ysPos = []
    ysOri = []
    errorsPos = []
    errorsOri = []
    for trainRun in e.trainRunOrder:
        resDict = e.testPosValues[trainRun]
        ysPos.append(np.mean([resDict[j][0] for j in xrange(len(resDict))]))
        errorsPos.append(np.mean([resDict[j][2] for j in xrange(len(resDict))])) 
        ysOri.append(np.mean([resDict[j][1] for j in xrange(len(resDict))]))
        errorsOri.append(np.mean([resDict[j][3] for j in xrange(len(resDict))]))
    row[0].errorbar(xs, ysPos, yerr= errorsPos, fmt='o')
    row[1].errorbar(xs, ysOri, yerr = errorsOri, fmt='o')
    row[0].set_title("Position")
    row[1].set_title("Orientation")
    row[0].set_xlim(0,35)
    row[1].set_xlim(0,35)
#    row[0].set_xlabel("Number of training examples")
    row[0].set_ylabel("Difference [m]")
#    row[1].set_xlabel("Number of training examples")
    row[1].set_ylabel("Difference [rad]")
#    fig.suptitle("Learn curve for experiment " + e.name)
    plt.xlabel("Number of training examples")

for e in experiments.values():


#    pp = PdfPages("../pdfs/"+ e.name +".pdf")
    
    learnCurve(e)
#    eachTestPosSep(e)
    
    plt.tight_layout()
    plt.savefig("../pdfs/LearningCurve.pdf", 
            #This is simple recomendation for publication plots
            dpi=1000, 
            # Plot will be occupy a maximum of available space
            bbox_inches='tight', 
            )
#    pp.savefig()
#    pp.close()
    
plt.show()