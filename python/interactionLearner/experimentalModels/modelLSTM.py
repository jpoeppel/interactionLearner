#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:07:35 2015

@author: jpoeppel
"""

import ocrolib.lstm as lstm
import numpy as np

NUM_MEM_UNITS = 200
INPUT_DIM = 32
OUTPUT_DIM = 28

L_RATE = 0.1
MOMENTUM = 0.9

class ModelLSTM(object):
    
    def __init__(self, inputFile=None, outputFile=None):
        self.net = lstm.LSTM1(INPUT_DIM,NUM_MEM_UNITS,OUTPUT_DIM)
        self.net.setLearningRate(L_RATE, MOMENTUM)
        npInputSeqs = []
        npOutputSeqs = []
        if inputFile != None and outputFile != None:
            seq = []
            for line in open(inputFile, 'r'):                
                if line and not line.startswith('#'):
                    seq.append(np.array(line.split()).astype(float))
                else:
                    npInputSeqs.append(seq)
                    seq = []
                    
            seq = []
            for line in open(outputFile, 'r'):                
                if line and not line.startswith('#'):
                    seq.append(np.array(line.split()).astype(float))
                else:
                    npInputSeqs.append(seq)
                    seq = []
            print "input seqs: ", npInputSeqs
            xs = np.array(npInputSeqs)
            ys = np.array(npOutputSeqs)
            print "shape xs: ", np.shape(xs)
            for i in range(20):
                self.update(xs,ys)

    def predict(self, xs):
        """
        Function to query the model
        
        Parameters
        ----------
        xs: array numSeqElems x inDims
            Input sequence, can be one element
        """
        return np.round(self.net.predict(xs),4)
                
        
    def update(self, xs, ys):
        """
        Function to update the LSTM
        
        Parameters
        ----------
        
        xs: array numSeqElems x inDims
            Input sequence
        ys: array numSeqElems x outDims
            Target output sequence
        
        """
        preds = self.net.train(xs,ys)