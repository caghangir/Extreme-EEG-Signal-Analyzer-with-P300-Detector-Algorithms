%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extreme EEG Signal Analyzer with P300 Detector           %
% Algorithms                                               %
%                                                          %
% Copyright (C) 2019 Cagatay Demirel. All rights reserved. %
%                    demirelc16@itu.edu.tr                 %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter, butter, hamming, lfilter, filtfilt, resample
from scipy.signal import lfilter, hamming, savgol_filter, hilbert, fftconvolve, butter, iirnotch, freqz, firwin, iirfilter
from struct import unpack
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model, svm, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
import random
import itertools
import pickle
import librosa
from scipy.fftpack import fft

def envelopeCreator(timeSignal, degree, fs):
    absoluteSignal = np.abs(hilbert(timeSignal))
    intervalLength = int(fs / 40 + 1) 
    amplitude_envelopeFiltered = savgol_filter(absoluteSignal, intervalLength, degree)
    return amplitude_envelopeFiltered  

def butter_bandpass(lowcut, highcut, fs, order=3): # 3 ten sonra lfilter NaN degerler vermeye basliyor
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
#    b, a = iirfilter(5, [low, high], rs=60, rp=60, btype='band', analog=False, ftype='cheby1')
    return b, a
    
def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# band-pass filter between two frequency     
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def FFT(signal, Fs):
    nFFT = len(signal) / 2
    nFFT = int(nFFT)
    #Hamming Window
    w = np.hamming(len(signal))
    #FFT
    X = abs(fft(signal * w))                                  # get fft magnitude
    X = X[0:nFFT]                                    # normalize fft
    X = X / len(X)
    
    fIndexes = (Fs / (2*nFFT)) * np.r_[0:nFFT] # [1,9] 9peet üretti
    
    return X, fIndexes
    
def hpssFilter(data):
    data = librosa.effects.hpss(data.astype("float64"), margin=(1.0,5.0))    
    return data

def binPower(signal, Band, Fs):
    nFFT = len(signal) / 2
    nFFT = int(nFFT)
    #Hamming Window
    w = np.hamming(len(signal))
    #FFT
    X = abs(fft(signal * w))                                  # get fft magnitude
    X = X[0:nFFT]                                    # normalize fft
    X = X / len(X)
    
    power = np.zeros(len(Band) - 1)
    for freq_index in range(0, len(Band) - 1):
        freq = Band[freq_index]
        nextFreq = Band[freq_index + 1]
        beginInd = int(np.floor(freq * len(signal) / Fs))
        endInd = int(np.floor(nextFreq * len(signal) / Fs))
        power[freq_index] = sum(X[beginInd:endInd])
    power_ratio = power / sum(power)
    return power, power_ratio

def pfd(X, D=None):
    """Compute Petrosian Fractal Dimension """"
    if D is None:
        D = np.diff(X)
        D = D.tolist()
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))
    
def hfd(X, Kmax):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter
    """
    L = []
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(np.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
        x.append([np.log(float(1) / k), 1])

    (p, r1, r2, s) = np.linalg.lstsq(x, L)
    return p[0]

def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series """

    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)  # Hjorth Mobility and Complexity

def hurst(X):
    """ Compute the Hurst exponent of X. 
    """
    X = np.array(X)
    N = X.size
    T = np.arange(1, N + 1)
    Y = np.cumsum(X)
    Ave_T = Y / T

    S_T = np.zeros(N)
    R_T = np.zeros(N)

    for i in range(N):
        S_T[i] = np.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = np.ptp(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = np.log(R_S)[1:]
    n = np.log(T)[1:]
    A = np.column_stack((n, np.ones(n.size)))
    [m, c] = np.linalg.lstsq(A, R_S)[0]
    H = m
    return H
    
def dfa(X, Ave=None, L=None):
    """Compute Detrended Fluctuation Analysis 
    """

    X = np.array(X)

    if Ave is None:
        Ave = np.mean(X)

    Y = np.cumsum(X)
    Y -= Ave

    if L is None:
        L = np.floor(len(X) * 1 / (2 ** np.array(list(range(4, int(np.log2(len(X))) - 4)))))

    F = np.zeros(len(L))  # F(n) of different given box length n

    for i in range(0, len(L)):
        n = int(L[i])                        # for each box length L[i]
        if n == 0:
            print("time series is too short while the box length is too big")
            print("abort")
            exit()
        for j in range(0, len(X), n):  # for each box
            if j + n < len(X):
                c = list(range(j, j + n))
                # coordinates of time in the box
                c = np.vstack([c, np.ones(n)]).T
                # the value of data in the box
                y = Y[j:j + n]
                # add residue in this box
                F[i] += np.linalg.lstsq(c, y)[1]
        F[i] /= ((len(X) / n) * n)
    F = np.sqrt(F)

    Alpha = np.linalg.lstsq(np.vstack([np.log(L), np.ones(len(L))]).T, np.log(F))[0][0]

    return Alpha
#================================================================================================================================    
    
def RecvData(socket, requestedSize):
    returnStream = ''
    while len(returnStream) < requestedSize:
        databytes = socket.recv(requestedSize - len(returnStream))
        if databytes == '':
            raise (RuntimeError, "connection broken")
        returnStream += databytes
 
    return returnStream 
    
def SplitString(raw):
    stringlist = []
    s = ""
    for i in range(len(raw)):
        if raw[i] != '\x00':
            s = s + raw[i]
        else:
            stringlist.append(s)
            s = ""

    return stringlist

    # read from tcpip socket
def GetProperties(rawdata):

    # Extract numerical data
    (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])

    # Extract resolutions
    resolutions = []
    for c in range(channelCount):
        index = 12 + c * 8
        restuple = unpack('<d', rawdata[index:index+8])
        resolutions.append(restuple[0])

    # Extract channel names
    channelNames = SplitString(rawdata[12 + 8 * channelCount:])

    return (channelCount, samplingInterval, resolutions, channelNames)
    
def GetData(rawdata, channelCount):

    # Extract numerical data
    (block, points, markerCount) = unpack('<LLL', rawdata[:12])

    # Extract eeg data as array of floats
    data = []
    for i in range(points * channelCount):
        index = 12 + 4 * i
        value = unpack('<f', rawdata[index:index+4])
        data.append(value[0])

    # Extract markers
    markers = []
    index = 12 + 4 * points * channelCount
    for m in range(markerCount):
        markersize = unpack('<L', rawdata[index:index+4])

        ma = Marker()
        (ma.position, ma.points, ma.channel) = unpack('<LLl', rawdata[index+4:index+16])
        typedesc = SplitString(rawdata[index+16:index+markersize[0]])
        ma.type = typedesc[0]
        ma.description = typedesc[1]

        markers.append(ma)
        index = index + markersize[0]

    return (block, points, markerCount, data, markers)

def distinguishData(oneBigSignal, channelCount, resolutions):
    i = 0
    sampleCount = 0  
    chunkAmount = 1
    dataSeparated = np.zeros((channelCount, int(len(oneBigSignal)/channelCount)))#
    while 1:
        for j in range(0,channelCount):
            dataSeparated[j,sampleCount:sampleCount+chunkAmount] = [k * resolutions[0] for k in oneBigSignal[i:i+chunkAmount]]
            i = i + chunkAmount
        sampleCount = sampleCount + chunkAmount   
        if(i >= len(oneBigSignal)):
            break
    return dataSeparated

def envelopeCreator(timeSignal, degree, Fs):
    absoluteSignal = np.abs(hilbert(timeSignal))
    intervalLength = int(Fs / 10 + 1) 
    amplitude_envelopeFiltered = savgol_filter(absoluteSignal, intervalLength, degree)
    return amplitude_envelopeFiltered
    
def notchFilter(data, Fs, f0, Q):
    w0 = f0/(Fs/2)
    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, data)
#    bp_stop_Hz = np.array([49.0, 51.0])
#    b, a = butter(2,bp_stop_Hz/(Fs / 2.0), 'bandstop')
#    w, h = freqz(b, a)
    return y

def eegFilteringOfflineEyeClosed(data, channelCount, stimulusLog, sampFreq, lowPass, highPass, deletionWindowAmount, order=3):
    dataSeparatedFilt = np.zeros((channelCount, len(data[0,:])))   
    for i in range(0,channelCount):    
        tempData = notchFilter(data[i], sampFreq, 50, 30)
        dataSeparatedFilt[i] = butter_bandpass_filter(tempData, lowPass, highPass, sampFreq, order)
    
    dataSeparatedFilt2 = dataSeparatedFilt[:,2500:]
   
    eegSignals = ([],[],[],[])
    for i in range(0, channelCount):
        eegSignals[i].append([])
        eegSignals[i].append([])
        eegSignals[i].append([])
        eegSignals[i].append([])
        eegSignals[i].append([])
 
    count = 0
    for i in range(0, len(stimulusLog)):
        for j in range(0, channelCount):
             if(count+300 > len(dataSeparatedFilt2[0,:])):
                 break
             eegSignals[j][stimulusLog[i]-1].append(dataSeparatedFilt2[j,count:count+300])
        count += 200
        
#    eegSignals = eegSignalDeletion(eegSignals, channelCount, deletionWindowAmount)

    return eegSignals

def eegFilteringOnlineEyeClose(data, channelCount, sampFreq, lowPass, highPass, correctionMs, order=3):
    correctionWindowAmount = int(correctionMs / 2)
    dataSeparatedFilt = np.zeros((channelCount, len(data[0,:])))
    for i in range(0,channelCount):    
        tempData = notchFilter(data[i], sampFreq, 50, 30)
        tempData = butter_bandpass_filter(data[i], lowPass, highPass, sampFreq, order)
        dataSeparatedFilt[i] = tempData - np.mean(tempData[0:correctionWindowAmount])
    return dataSeparatedFilt

def eegFilteringOnlineEyeOpen(data, channelCount, sampFreq, lowPass, highPass, correctionMs, order=3):
    correctionWindowAmount = int(correctionMs / 2)
    dataSeparatedFilt = np.zeros((channelCount, len(data[0,:])))
    for i in range(0,channelCount):    
#        tempData = notchFilter(data[i], sampFreq, 50, 30)
        tempData = butter_bandpass_filter(data[i], lowPass, highPass, sampFreq, order)
        dataSeparatedFilt[i] = tempData - np.mean(tempData[0:correctionWindowAmount])
    return dataSeparatedFilt

def eegFilteringOfflineEyeOpen(data, channelCount, sampFreq, lowPass, highPass, peakAmp, stimulusLog, deletionWindowAmount, 
                               order=3):
    dataSeparatedFilt = np.zeros((channelCount, len(data[0,:])))   
    for i in range(0,channelCount):    
        tempData = notchFilter(data[i], sampFreq, 50, 30)
        dataSeparatedFilt[i] = butter_bandpass_filter(tempData, lowPass, highPass, sampFreq, order)
    
    dataSeparatedFilt2 = dataSeparatedFilt[:,2500:]
    
    eegSignals = ([],[],[],[])
    for i in range(0, channelCount):
        eegSignals[i].append([])
        eegSignals[i].append([])
        eegSignals[i].append([])
        eegSignals[i].append([])
        eegSignals[i].append([])
    
    count = 0
    detectedBlinks = 0
    for i in range(0, len(stimulusLog)):
        if(count+300 > len(dataSeparatedFilt2[0,:])):
            break
        tempFp1Signal = dataSeparatedFilt2[0,count:count+300]
        if(np.max(tempFp1Signal) < 40000):            
            for j in range(0, channelCount):           
                eegSignals[j][stimulusLog[i]-1].append(dataSeparatedFilt2[j,count:count+300])
        else:
            detectedBlinks += 1
        count += 200        
        
#    eegSignals = eegSignalDeletion(eegSignals, channelCount, deletionWindowAmount)

    
    return eegSignals, dataSeparatedFilt2, detectedBlinks

#def eegSegmentedDataOfflineEyeClose(eegSignals)

def eegSignalDeletion(eegSignals, channelCount, deletionWindowAmount):
    for i in range(0, channelCount):
        for j in range(5):
            for k in range(deletionWindowAmount):
                del eegSignals[i][j][-1] 
    return eegSignals

def baselineCorrection(eegSignals, channelCount, correctionMs):
    correctionWindowAmount = int(correctionMs / 2)
    for i in range(channelCount):
        for j in range(3):
            for k in range(len(eegSignals[i][j])):
                eegSignals[i][j][k] = eegSignals[i][j][k] - np.mean(eegSignals[i][j][k][0:correctionWindowAmount])
    return eegSignals

def p300Creation(eegSignals, channelCount, windowMilliSecond):  
    windowSize = int(windowMilliSecond / 2)              
    p300Signals = np.zeros((5*channelCount, windowSize))
    stdWindows = np.zeros((channelCount * 5, windowSize))
    for i in range(channelCount):
        for j in range(5):        
            p300Signals[5*i+j] = np.mean(eegSignals[i][j], axis=0)  
            stdWindows[5*i+j] = np.std(eegSignals[i][j], axis=0)
    return p300Signals, stdWindows

def dataSeparationFromRAW(data, channelCount, resolutions):
    i = 0
    sampleCount = 0  
    chunkAmount = 1
    dataSeparated = np.zeros((channelCount, int(len(data)/channelCount)))
    while 1:
        for j in range(0,channelCount):
            dataSeparated[j,sampleCount:sampleCount+chunkAmount] = [k * resolutions[0] for k in data[i:i+chunkAmount]]
            i = i + chunkAmount
        sampleCount = sampleCount + chunkAmount   
        if(i >= len(data)):
            break
    return dataSeparated

def segmentedEEGSignalsP300(eegSignals, channelCount, setDirectory, expNo):
    os.chdir(setDirectory)
    windowMilliSecond = 600
    windowSize = windowMilliSecond / 2              
    stimulusAmount = 3    
    
    foundStimulus = np.zeros((channelCount))
    strings = ['Fp1', 'Fz', 'Cz', 'Pz']
    for i in range(channelCount):
        p300Signals = np.zeros((3, windowSize))
        for j in range(stimulusAmount):    
            p300Signals[j] = np.mean(eegSignals[i][j], axis=0)        
        foundStimulus[i] = P300FinderAlgorithmSTD(p300Signals)
         
        xAxis = np.arange(0, 599, 2)
        plt.figure()
        for j in range(stimulusAmount):
            plt.plot(xAxis, p300Signals[j], label=[j])
            plt.ylabel('Amplitude [uV]', fontsize=20)
            plt.xlabel('Time [Ms]', fontsize=20)
            plt.legend(loc='upper right', fontsize=10)
            plt.title(strings[i] + ' Location, Found Stimulus :' + str(foundStimulus[i]))
            plt.show()
        plt.savefig(strings[i] + '_experiment_eegSegmentedWindowsFiltered' + str(expNo), bbox_inches='tight', 
        pad_inches=0, dpi=200)
        plt.close()
        
    return foundStimulus

def plotP300(p300Signals, channelCount, setDirectory, expNo):
    os.chdir(setDirectory)
    strings = ['Fp1', 'Fz', 'Cz', 'Pz']       
    xAxis = np.arange(0, 599, 2)
    for i in range(channelCount):
        plt.figure()
        for j in range(5): 
            plt.plot(xAxis, p300Signals[5*i+j], label=[j])
            plt.ylabel('Amplitude [uV]', fontsize=20)
            plt.xlabel('Time [Ms]', fontsize=20)
            plt.legend(loc='upper right', fontsize=10)
            plt.title(strings[i] + ' Location')
            plt.show()
        plt.savefig(strings[i] + '_deney_JustStimuluses' + str(expNo), bbox_inches='tight', 
        pad_inches=0, dpi=200)
        plt.close()
        
def plotP300TargetNonTargetFrequent(p300Signals, channelCount, setDirectory, expNo, stimulusNo):
    os.chdir(setDirectory)
    strings = ['Fp1', 'Fz', 'Cz', 'Pz']       
    xAxis = np.arange(0, 599, 2)
    if(stimulusNo==0):
        nontarget = np.array([1,2])
    elif(stimulusNo==1):
        nontarget = np.array([0,2])
    else:
        nontarget = np.array([0,1])
        
    frequent = np.array([3,4])
    
    targetNontargetFreq = np.zeros((3,300))
    label = ['Target','Nontarget','Frequent']
    for i in range(channelCount):
        plt.figure()
        targetNontargetFreq[0] = p300Signals[5*i+stimulusNo]
        targetNontargetFreq[1] = np.mean([p300Signals[nontarget[0]], p300Signals[nontarget[1]]], axis=0)
        targetNontargetFreq[2] = np.mean([p300Signals[frequent[0]], p300Signals[frequent[1]]], axis=0)
        for j in range(3): 
            plt.plot(xAxis, targetNontargetFreq[j], label=label[j])
            plt.ylabel('Amplitude [uV]', fontsize=20)
            plt.xlabel('Time [Ms]', fontsize=20)
            plt.legend(loc='upper right', fontsize=10)
            plt.title(strings[i] + ' Location')
            plt.show()
        plt.savefig(strings[i] + '_tarNontarFrequent_deney' + str(expNo), bbox_inches='tight', 
        pad_inches=0, dpi=200)
        plt.close()        
        
def plotP300WithStds(p300Signals, stdWindows, channelCount, setDirectory, targetStimulus):
    os.chdir(setDirectory)
    strings = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'P4', 'P3']      
    xAxis = np.arange(0, 599, 2)
    linestyle = '--'
    for i in range(channelCount):
        plt.figure()
        plt.plot(xAxis, p300Signals[3*i+targetStimulus], linewidth = 4, label=['Target Stimulus'])
        plt.plot(xAxis, p300Signals[3*i+targetStimulus] + stdWindows[3*i+targetStimulus], color = 'black', linestyle = linestyle,
                 label=['P300+Std'], linewidth = 0.7)
        plt.plot(xAxis, p300Signals[3*i+targetStimulus] - stdWindows[3*i+targetStimulus], color = 'black', linestyle = linestyle,
                 label=['P300-Std'], linewidth = 0.7)
        plt.ylabel('Amplitude [uV]', fontsize=20)
        plt.xlabel('Time [Ms]', fontsize=20)
        plt.legend(loc='upper left', fontsize=5)
        plt.title(strings[i] + ' Location with Standart Deviations')
        plt.show()
        plt.savefig(strings[i] + '_experiment3_p300_WithStds', bbox_inches='tight', 
        pad_inches=0, dpi=200)
        plt.close()

def plotP300Stds(stdWindows, setDirectory, targetStimulus):
    os.chdir(setDirectory)
    xAxis = np.arange(0, 599, 2)
    strings = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'P4', 'P3']      
    for i in range(channelCount):
        plt.plot(xAxis, stdWindows[3*i+targetStimulus])
        plt.ylabel('Amplitude [uV]', fontsize=20)
        plt.xlabel('Time [Sample]', fontsize=20)
        plt.title("Standart Deviation of " + strings[i] + " Windows") 
        plt.show()
        plt.savefig(strings[i] + '_experiment3_stdofWindows', bbox_inches='tight', 
        pad_inches=0, dpi=200)
        plt.close()
        
def P300FinderAlgorithmPeak(p300Signals):
    peaks = np.max(np.abs(p300Signals), axis=1) #P300 finding algorithm
    stimulus = np.argmax(peaks)
    return stimulus

def P300FinderAlgorithmSTD(p300Signals):
    stds = np.std(p300Signals, axis=1)    
    stimulus = np.argmax(stds)    
    return stimulus
    
def P300FinderAlgorithmTotEnergy(p300Signals):
    totens = np.sum(np.abs(p300Signals), axis=1)
    stimulus = np.argmax(totens)    
    return stimulus

def P300TravellerFinder(p300Signals, intervalLength):
    stimulus = P300FinderAlgorithmPeak(p300Signals)
    
    index = np.argmax(p300Signals[stimulus])
    positive_interval, negative_interval = intervalLength, intervalLength
    if(index + intervalLength > 300):
       positive_interval = 300 - index
    if(index - intervalLength < 0):
        negative_interval = index
    
    newP300 = np.zeros((3, positive_interval + negative_interval))
    for i in range(len(p300Signals)):
        newP300[i] = p300Signals[i,index - negative_interval: index + positive_interval]
    
    finalStimulus = P300FinderAlgorithmSTD(newP300)
    return finalStimulus
#=========================================== Trains ========================================================================
#===========================================================================================================================
def allTypeofTrainsetCreator_forAllBrainChannels(eegSignals, stimulusLogs, downSamplingSize, lastNStimulus, label):
    eegSignals_channel0, eegSignals_channel1, eegSignals_channel2, eegSignals_channel3 = list(), list(), list(), list()
    for i in range(len(eegSignals)):
        eegSignals_channel0.append(eegSignals[i][0])
        eegSignals_channel1.append(eegSignals[i][1])
        eegSignals_channel2.append(eegSignals[i][2])
        eegSignals_channel3.append(eegSignals[i][3])
        
    trainsXY_channel0 = allTypeofTrainsetCreator(eegSignals_channel0, stimulusLogs, downSamplingSize, lastNStimulus, label)     
    trainsXY_channel1 = allTypeofTrainsetCreator(eegSignals_channel1, stimulusLogs, downSamplingSize, lastNStimulus, label) 
    trainsXY_channel2 = allTypeofTrainsetCreator(eegSignals_channel2, stimulusLogs, downSamplingSize, lastNStimulus, label) 
    trainsXY_channel3 = allTypeofTrainsetCreator(eegSignals_channel3, stimulusLogs, downSamplingSize, lastNStimulus, label)     
    return trainsXY_channel0, trainsXY_channel1, trainsXY_channel2, trainsXY_channel3    
    

def allTypeofTrainsetCreator(eegSignals, stimulusLogs, downSamplingSize, lastNStimulus, label):
    trainX0All, trainX1All, trainX2All, trainX3All, trainY0All, trainY1All, trainY2All, trainY3All = [],[],[],[],[],[],[],[]
    random.seed(312)
    for i in range(len(eegSignals)):
        temp_eegSignals = eegSignals[i]
        stimulusLog = stimulusLogs[i]
        trainX0, trainY0 = P300SKLDADownSampledTrainsetCreator(temp_eegSignals, downSamplingSize, label[i])    
        trainX1, trainY1 = P300SKLDAOddballParadigmDownsampledTrainsetCreator(temp_eegSignals, lastNStimulus, stimulusLog, downSamplingSize, label[i])
        trainX2, trainY2 = P300SKLDATrainsetCreator(temp_eegSignals, label[i])
        trainX3, trainY3 = P300SKLDAOddballParadigmTrain(temp_eegSignals, stimulusLog, lastNStimulus, label[i])
        
        if(i==0):
            trainX0All, trainX1All, trainX2All, trainX3All = trainX0, trainX1, trainX2, trainX3
            trainY0All, trainY1All, trainY2All, trainY3All = trainY0, trainY1, trainY2, trainY3
        else:
            trainX0All = np.append(trainX0All, trainX0, axis=0)
            trainX1All = np.append(trainX1All, trainX1, axis=0)
            trainX2All = np.append(trainX2All, trainX2, axis=0)
            trainX3All = np.append(trainX3All, trainX3, axis=0)

            trainY0All = np.append(trainY0All, trainY0, axis=0)
            trainY1All = np.append(trainY1All, trainY1, axis=0)
            trainY2All = np.append(trainY2All, trainY2, axis=0)
            trainY3All = np.append(trainY3All, trainY3, axis=0)
        
    trainsXY = ((trainX0All,trainY0All),(trainX1All,trainY1All),(trainX2All,trainY2All),(trainX3All,trainY3All))
    return trainsXY

def allTypeofModelCreator_andCrossValidationAccuracyFinder(trainsXY, typeOfClf, randFrstEstimators, ann_layer1, ann_layer2, brainChannel, plotConf, directorySaveModel, modelFilename):
    scores = np.zeros((2))
    models = list()
    confMats = list()
    j=0
    class_names = ['Non-target Stimulus', 'Target Stimulus']
    for i in range(1): #only take 1. and 3. algorithm
#        if(i==0):
#            j=1
#        else:
        j=3
        trainsXYTemp = trainsXY[j]
        #========= Classifiers ==================
        clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage='auto',
                                     solver='lsqr', store_covariance=False, tol=0.0001) 
        rndfrst = RandomForestClassifier(n_estimators=randFrstEstimators, criterion='gini', max_depth=None, min_samples_split=2, 
                                     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                     max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, 
                                     verbose=0, warm_start=False, class_weight=None)
        lineardisc = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
                                            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                                            learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
                                            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
                                            verbose=0, warm_start=False)
        svmModel = svm.SVC()
        nbrs = neighbors.KNeighborsClassifier(10, weights='distance')
        
        mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
        beta_1=0.9, beta_2=0.999, early_stopping=False,
        epsilon=1e-08, hidden_layer_sizes=(ann_layer1,ann_layer2), learning_rate='constant',
        learning_rate_init=0.001, max_iter=200, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
        warm_start=False)
        #==========Train==========================
        random.seed(312)
        if(typeOfClf == 0):
            scores[i] = np.mean(cross_val_score(clf, trainsXYTemp[0], trainsXYTemp[1].astype("int"), cv=10))
            y_pred = cross_val_predict(clf,trainsXYTemp[0] ,trainsXYTemp[1].astype("int"), cv=10)
            model = clf.fit(trainsXYTemp[0],trainsXYTemp[1])
        elif(typeOfClf == 1):
            scores[i] = np.mean(cross_val_score(rndfrst, trainsXYTemp[0], trainsXYTemp[1].astype("int"), cv=10))
            print(i)
            y_pred = cross_val_predict(rndfrst,trainsXYTemp[0] ,trainsXYTemp[1].astype("int"), cv=10)
            model = rndfrst.fit(trainsXYTemp[0],trainsXYTemp[1])
        elif(typeOfClf == 2):
            scores[i] = np.mean(cross_val_score(lineardisc, trainsXYTemp[0], trainsXYTemp[1].astype("int"), cv=10))
            y_pred = cross_val_predict(lineardisc,trainsXYTemp[0] ,trainsXYTemp[1].astype("int"), cv=10)
            model = lineardisc.fit(trainsXYTemp[0],trainsXYTemp[1])
        elif(typeOfClf == 3):
            scores[i] = np.mean(cross_val_score(svmModel, trainsXYTemp[0], trainsXYTemp[1].astype("int"), cv=10))
            y_pred = cross_val_predict(svmModel,trainsXYTemp[0] ,trainsXYTemp[1].astype("int"), cv=10)
            model = svmModel.fit(trainsXYTemp[0],trainsXYTemp[1])
        elif(typeOfClf == 4):
            scores[i] = np.mean(cross_val_score(nbrs, trainsXYTemp[0], trainsXYTemp[1].astype("int"), cv=10))
            y_pred = cross_val_predict(nbrs,trainsXYTemp[0] ,trainsXYTemp[1].astype("int"), cv=10)
            model = nbrs.fit(trainsXYTemp[0],trainsXYTemp[1])
        elif(typeOfClf == 5):
            scores[i] = np.mean(cross_val_score(mlp, trainsXYTemp[0], trainsXYTemp[1].astype("int"), cv=10))
            y_pred = cross_val_predict(mlp,trainsXYTemp[0] ,trainsXYTemp[1].astype("int"), cv=10)
            model = mlp.fit(trainsXYTemp[0],trainsXYTemp[1])
        models.append(model)
        #========== Save Model =====================
        os.chdir(directorySaveModel)
        # save the model to disk
        pickle.dump(model, open(modelFilename, 'wb'))
        #======== Confusion Matrix==============
        confMat = confusion_matrix(trainsXYTemp[1].astype("int"),y_pred)
        if(plotConf == 1):
            confMats.append(confMat)
            plt.figure()
            plot_confusion_matrix(confMat, classes=class_names, title= (brainChannel + ' Confusion matrix'))   
    
    return models, scores, confMats
#============================================= Sub-Train Methods ===================================================================
def P300SKLDADownSampledTrainsetCreator(eegSignals, downSamplingSize, targetStimulus):
    ratio = downSamplingSize / 500
    size = int(300 * ratio)
    
    p300Candidates = np.empty(shape=[0,size])
    trainY = np.empty(shape=[0,1])
    for i in range(3):
        for j in range(len(eegSignals[i])):
            if(i==targetStimulus):
                p300Candidates = np.row_stack((p300Candidates, resample(eegSignals[i][j], size)))
                trainY = np.row_stack((trainY,1))
            else:
                if(random.random() > 0.5):
                    p300Candidates = np.row_stack((p300Candidates, resample(eegSignals[i][j], size)))
                    trainY = np.row_stack((trainY,0))

    return p300Candidates, trainY.astype("int").flatten()
    
def P300SKLDAOddballParadigmDownsampledTrainsetCreator(eegSignals, lastNStimulus, stimulusLog, downSamplingSize, targetStimulus):
    stimulusAmount = 3 # 3 amount of stimulus
    windowMilliSecond = 600
    windowSize = int(windowMilliSecond / 2)   
    ratio = downSamplingSize / 500
    size = int(windowSize * ratio)     
    stimulusAmounts = np.zeros((stimulusAmount)).astype("int")
    
    for i in range(len(stimulusLog)):
        stimulusTemp = stimulusLog[i] - 1
        if(stimulusAmounts[0] >= lastNStimulus and stimulusAmounts[1] >= lastNStimulus and stimulusAmounts[2] >= lastNStimulus):
            break
        else:
            if(stimulusTemp < 3):
                stimulusAmounts[stimulusTemp] += 1
    
    p300Candidates = np.empty(shape=[0,size])
    trainY = np.empty(shape=[0,1])
    for i in range(stimulusAmount):
        for j in range(stimulusAmounts[i]-lastNStimulus, len(eegSignals[i])-lastNStimulus):
            tempP300 = np.mean(eegSignals[i][j:j+lastNStimulus], axis=0)
            tempP300 = resample(tempP300, size)
            if(i==targetStimulus):
                p300Candidates = np.row_stack((p300Candidates, tempP300))
                trainY = np.row_stack((trainY,1))
            else:
                if(random.random() > 0.5):
                    p300Candidates = np.row_stack((p300Candidates, tempP300))
                    trainY = np.row_stack((trainY,0))
            
    return p300Candidates, trainY.astype("int").flatten()

def P300SKLDATrainsetCreator(eegSignals, targetStimulus):
    p300Candidates = np.empty(shape=[0,300])
    trainY = np.empty(shape=[0,1])
    for i in range(3):
        for j in range(len(eegSignals[i])):
            if(i==targetStimulus):
                p300Candidates = np.row_stack((p300Candidates, eegSignals[i][j]))
                trainY = np.row_stack((trainY,1))
            else:
                if(random.random() > 0.5):
                    p300Candidates = np.row_stack((p300Candidates, eegSignals[i][j]))
                    trainY = np.row_stack((trainY,0))   

    return p300Candidates, trainY.astype("int").flatten()
    
def P300SKLDAOddballParadigmTrain(eegSignals, stimulusLog, lastNStimulus, targetStimulus):
    stimulusAmount = 3 # 5 amount of stimulus
    stimulusAmounts = np.zeros((stimulusAmount)).astype("int")      
    
    for i in range(len(stimulusLog)):
        stimulusTemp = stimulusLog[i] - 1
        if(stimulusAmounts[0] >= lastNStimulus and stimulusAmounts[1] >= lastNStimulus and stimulusAmounts[2] >= lastNStimulus):
            break
        else:
            if(stimulusTemp < 3):
                stimulusAmounts[stimulusTemp] += 1
    
    p300Candidates = np.empty(shape=[0,300])
    trainY = np.empty(shape=[0,1])
    for i in range(stimulusAmount):
        for j in range(stimulusAmounts[i]-lastNStimulus, len(eegSignals[i])-lastNStimulus):
            tempP300 = np.mean(eegSignals[i][j:j+lastNStimulus], axis=0)
            if(i==targetStimulus):
                p300Candidates = np.row_stack((p300Candidates, tempP300))
                trainY = np.row_stack((trainY,1))
            else:
                if(random.random() > 0.5):
                    trainY = np.row_stack((trainY,0))   
                    p300Candidates = np.row_stack((p300Candidates, tempP300))
            
    return p300Candidates, trainY.astype("int").flatten()
#==================================================== Tests =======================================================================
def P300SKLDADownSampledTest(model, instantEEGSignal, downSamplingSize):
    ratio = downSamplingSize / 500
    size = int(300 * ratio)     
    foundStimuluses = np.zeros((3))
    
    for i in range(3):
        tempX = resample(instantEEGSignal[i], size).reshape(1,-1)        
        foundStimuluses[i] = model.predict(tempX)
        
    foundStimulus = np.argmax(foundStimuluses)
    return foundStimulus

def P300SLDAOddballParadigmTest(model, lastNEEGSignals, ifANN):
    foundStimuluses = np.zeros((3))
    targetProbs = np.zeros((3))
    
    if(ifANN == 1):
        for i in range(3):
            p300Signals = np.mean(lastNEEGSignals[i], axis=0).reshape(1,-1)
            tempProbs = model.predict_proba(p300Signals)
            targetProbs[i] = tempProbs[0,1]            
        foundStimulus = np.argmax(targetProbs)   
    else:    
        for i in range(3):
            p300Signals = np.mean(lastNEEGSignals[i], axis=0).reshape(1,-1)
            foundStimuluses[i] = model.predict(p300Signals)        
        foundStimulus = np.argmax(foundStimuluses)
        
    return foundStimulus
 
def P300SKLDAOddballParadigmDownsampledTest(model, lastNEEGSignals, downSamplingSize, ifANN):
    ratio = downSamplingSize / 500
    size = int(300 * ratio)     
    foundStimuluses = np.zeros((3,2))
    targetProbs = np.zeros((3))
    
    if(ifANN == 1):
        for i in range(3):
            p300Signals = np.mean(lastNEEGSignals[i], axis=0)
            p300SignalDownSampled = resample(p300Signals, size).reshape(1,-1)
            tempProbs = model.predict_proba(p300SignalDownSampled)
            targetProbs[i] = tempProbs[0,1]            
        foundStimulus = np.argmax(targetProbs)   
    else:
         for i in range(3):
            p300Signals = np.mean(lastNEEGSignals[i], axis=0)
            p300SignalDownSampled = resample(p300Signals, size).reshape(1,-1)
            foundStimuluses[i] = model.predict(p300SignalDownSampled)
         foundStimulus = np.argmax(foundStimuluses)   
         
    return foundStimulus
    
def P300SKLDATest(model, instantEEGSignal):
    foundStimuluses = np.zeros((3))
    
    for i in range(3):
        foundStimuluses[i] = model.predict(instantEEGSignal[i].reshape(1,-1))
        
    foundStimulus = np.argmax(foundStimuluses)
    return foundStimulus
#===================================================================================================================================
def P300RealTimeAnalyzer(eegSignals, channelCount, string):
    windowMilliSecond = 600
    stimulusAmount = len(eegSignals)
    windowSize = int(windowMilliSecond / 2)              
    p300Signals = np.zeros((stimulusAmount, windowSize))

    for i in range(stimulusAmount):        
        p300Signals[i] = np.mean(eegSignals[i], axis=0)  
        
    stimulusStd = P300FinderAlgorithmSTD(p300Signals)
    stimulusPeak = P300FinderAlgorithmPeak(p300Signals)
    stimulusTotEn = P300FinderAlgorithmTotEnergy(p300Signals)
    stimulusTraveller = P300TravellerFinder(p300Signals, 50)
    #===Plotting Realtime P300 ============
#    string = 'Fz'
#    xAxis = np.arange(0, 599, 2)
#    plt.cla()
#    for i in range(stimulusAmount): 
#        plt.plot(xAxis, p300Signals[i], label=[i])
#        plt.ylabel('Amplitude [uV]', fontsize=20)
#        plt.xlabel('Time [Ms]', fontsize=20)
#        plt.legend(loc='upper right', fontsize=10)
#        plt.title(string + ' Location, Found Stimulus :' + str(stimulus))
#        plt.show()
#    plt.pause(.005)
    
    return stimulusStd, stimulusPeak, stimulusTotEn, stimulusTraveller   

def onflineP300Finder(eegSignals, stimulusAmount, stimulusLog, travallerIntervalLength):
    windowAmount = 0
    for i in range(stimulusAmount):
        windowAmount += len(eegSignals[i])
     
    windowMilliSecond = 600    
    windowSize = int(windowMilliSecond / 2)              
    p300Signals = np.zeros((3, windowSize))    
    
    foundStimulusLogStd = []
    foundStimulusLogPeak = []
    foundStimulusLogTotEn = []
    foundStimulusLogTraveller = []  
    
    tempEEGWindows = np.zeros((stimulusAmount,300))
    stCounts = np.zeros((3)).astype("int")
    for i in range(len(stimulusLog)):
        print(i)
        tempEEGWindows[stimulusLog[i]-1] += eegSignals[stimulusLog[i] - 1][stCounts[stimulusLog[i] - 1]]
        stCounts[stimulusLog[i] - 1] += 1            
        p300Signals[stimulusLog[i]-1] = tempEEGWindows[stimulusLog[i]-1] / (i+1)
            
        stimulusStd = P300FinderAlgorithmSTD(p300Signals)
        stimulusPeak = P300FinderAlgorithmPeak(p300Signals)
        stimulusTotEn = P300FinderAlgorithmTotEnergy(p300Signals)
        stimulusTraveller = P300TravellerFinder(p300Signals, travallerIntervalLength)    
        
        foundStimulusLogStd.append(stimulusStd)
        foundStimulusLogPeak.append(stimulusPeak)
        foundStimulusLogTotEn.append(stimulusTotEn)
        foundStimulusLogTraveller.append(stimulusTraveller)  
                
        totStimAmounts = np.zeros((4,stimulusAmount))
        for j in range(stimulusAmount):
            totStimAmounts[0][j] = foundStimulusLogStd.count(j)
        for j in range(stimulusAmount):
            totStimAmounts[1][j] = foundStimulusLogPeak.count(j)
        for j in range(stimulusAmount):
            totStimAmounts[2][j] = foundStimulusLogTotEn.count(j)
        for j in range(stimulusAmount):
            totStimAmounts[3][j] = foundStimulusLogTraveller.count(j)
        
        
    return foundStimulusLogStd, foundStimulusLogPeak, foundStimulusLogTotEn, foundStimulusLogTraveller, totStimAmounts

def brainwaveFinder(eegSignal, Fs):
#    p300Signal = np.mean(eegSignals, axis=0)
    eegSignal = notchFilter(eegSignal, Fs, 50, 30)
    deltaSignal = butter_bandpass_filter(eegSignal, 0.5, 3, 500, order=3)
    thetaSignal = butter_bandpass_filter(eegSignal, 3, 8, Fs, order=3)
    alphaSignal = butter_bandpass_filter(eegSignal, 8, 12, Fs, order=3)
    betaSignal = butter_bandpass_filter(eegSignal, 12, 38, Fs, order=3)
    gammaSignal = butter_bandpass_filter(eegSignal, 38, 48, Fs, order=3)
    
    deltaSignalEnergy = np.sum(deltaSignal**2)
    thetaSignalEnergy = np.sum(thetaSignal**2)
    alphaSignalEnergy = np.sum(alphaSignal**2)
    betaSignalEnergy = np.sum(betaSignal**2)
    gammaSignalEnergy = np.sum(gammaSignal**2)
    
    allEnergies = np.array([deltaSignalEnergy, thetaSignalEnergy, alphaSignalEnergy, betaSignalEnergy, gammaSignalEnergy])
    
    return allEnergies

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#==================================== Area 51 ======================================================
#    ratio = 20 / 500
#    size = int(300 * ratio)     
#    targetProbs = np.zeros((3))
#    for i in range(3):
#        p300Signals = np.mean(tempBigWindow[i], axis=0)
#        p300SignalDownSampled = resample(p300Signals, size).reshape(1,-1)
#        tempProbs = models1[0].predict_proba(p300SignalDownSampled)
#        foundStimuluses[i,0] = np.max(tempProbs)
#        foundStimuluses[i,1] = np.argmax(tempProbs)
#        
#    foundStimulus = foundStimuluses[np.argmax(foundStimuluses[:,0]),1]

# for i in range(3):
#     p300Signals = np.mean(tempBigWindow[i], axis=0)
#     p300SignalDownSampled = resample(p300Signals, size).reshape(1,-1)
#     foundStimuluses[i] = models2[0].predict(p300SignalDownSampled)
# foundStimulus = np.argmax(foundStimuluses)   
#      for i in range(3):
#            p300Signals = np.mean(tempBigWindow[i], axis=0).reshape(1,-1)
#            tempProbs = loaded_model.predict_proba(p300Signals)
#            targetProbs[i] = tempProbs[0,1]            
#      foundStimulus = np.argmax(targetProbs)   
#     foundStimuluses = np.zeros((3))
#     for i in range(3):
#            p300Signals = np.mean(tempBigWindow[i], axis=0).reshape(1,-1)
#            foundStimuluses[i] = models3[1].predict(p300Signals)        
#     foundStimulus = np.argmax(foundStimuluses)
