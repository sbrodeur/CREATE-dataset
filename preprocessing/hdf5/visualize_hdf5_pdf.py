#!/usr/bin/env python

# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright 
#    notice, this list of conditions and the following disclaimer.
#   
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
# OF SUCH DAMAGE.

import os
import logging
import numpy as np
import h5py

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from optparse import OptionParser

logger = logging.getLogger(__name__)

class Hdf5CreateDataset:

    def __init__(self, filePath):
        self.h = h5py.File(filePath, mode='r')
        
    def getData(self, name, group=None):
        
        if group is not None:
            g = self.h[group]
        else:
            g = self.h
        g = g[name]
            
        assert 'data' in g.keys()
        data = np.array(g['data'])
        clock = np.array(g['clock'])
        if 'shape' in g.keys():
            shape = np.array(g['shape'])
        else:
            shape = None
            
        return [data, clock, shape]
            
    def close(self):
        self.h.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

def getPerSampleClock(data, clock, fs):
    chunkSize = data.shape[1]
    sclock = []
    for t in clock:
        tc = t - np.arange(chunkSize)[::-1] / fs
        sclock.append(tc)
    sclock = np.array(sclock).ravel()
    return sclock

def plotAudioSignal(dataset):
    
    # NOTE: This is the known sampling frequency
    fs = 16000
    
    ldata, lclock, _ = dataset.getData('left', 'audio')
    lclock = getPerSampleClock(ldata, lclock, fs)
    ldata = np.array(ldata.flatten(), dtype=np.float32) / np.iinfo('int16').max
    
    rdata, rclock, _ = dataset.getData('right', 'audio')
    rclock = getPerSampleClock(rdata, rclock, fs)
    rdata = np.array(rdata.flatten(), dtype=np.float32) / np.iinfo('int16').max
    
    fig = plt.figure(figsize=(12,4), facecolor='white', frameon=False)

    # Left channel
    plt.subplot(121, rasterized=True)
    plt.title('Audio signal (left channel)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.plot(lclock, ldata)
    
    # Right channel
    plt.subplot(122, rasterized=True)
    plt.title('Audio signal (right channel)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.plot(rclock, rdata)
    
    return fig

def plotAudioSpectrogram(dataset):
    
    # NOTE: This is the known sampling frequency
    fs = 16000
    
    ldata, lclock, _ = dataset.getData('left', 'audio')
    lclock = getPerSampleClock(ldata, lclock, fs)
    ldata = np.array(ldata.flatten(), dtype=np.float32) / np.iinfo('int16').max
    
    rdata, rclock, _ = dataset.getData('right', 'audio')
    rclock = getPerSampleClock(rdata, rclock, fs)
    rdata = np.array(rdata.flatten(), dtype=np.float32) / np.iinfo('int16').max
    
    fig = plt.figure(figsize=(12,4), facecolor='white', frameon=False)

    # NOTE: make sure left and right channels are approximately aligned in time, 
    #       since the 'specgram' of Matplotlib doesn't allow to specify the times.
    nbTrimSamples = int(np.abs(rclock[0] - lclock[0]) * fs)
    if rclock[0] > lclock[0]:
        ldata = ldata[nbTrimSamples:]
    else:
        rdata = rdata[nbTrimSamples:]
    
    # Left channel
    plt.subplot(121, rasterized=True)
    vmin = 20*np.log10(np.max(ldata)) - 120
    _, _, _, cax = plt.specgram(ldata, NFFT=256, Fs=fs, noverlap=128, vmin=vmin, mode='magnitude', scale='dB', cmap='inferno')
    fig.colorbar(cax).set_label('Intensity [dB]')
    plt.title('Magnitude spectrum (left channel)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency')
    
    # Right channel
    plt.subplot(122, rasterized=True)
    vmin = 20*np.log10(np.max(ldata)) - 120
    _, _, _, cax = plt.specgram(rdata, NFFT=256, Fs=fs, noverlap=128, vmin=vmin, mode='magnitude', scale='dB', cmap='inferno')
    fig.colorbar(cax).set_label('Intensity [dB]')
    plt.title('Magnitude spectrum (right channel)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency')
    
    return fig
    
def plotMotors(dataset):
    
    data, clock, _ = dataset.getData('linear_velocity', 'motor')
    ldata = data[:,0] # Left motor
    rdata = data[:,1] # Right motor
    
    fig = plt.figure(figsize=(12,4), facecolor='white')

    # Motors
    plt.subplot(111, rasterized=True)
    plt.title('Motors')
    plt.plot(clock, ldata, c='b', label='Left motor')
    plt.plot(clock, rdata, c='g', label='Right motor')
    plt.xlabel('Time [sec]')
    plt.ylabel('Linear velocity [m/sec]')
    plt.legend()
    
    return fig
    
def plotImuAngular(dataset):
    
    data, clock, _ = dataset.getData('angular_velocity', 'imu')
    angularVelXData = data[:,0] # X-axis
    angularVelYData = data[:,1] # Y-axis
    angularVelZData = data[:,2] # Z-axis
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(111, rasterized=True)
    plt.title('IMU gyroscope')
    plt.plot(clock, angularVelXData, c='b', label='X-axis')
    plt.plot(clock, angularVelYData, c='g', label='Y-axis')
    plt.plot(clock, angularVelZData, c='r', label='Z-axis')
    plt.xlabel('Time [sec]')
    plt.ylabel('Angular velocity [rad/sec]')
    plt.legend()
    
    return fig
    
def plotImuLinear(dataset):
    
    data, clock, _ = dataset.getData('linear_acceleration', 'imu')
    linearAccelXData = data[:,0] # X-axis
    linearAccelYData = data[:,1] # Y-axis
    linearAccelZData = data[:,2] # Z-axis
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(111, rasterized=True)
    plt.title('IMU linear acceleration')
    plt.plot(clock, linearAccelXData, c='b', label='X-axis')
    plt.plot(clock, linearAccelYData, c='g', label='Y-axis')
    plt.plot(clock, linearAccelZData, c='r', label='Z-axis')
    plt.xlabel('Time [sec]')
    plt.ylabel('Linear acceleration [m/sec^2]')
    plt.legend()
    
    return fig
    
def plotImuMagnetic(dataset):
    
    data, clock, _ = dataset.getData('magnetic_field', 'imu')

    # Note: convert from T to uT
    data = data * 1e6
    
    magFieldXData = data[:,0] # X-axis
    magFieldYData = data[:,1] # Y-axis
    magFieldZData = data[:,2] # Z-axis
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(111, rasterized=True)
    plt.title('IMU magnetic field')
    plt.plot(clock, magFieldXData, c='b', label='X-axis')
    plt.plot(clock, magFieldYData, c='g', label='Y-axis')
    plt.plot(clock, magFieldZData, c='r', label='Z-axis')
    plt.xlabel('Time [sec]')
    plt.ylabel('Magnetic field [uT]')
    plt.legend()
    
    return fig

def plotOdometryPosition(dataset):

    data, _, _ = dataset.getData('position', 'odometry')
    x = data[:,0]
    y = data[:,1]
    
    fig = plt.figure(figsize=(8,8), facecolor='white', frameon=False)
    
    plt.subplot(111, rasterized=True)
    plt.scatter(x, y, s=10)
    plt.scatter(x[-1], y[-1], c=[1.0,0.0,0.0], s=50)
    
    plt.grid(True)
    plt.title('Odometry position')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    
    return fig

def plotOdometryAngular(dataset):
    
    data, clock, _ = dataset.getData('angular_velocity', 'odometry')
    angularVelXData = data[:,0] # X-axis
    angularVelYData = data[:,1] # Y-axis
    angularVelZData = data[:,2] # Z-axis
    
    fig = plt.figure(figsize=(12,4), facecolor='white')

    plt.subplot(111, rasterized=True)
    plt.title('Odometry angular velocity')
    plt.plot(clock, angularVelXData, c='b', label='X-axis')
    plt.plot(clock, angularVelYData, c='g', label='Y-axis')
    plt.plot(clock, angularVelZData, c='r', label='Z-axis')
    plt.xlabel('Time [sec]')
    plt.ylabel('Angular velocity [rad/sec]')
    plt.legend()
    
    return fig

def plotOdometryLinear(dataset):
    
    data, clock, _ = dataset.getData('linear_velocity', 'odometry')
    linearVelXData = data[:,0] # X-axis
    linearVelYData = data[:,1] # Y-axis
    linearVelZData = data[:,2] # Z-axis
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(111, rasterized=True)
    plt.title('Odometry linear velocity')
    plt.plot(clock, linearVelXData, c='b', label='X-axis')
    plt.plot(clock, linearVelYData, c='g', label='Y-axis')
    plt.plot(clock, linearVelZData, c='r', label='Z-axis')
    plt.xlabel('Time [sec]')
    plt.ylabel('Linear velocity [m/sec]')
    plt.legend()
    
    return fig

def plotTemperature(dataset):
    
    data, clock, _ = dataset.getData('temperature', 'imu')

    fig = plt.figure(figsize=(12,4), facecolor='white')

    plt.subplot(111, rasterized=True)
    plt.plot(clock, data)
    plt.title('Temperature')
    plt.xlabel("Time [sec]")
    plt.ylabel("Temperature [celcius]")
    plt.ylim([15.0, 30.0])
    
    return fig

def plotPressure(dataset):
    
    data, clock, _ = dataset.getData('pressure', 'imu')

    # Note: convert from Pa to kPa
    data = data / 1000.0
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(111, rasterized=True)
    plt.plot(clock, data)
    plt.title('Atmospheric pressure')
    plt.xlabel("Time [sec]")
    plt.ylabel("Pressure [kPa]")
    
    return fig

def plotBatteryCharge(dataset):
    
    data, clock, _ = dataset.getData('charge', 'battery')
    voltage = data[:,0] # Voltage
    current = data[:,1] # Current
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(121, rasterized=True)
    plt.plot(clock, voltage)
    plt.title('Battery voltage')
    plt.xlabel("Time [sec]")
    plt.ylabel("Voltage [V]")
    plt.ylim([10.0, 18.0])
    
    plt.subplot(122, rasterized=True)
    plt.plot(clock, current)
    plt.title('Battery current')
    plt.xlabel("Time [sec]")
    plt.ylabel("Current [A]")
    plt.ylim([-2.0, 2.0])
    
    return fig

def plotBatteryCapacity(dataset):
    
    data, clock, _ = dataset.getData('charge', 'battery')
    charge = data[:,2]         # Charge remaining
    capacity = data[:,3]       # Capacity (estimated)
    percentage  = data[:,4]    # Percentage remaining
    
    fig = plt.figure(figsize=(16,4), facecolor='white')
     
    plt.subplot(131, rasterized=True)
    plt.plot(clock, charge)
    plt.title('Battery charge remaining')
    plt.xlabel("Time [sec]")
    plt.ylabel("Charge [Ah]")
    
    plt.subplot(132, rasterized=True)
    plt.plot(clock, percentage * 100.0)
    plt.title('Battery percentage remaining')
    plt.xlabel("Time [sec]")
    plt.ylabel("Percentage [%]")
    
    plt.subplot(133, rasterized=True)
    plt.plot(clock, capacity)
    plt.title('Battery capacity')
    plt.xlabel("Time [sec]")
    plt.ylabel("Capacity [Ah]")
    
    return fig

def plotCollisionRange(dataset):
    
    data, clock, _ = dataset.getData('range', 'collision')

    wall = data[:,0]
    cliffLeft = data[:,1]
    cliffFrontLeft = data[:,2]
    cliffFrontRight = data[:,3] 
    cliffRight = data[:,4] 
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(111, rasterized=True)
    plt.plot(clock, wall, '-r', label='Wall')
    plt.plot(clock, cliffLeft, '-g', label='Cliff left')
    plt.plot(clock, cliffFrontLeft, '-b', label='Cliff front-left')
    plt.plot(clock, cliffFrontRight, '-c', label='Cliff front-right')
    plt.plot(clock, cliffRight, '-m', label='Cliff right')
    plt.title('IR range sensors')
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    
    return fig
    
def plotCollisionBumper(dataset):
    
    data, clock, _ = dataset.getData('switch', 'collision')

    bumperLeft = data[:,0]
    bumperRight = data[:,1]
    wall = data[:,9]
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(111, rasterized=True)
    plt.plot(clock, bumperLeft, '-r', label='Bumper left')
    plt.plot(clock, bumperRight, '-g', label='Bumper right')
    plt.plot(clock, wall, '-b', label='Wall')
    plt.title('Contact sensors')
    plt.xlabel("Time [sec]")
    plt.ylabel("Activation")
    plt.legend(loc='upper right')
    plt.ylim([-0.1, 1.1])
    plt.yticks([0.0, 1.0], ['Off', 'On'])
    
    return fig
    
def plotCollisionCliff(dataset):
    
    data, clock, _ = dataset.getData('switch', 'collision')
    
    cliffLeft = data[:,5]
    cliffFrontLeft = data[:,6]
    cliffFrontRight = data[:,7] 
    cliffRight = data[:,8] 
    
    fig = plt.figure(figsize=(12,4), facecolor='white')

    plt.subplot(111, rasterized=True)
    plt.plot(clock, cliffLeft, '-r', label='Cliff left')
    plt.plot(clock, cliffFrontLeft, '-g', label='Cliff front-left')
    plt.plot(clock, cliffRight, '-b', label='Cliff front-right')
    plt.plot(clock, cliffFrontRight, '-c', label='Cliff right')
    plt.title('Cliff sensors')
    plt.xlabel("Time [sec]")
    plt.ylabel("Activation")
    plt.legend(loc='upper right')
    plt.ylim([-0.1, 1.1])
    plt.yticks([0.0, 1.0], ['Off', 'On'])
    
    return fig

def plotCollisionWheelDrop(dataset):
    
    data, clock, _ = dataset.getData('switch', 'collision')

    caster = data[:,2]
    wheelLeft = data[:,3]
    wheelRight = data[:,4]
    
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    plt.subplot(111, rasterized=True)
    plt.plot(clock, caster, '-r', label='Caster')
    plt.plot(clock, wheelLeft, '-g', label='Wheel left')
    plt.plot(clock, wheelRight, '-b', label='Wheel right')
    plt.title('Wheel drop sensors')
    plt.xlabel("Time [sec]")
    plt.ylabel("Activation")
    plt.legend(loc='upper right')
    plt.ylim([-0.1, 1.1])
    plt.yticks([0.0, 1.0], ['Off', 'On'])
    
    return fig

def plotOpticalFlow(dataset):

    ldata, lclock, _ = dataset.getData('left', 'optical_flow')
    rdata, rclock, _ = dataset.getData('right', 'optical_flow')
    
    # NOTE: only show the frames at time 100.0 sec
    frameTime = 100.0
    lframeIndex = np.argmin(np.abs(lclock - frameTime))
    rframeIndex = np.argmin(np.abs(rclock - frameTime))
    
    lflow = ldata[lframeIndex,:,:,:]
    rflow = rdata[rframeIndex,:,:,:]
    
    # NOTE: hardcoded image size and border used to compute optical flow
    ih, iw = 240, 320
    border = 0.1
    h,w = lflow.shape[:2]
    y, x = np.meshgrid(np.linspace(border*ih, (1.0-border)*ih, h, dtype=np.int),
                       np.linspace(border*iw, (1.0-border)*iw, w, dtype=np.int),
                       indexing='ij')
    
    fig = plt.figure(figsize=(12,6), facecolor='white', frameon=False)
    
    # Left channel
    plt.subplot(121, rasterized=True)
    plt.title('Optical flow (left camera)')
    plt.quiver(x, y, lflow[:,:,0], -lflow[:,:,1], 
               edgecolor='k', scale=1, scale_units='xy')
    plt.gca().invert_yaxis()
    plt.axis('off')
    
    # Right channel
    plt.subplot(122, rasterized=True)
    plt.title('Optical flow (right camera)')
    plt.quiver(x, y, rflow[:,:,0], -rflow[:,:,1], 
               edgecolor='k', scale=1, scale_units='xy')
    plt.gca().invert_yaxis()
    plt.axis('off')
    
    return fig

def main(args=None):

    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default=None,
                      help='specify the path of the input hdf5 dataset file')
    parser.add_option("-o", "--output", dest="output", default='.',
                      help='specify the path of the output pdf file')
    parser.add_option("-d", "--dpi", dest="dpi", type="int", default=300,
                      help='specify the DPI of the output pdf file')
    
    (options,args) = parser.parse_args(args=args)

    datasetPath = os.path.abspath(options.input)
    logger.info('Using input HDF5 dataset file: %s' % (datasetPath))

    outputPath = os.path.abspath(options.output)    
    logger.info('Using output PDF file: %s' % (outputPath))
    
    dpi = options.dpi
    with Hdf5CreateDataset(datasetPath) as dataset:
        
        with PdfPages(outputPath) as pdf:
            
            plotAudioSignal(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
            
            plotAudioSpectrogram(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
            
            plotMotors(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotImuAngular(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotImuLinear(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotImuMagnetic(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotOdometryPosition(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotOdometryAngular(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotOdometryLinear(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotTemperature(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotPressure(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotBatteryCharge(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotBatteryCapacity(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotCollisionRange(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotCollisionBumper(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotCollisionCliff(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
            plotCollisionWheelDrop(dataset)
            pdf.savefig(dpi=dpi)  # saves the current figure into a pdf page
            plt.close()
             
    logger.info('All done.')
                
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
