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
import cv2
import scipy.signal

import matplotlib.pyplot as plt

from h5utils import Hdf5Dataset
from optparse import OptionParser

logger = logging.getLogger(__name__)

def is_cv2():
    import cv2 as lib
    return lib.__version__.startswith("2.")
 
def is_cv3():
    import cv2 as lib
    return lib.__version__.startswith("3.")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

class FlowPlotter(object):
    
    def __init__(self, x, y, scale):
        
        h, w = x.shape
        self.x = x
        self.y = y 
        
        # NOTE: the y-axis needs to be inverted to be in image-coordinates
        fig = plt.figure(figsize=(20,10), facecolor='white')
        ax = fig.add_subplot(121)
        q = ax.quiver(x, y, np.zeros((h,w)), np.zeros((h,w)), edgecolor='k', scale=1, angles='xy', scale_units='xy')
        ax.invert_yaxis()
        plt.axis('off')
        
        self.fig = fig
        self.ax = ax
        self.q = q

        ax2 = fig.add_subplot(122)
        m = ax2.imshow(np.zeros((int(h/scale), int(w/scale))), vmin = 0, vmax = 255, cmap = plt.get_cmap('gray'))
        plt.axis('off')
        self.ax2 = ax2
        self.m = m
        
        plt.ion()
    
    def update(self, flow, img):
        
        # NOTE: the y-axis needs to be negated to be in image-coordinates
        self.q.set_UVC(flow[:,:,0], -flow[:,:,1])
        self.m.set_data(img)
        self.fig.canvas.draw()

def computeSparseFlow(img, imgPrev, gridShape, border):
    
    # Define the fixed grid where optical flow is calculated
    h, w = img.shape[:2]
    y, x = np.meshgrid(np.linspace(border * h, (1.0 - border) * h, gridShape[0], dtype=np.int),
                       np.linspace(border * w, (1.0 - border) * w, gridShape[1], dtype=np.int),
                       indexing='ij')
    p0 = np.stack((y,x), axis=-1).astype(np.float32)
    
    # Calculate optical flow
    if is_cv2():
        p1, _, _ = cv2.calcOpticalFlowPyrLK(imgPrev, img, p0.reshape((-1,2)), None, winSize=(63,63), maxLevel=5)
    else:
        p1, _, _ = cv2.calcOpticalFlowPyrLK(imgPrev, img, p0.reshape((-1,2)), None, winSize=(63,63), maxLevel=5)

    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[y, x] = (p1.reshape(p0.shape) - p0)
    # TODO: test if all valid with p1[st==1] ?
    f = flow[y, x]
    
    return f, x, y

def computeCompactFlow(img, imgPrev, gridShape, border, filtering=False, filterSize=16):
    
    # Calculate optical flow
    if is_cv2():
        flow = cv2.calcOpticalFlowFarneback(imgPrev, img, pyr_scale=0.5, levels=5, winsize=63, iterations=3, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    else:
        flow = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(imgPrev, img, flow, pyr_scale=0.5, levels=5, winsize=63, iterations=3, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    flow = np.array(flow, np.float32)
    
    h, w = flow.shape[:2]
    y, x = np.meshgrid(np.linspace(border * h, (1.0 - border) * h, gridShape[0], dtype=np.int),
                       np.linspace(border * w, (1.0 - border) * w, gridShape[1], dtype=np.int),
                       indexing='ij')
    
    if filtering:
        kernel = np.ones((filterSize, filterSize)) / float(filterSize * filterSize)
        flow = np.stack([scipy.signal.convolve2d(flow[:,:,k], kernel, mode='same', boundary='symm')
                         for k in range(flow.shape[-1])], axis=-1)
    f = flow[y, x]
    return f, x, y

def preprocessImage(img, scale=0.5):
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imggray = clahe.apply(imggray)

    imggray = np.array(imggray * 255, dtype=np.uint8)
    imggray = cv2.resize(imggray, None,fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return imggray

def processImageSequence(raw, clock, shape, scale=0.25, gridShape=(12,16), border=0.1, filtering=True, filterSize=5, visualize=False, stepSize=1):
    
    if visualize:
        plotter = None
    
    flows = []
    for i in range(stepSize, raw.shape[0]):
        
        logger.info('Processing image %d (%d total)' % (i+1, raw.shape[0]))
        
        img = cv2.imdecode(raw[i, :shape[i,0]], flags=-1) #CV_LOAD_IMAGE_UNCHANGED
        img = preprocessImage(img, scale)
        
        imgPrev = cv2.imdecode(raw[i-stepSize, :shape[0,0]], flags=-1) # CV_LOAD_IMAGE_UNCHANGED
        imgPrev = preprocessImage(imgPrev, scale)
        
        flow, x, y = computeCompactFlow(img, imgPrev, gridShape, border, filtering, filterSize)
        #flow, x, y = computeSparseFlow(img, imgPrev, gridShape, border)
        
        # Divide by the time interval and scale to get flow in unit of pixels/sec relative to the full-size image
        dt = clock[i] - clock[i-stepSize]
        assert dt > 0
        flow /=  (dt * scale)
        
        if visualize:
            if plotter is None:
                plotter = FlowPlotter(x, y, scale)
            plotter.update(flow, img)
            plt.show()
        
        flows.append(flow)

    flows = np.stack(flows, axis=0)
    clock = clock[stepSize:]
    
#     # Temporal filtering
#     if filtering:
#         kernel = np.ones((filterSize,)) / float(filterSize)
#         for i in range(flow.shape[0]):
#             for j in range(flow.shape[1]):
#                 flow[i,j,:] = scipy.signal.convolve(flow[i,j,:], kernel, mode='same')
        
    return flows, clock

def main(args=None):

    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", type='string', default=None,
                      help='specify the path of the input HDF5 file')
    parser.add_option("-o", "--output", dest="output", type='string', default=None,
                      help='specify the path of the output HDF5 file')
    parser.add_option("-s", "--scale", dest="scale", type='float', default=0.5,
                      help='specify the rescaling of images')
    parser.add_option("-b", "--border", dest="border", type='float', default=0.1,
                      help='specify the border to keep')
    parser.add_option("-f", "--filter-size", dest="filterSize", type='int', default=16,
                      help='specify the size of the smoothing filter')
    parser.add_option("-m", "--step-size", dest="stepSize", type='int', default=1,
                      help='specify the number of frame between optical flow calculation')
    parser.add_option("-y", "--grid-shape-y", dest="gridShape_y", type='int', default=12,
                  help='specify the shape of the optical flow on the y-axis')
    parser.add_option("-x", "--grid-shape-x", dest="gridShape_x", type='int', default=16,
                  help='specify the shape of the optical flow on the x-axis')
    parser.add_option("-t", "--filtering", action="store_true", dest="filtering", default=False,
                      help='specify to visualize computed optical flow')
    parser.add_option("-v", "--visualize", action="store_true", dest="visualize", default=False,
                      help='specify to visualize computed optical flow')
    (options,args) = parser.parse_args(args=args)

    inputDatasetPath = os.path.abspath(options.input)
    logger.info('Using input HDF5 dataset file: %s' % (inputDatasetPath))

    outputDatasetPath = os.path.abspath(options.output)    
    logger.info('Using output HDF5 dataset file: %s' % (outputDatasetPath))

    with Hdf5Dataset(outputDatasetPath, mode='w') as outHdf5:
        with Hdf5Dataset(inputDatasetPath, mode='r') as inHdf5:
            
            # Process states
            for state in inHdf5.getAllStates():
                name, group, raw, clock, shape = state
                
                # Write original data to output file
                if group is not None:
                    ngroup = '/' + group
                else:
                    ngroup = ''
                fs = 1.0/np.mean(clock[1:] - clock[:-1])
                logger.info('Writing to output HDF5 dataset file: %s, shape=%s, fs=%f Hz' % (ngroup + '/' + name, str(raw.shape), fs))
                
                outHdf5.addStates(name, raw, clock, group, shape)
                
                if group == 'video':
                    rawFlow, clockFlow = processImageSequence(raw, clock, shape,
                                                              scale=options.scale, gridShape=(options.gridShape_y, options.gridShape_x), border=options.border, 
                                                              filtering=options.filtering, filterSize=options.filterSize,
                                                              visualize=options.visualize, stepSize=options.stepSize)
                    groupFlow = 'optical_flow'
                    
                    # Write original data to output file
                    fsFlow = 1.0/np.mean(clockFlow[1:] - clockFlow[:-1])
                    logger.info('Writing to output HDF5 dataset file: %s, shape=%s, fs=%f Hz' % ('/' + groupFlow + '/' + name, str(rawFlow.shape), fsFlow))
                    
                    outHdf5.addStates(name, rawFlow, clockFlow, groupFlow, shape=None)
                        
    logger.info('All done.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
