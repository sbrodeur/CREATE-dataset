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
import time
import logging
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import interp1d

from h5utils import Hdf5Dataset
from optparse import OptionParser

logger = logging.getLogger(__name__)
            
def is_cv2():
    import cv2 as lib
    return lib.__version__.startswith("2.")
 
def is_cv3():
    import cv2 as lib
    return lib.__version__.startswith("3.")
            
def decodeToRGB(data):
    # JPEG decoding using OpenCV
    img = cv2.imdecode(data, flags=1) # cv2.CV_LOAD_IMAGE_COLOR
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
# Adapted from: http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def set_data(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
# Adapted from: https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
def quat2mat(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < np.finfo(np.float64).eps:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def synchronizeData(data, clock, shape, syncClock):
    
    # Interpolation using indices for reduced memory usage
    indices = np.arange(data.shape[0], dtype=np.int)
    f = interp1d(clock, indices, kind='nearest', axis=0, copy=False)
    syncIndices = f(syncClock).astype(indices.dtype)
    syncData = data[syncIndices]
    
    if shape is not None:
        syncShape = shape[syncIndices]
    else:
        syncShape = None
        
    return syncData, syncClock, syncShape

def processDataset(dataset, syncClock, outputVideoFile, fsVideo=None, speedup=1.0, dpi=100):
    
    # Estimate sampling rate from clock (from any sensor)
    [_, _, _, clock, _] = dataset.getStates('linear_velocity', 'motor')
    fps = int(np.round(1.0/np.mean(clock[1:] - clock[:-1])))
    logger.info('Estimated sampling rate of %d Hz' % (fps))
    
    if fsVideo is None:
        fsVideo = fps
    downsampleRatio = np.max([1, int(fps/fsVideo)])
    logger.info('Using downfactor ratio: %d' % (downsampleRatio))
    
    # Left and right cameras
    [_, _, data, clock, shape] = dataset.getStates('left', 'video')
    leftCamData, _,  leftCamShape = synchronizeData(data, clock, shape, syncClock)
    [_, _, data, clock, shape] = dataset.getStates('right', 'video')
    rightCamData, _,  rightCamShape = synchronizeData(data, clock, shape, syncClock)
    assert len(syncClock) == len(leftCamData) == len(rightCamData)
    
    # Left and right optical flow fields
    [_, _, data, clock, shape] = dataset.getStates('left', 'optical_flow')
    leftOpflowData, _,  _ = synchronizeData(data, clock, shape, syncClock)
    [_, _, data, clock, shape] = dataset.getStates('right', 'optical_flow')
    rightOpflowData, _,  _ = synchronizeData(data, clock, shape, syncClock)
    assert len(syncClock) == len(leftOpflowData) == len(rightOpflowData)
     
    # NOTE: hardcoded image size and border used to compute optical flow
    ih, iw = 240, 320
    border = 0.1
    h,w = leftOpflowData.shape[1:3]
    y, x = np.meshgrid(np.linspace(border*ih, (1.0-border)*ih, h, dtype=np.int),
                       np.linspace(border*iw, (1.0-border)*iw, w, dtype=np.int),
                       indexing='ij')
     
    # Odometry position and orientation
    [_, _, data, clock, shape] = dataset.getStates('position', 'odometry')
    odomPosData, _,  _ = synchronizeData(data, clock, shape, syncClock)
    [_, _, data, clock, shape] = dataset.getStates('orientation', 'odometry')
    odomOriData, _,  _ = synchronizeData(data, clock, shape, syncClock)
    assert len(syncClock) == len(odomPosData) == len(odomOriData)
    
    # Create the video file writer
    writer = animation.FFMpegWriter(fps=speedup * fps/float(downsampleRatio), codec='libx264', extra_args=['-preset', 'fast', '-b:v', '1M'])
    
    # Create figure
    fig = plt.figure(figsize=(10,8), facecolor='white', frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.075, top=0.95, wspace=0.2, hspace=0.1)
        
    plt.subplot(221, rasterized=True)
    iml = plt.imshow(np.zeros((ih,iw,3)))
    ql = plt.quiver(x, y, np.zeros((h,w)), np.zeros((h,w)), edgecolor='k', scale=1, angles='xy', scale_units='xy')
    #plt.gca().invert_yaxis()
    plt.axis('off')
    plt.title('Left camera  (RGB and optical flow)')
    
    # Right camera
    plt.subplot(222, rasterized=True)
    imr = plt.imshow(np.zeros((ih,iw,3)))
    qr = plt.quiver(x, y, np.zeros((h,w)), np.zeros((h,w)), edgecolor='k', scale=1, angles='xy', scale_units='xy')
    #plt.gca().invert_yaxis()
    plt.axis('off')
    plt.title('Right camera (RGB and optical flow)')
    
    # Odometry orientation
    ax = plt.subplot(223, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_zticks([])
    ax.view_init(elev=45.0, azim=45.0)
    ax.axis([-1.0, 1.0, -1.0, 1.0])
    ax.set_zlim(-1.0, 1.0)
    plt.title('Odometry orientation', y=1.085)
    
    arrows = []
    labels = ['x', 'y', 'z']
    colors = ['r', 'g', 'b']
    vectors = np.eye(3)
    for i in range(3):
        x,y,z = vectors[:,i]
        arrow = Arrow3D([0.0, x], [0.0, y], [0.0, z], mutation_scale=20, 
                        lw=3, arrowstyle="-|>", color=colors[i], label=labels[i])
        arrows.append(arrow)
        ax.add_artist(arrow)
    
    proxies = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    ax.legend(proxies, labels, loc='upper right')
    
    # Odometry position
    odomPosAx = plt.subplot(224)
    scatPast = plt.scatter([0,], [0,], s=10)
    scatCur = plt.scatter([0,], [0,], c=[1.0,0.0,0.0], s=50)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis([-1.0, 1.0, -1.0, 1.0])
    plt.title('Odometry position')
    
    odomPosWindowsSize = 100000
    odomPosDataPts = np.zeros((odomPosWindowsSize, 2), dtype=np.float32)
    
    # Process each frame
    startTime = time.time()
    with writer.saving(fig, outputVideoFile, dpi):
    
        nbOdomPosPoints = 0
        nbFrames = len(syncClock)
        for n in range(nbFrames):
            
            # Update buffer
            odomPosDataPts[0:-1:,:] = odomPosDataPts[1::,:]
            odomPosDataPts[-1,:] = odomPosData[n,0:2]
            if nbOdomPosPoints < odomPosWindowsSize:
                nbOdomPosPoints += 1
            
            if n % int(downsampleRatio) == 0:
                
                # Update left camera plot
                leftCamImg = decodeToRGB(leftCamData[n,:leftCamShape[n,0]])
                iml.set_data(leftCamImg)
                
                # Update right camera plot
                rightCamImg = decodeToRGB(rightCamData[n,:rightCamShape[n,0]])
                imr.set_data(rightCamImg)
                
                # Update left optical flow plot
                ql.set_UVC(leftOpflowData[n,:,:,0], -leftOpflowData[n,:,:,1])
                
                # Update right optical flow plot
                qr.set_UVC(rightOpflowData[n,:,:,0], -rightOpflowData[n,:,:,1])
                
                # Update orientation
                # Convert from quaternion (w, x, y, z) to rotation matrix
                x,y,z,w = odomOriData[n]
                quaternion = np.array([w,x,y,z])
                R = quat2mat(quaternion)
                directions = np.eye(3) # x, y, z as column vectors
                vectors = np.dot(R, directions)
                assert np.allclose(np.linalg.norm(vectors, 2, axis=0), np.ones((3,)), atol=1e-6)
                for i in range(3):
                    x,y,z = vectors[:,i]
                    arrows[i].set_data([0.0, x], [0.0, y], [0.0, z])
                
                # Update odometry position
                cdata = odomPosDataPts[odomPosWindowsSize-nbOdomPosPoints:,:]
                scatPast.set_offsets(cdata)
                scatCur.set_offsets(cdata[-1,:])
                
                border = 0.5
                xlim = np.array([np.min(cdata[:,0])-border, np.max(cdata[:,0])+border])
                ylim = np.array([np.min(cdata[:,1])-border, np.max(cdata[:,1])+border])
                scale = np.max([xlim[1] - xlim[0], ylim[1] - ylim[0]])
                odomPosAx.set_xlim(xlim)
                odomPosAx.set_ylim(ylim)

                # Apply the rotation to the vector pointing in the forward direction (x-axis)
                x,y,z,w = odomOriData[n]
                quaternion = np.array([w,x,y,z])
                R = quat2mat(quaternion)
                direction = np.array([1.0, 0.0, 0.0])
                vector = np.dot(R, direction)
                vector /= np.linalg.norm(vector, 2)

                x, y = cdata[-1,0], cdata[-1,1]
                dx, dy = vector[0] * 0.05 * scale, vector[1] * 0.05 * scale
                lines = odomPosAx.plot([x, x+dx], [y, y+dy], c='r')
                
                # Write frame
                writer.grab_frame()
                
                lines.pop(0).remove()
    
            if n % 100 == 0:
                logger.info('Number of frames processed: %d (of %d)' % (n, nbFrames))
    
    elapsedTime = time.time() - startTime
    logger.info('FPS = %f frame/sec' % (nbFrames/elapsedTime))

def main(args=None):

    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default=None,
                      help='specify the path of the input hdf5 dataset file')
    parser.add_option("-o", "--output", dest="output", default='.',
                      help='specify the path of the output video file')
    parser.add_option("-z", "--fs-sync", dest="fsSync", type='int', default=50,
                      help='Specify the framerate (Hz) of the synchronized clock')
    parser.add_option("-d", "--fs-video", dest="fsVideo", type='int', default=20,
                      help='Specify the framerate (Hz) of the output videos')
    parser.add_option("-s", "--speed-up", dest="speedup", type='float', default=1.0,
                      help='Specify the speedup of the output videos')
    (options,args) = parser.parse_args(args=args)

    datasetPath = os.path.abspath(options.input)
    logger.info('Using input HDF5 dataset file: %s' % (datasetPath))

    outputVideoFile = os.path.abspath(options.output)    
    logger.info('Using output directory: %s' % (outputVideoFile))
    
    syncFs = options.fsSync
    safeBorder = 0.0
    
    with Hdf5Dataset(datasetPath, mode='r') as dataset:
        
        # Find valid clock
        validStartTime = np.max([clock[0] + safeBorder for _,_,clock in dataset.getAllClocks()])
        validStopTime = np.min([clock[-1] - safeBorder for _,_,clock in dataset.getAllClocks()])
        syncClock = np.linspace(validStartTime, validStopTime,
                                 num=int((validStopTime - validStartTime) * syncFs))
        logger.debug('Found valid clock range: [%f, %f] sec' % (validStartTime, validStopTime))
        
        processDataset(dataset, syncClock, outputVideoFile, options.fsVideo, options.speedup)
    
    logger.info('All done.')
                
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
