% Copyright (c) 2018, Simon Brodeur
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright 
%    notice, this list of conditions and the following disclaimer.
%   
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its contributors
%    may be used to endorse or promote products derived from this software without
%    specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
% IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
% INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
% NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
% OF SUCH DAMAGE.

clc;
close all;

% Please set the correct path to the Create HDF5 dataset with the variable below:
inputDatasetPath = '/home/simon/Data/Create/';

inputHdf5File = fullfile(inputDatasetPath, 'E3/C1-3036/E3_C13036_S1_20161222T101923.h5');

plotVideo(inputHdf5File);
plotAudio(inputHdf5File);
plotAudioSpectrogram(inputHdf5File);
plotMotor(inputHdf5File);
plotImu(inputHdf5File);
plotOdometryPosition(inputHdf5File);
plotOdometryVelocities(inputHdf5File);
plotTemperature(inputHdf5File);
plotPressure(inputHdf5File);
plotBattery(inputHdf5File);
plotCollisionRangers(inputHdf5File);
plotCollisionSwitches(inputHdf5File);
plotOpticalFlow(inputHdf5File);

disp('All done.');

function [ data, clock, shape ] = loadHDF5VideoData( filename, name, group )
    data = hdf5read(filename, strcat('/', group, '/', name, '/data'));
    clock = hdf5read(filename, strcat('/', group, '/', name, '/clock'));
    shape = hdf5read(filename, strcat('/', group, '/', name, '/shape'));
end

function [ data, clock ] = loadHDF5Data( filename, name, group )
    data = hdf5read(filename, strcat('/', group, '/', name, '/data'));
    clock = hdf5read(filename, strcat('/', group, '/', name, '/clock'));
end

function [ img ] = jpegread( data )
    %Adapted from: https://stackoverflow.com/questions/18659586/from-raw-bits-to-jpeg-without-writing-into-a-file

    % Decode image stream using Java
    jImg = javax.imageio.ImageIO.read(java.io.ByteArrayInputStream(data));
    h = jImg.getHeight;
    w = jImg.getWidth;

    % Convert Java Image to MATLAB image
    p = reshape(typecast(jImg.getData.getDataStorage, 'uint8'), [3,w,h]);
    img = cat(3, ...
            transpose(reshape(p(3,:,:), [w,h])), ...
            transpose(reshape(p(2,:,:), [w,h])), ...
            transpose(reshape(p(1,:,:), [w,h])));
end

function plotVideo(filename)

    figure('color','white');

    [ldata, lclock, lshape] = loadHDF5VideoData(filename, 'left', 'video');
    [rdata, rclock, rshape] = loadHDF5VideoData(filename, 'right', 'video');
    
    % NOTE: only show the frames at time 100.0 sec
    frameTime = 100.0;
    [~, lframeIndex] = min(abs(lclock - frameTime));
    [~, rframeIndex] = min(abs(rclock - frameTime));
    
    imgLeft = jpegread(ldata(1:lshape(lframeIndex), lframeIndex));
    imgRight = jpegread(rdata(1:rshape(rframeIndex), rframeIndex));

    subplot(1,2,1);
    imshow(imgLeft);
    subplot(1,2,2);
    imshow(imgRight);
    
end

function sclock = getPerSampleClock(data, clock, fs)
    chunkSize = size(data,1);
    sclock = zeros(size(data));
    for n = 1:size(data,2)
        sclock(:, n) = (clock(n) - colon(chunkSize,-1,1) ./ fs).';
    end
    sclock = sclock(:);
end    

function plotAudio(filename)

    figure('color','white');

    % NOTE: only show this small time interval
    xrange = [81.0, 87.0];

    fs = 16000.0;
    
    [ldata, lclock] = loadHDF5Data(filename, 'left', 'audio');
    lclock = getPerSampleClock(ldata, lclock, fs);
    ldata = double(ldata(:)) ./ double(intmax('int16'));

    [rdata, rclock] = loadHDF5Data(filename, 'right', 'audio');
    rclock = getPerSampleClock(rdata, rclock, fs);
    rdata = double(rdata(:)) ./ double(intmax('int16'));

    % Left channel
    subplot(2,1,1);
    title('Audio signal (Left channel)');
    ylabel('Amplitude');

    plot(lclock, ldata);
    ymax = max(abs(ldata));
    xlim(xrange);
    ylim([-ymax, ymax]);

    % Right channel
    subplot(2,1,2);
    title('Audio signal (right channel)');
    xlabel('Time [sec]');
    ylabel('Amplitude');

    plot(rclock, rdata);
    ymax = max(abs(rdata));
    xlim(xrange);
    ylim([-ymax, ymax]);
    
end

function plotAudioSpectrogram(filename)

    figure('color','white');
    
    % NOTE: only show this small time interval (mins)
    xrange = [1.35, 1.45];

    fs = 16000.0;
    
    [ldata, lclock] = loadHDF5Data(filename, 'left', 'audio');
    lclock = getPerSampleClock(ldata, lclock, fs);
    ldata = double(ldata(:)) ./ double(intmax('int16'));

    [rdata, rclock] = loadHDF5Data(filename, 'right', 'audio');
    rclock = getPerSampleClock(rdata, rclock, fs);
    rdata = double(rdata(:)) ./ double(intmax('int16'));

    % NOTE: make sure left and right channels are approximately aligned in time, 
    %       since the 'spectrogram' of MATLAB doesn't allow to specify the times.
    nbTrimSamples = round(abs(rclock(1) - lclock(1)) .* fs);
    if rclock(1) > lclock(1)
        ldata = ldata(nbTrimSamples:end);
    else
        rdata = rdata(nbTrimSamples:end);
    end
    
    subplot(2,1,1);
    spectrogram(ldata, hanning(256), 128, 256, fs, 'yaxis', 'MinThreshold', -120.0);
    xlim(xrange);
    title('Power spectrum density (left channel)');

    subplot(2,1,2);
    spectrogram(rdata, hanning(256), 128, 256, fs, 'yaxis', 'MinThreshold', -120.0);
    xlim(xrange);
    title('Power spectrum density (right channel)');
    
end

function plotMotor(filename)

    figure('color','white');
    
    % NOTE: only show the first 60 seconds
    xrange = [0, 60];

    [data, clock] = loadHDF5Data(filename, 'linear_velocity', 'motor');
    
    ldata = data(1,:); % Left motor
    rdata = data(2,:); % Right motor
    
    % Motors
    subplot(1,1,1);
    hold on;
    title('Motors');
    plot(clock, ldata, '-b', 'DisplayName', 'Left motor');
    plot(clock, rdata, '-g', 'DisplayName', 'Right motor');
    xlabel('Time [sec]');
    ylabel('Linear velocity [m/sec]');
    xlim(xrange);
    legend('show');
    
end

function plotImu(filename)

    figure('color','white');

    % NOTE: only show the first 60 seconds
    xrange = [0, 60];

    % Angular velocity data
    [data, clock] = loadHDF5Data(filename, 'angular_velocity', 'imu');
    angularVelXData = data(1,:); % X-axis
    angularVelYData = data(2,:); % Y-axis
    angularVelZData = data(3,:); % Z-axis

    subplot(3,1,1);
    hold on;
    title('IMU gyroscope');
    plot(clock, angularVelXData, '-b', 'DisplayName', 'X-axis');
    plot(clock, angularVelYData, '-g', 'DisplayName', 'Y-axis');
    plot(clock, angularVelZData, '-r', 'DisplayName', 'Z-axis');
    xlabel('Time [sec]');
    ylabel('Angular velocity [rad/sec]');
    xlim(xrange);
    ylim([-2.5, 2.5]);
    legend('show');
    
    % Linear acceleration
    [data, clock] = loadHDF5Data(filename, 'linear_acceleration', 'imu');
    linearAccelXData = data(1,:); % X-axis
    linearAccelYData = data(2,:); % Y-axis
    linearAccelZData = data(3,:); % Z-axis

    subplot(3,1,2);
    hold on;
    title('IMU linear acceleration');
    plot(clock, linearAccelXData, '-b', 'DisplayName', 'X-axis');
    plot(clock, linearAccelYData, '-g', 'DisplayName', 'Y-axis');
    plot(clock, linearAccelZData, '-r', 'DisplayName', 'Z-axis');
    xlabel('Time [sec]');
    ylabel('Linear acceleration [m/sec^2]');
    xlim(xrange);
    ylim([-15, 30]);
    legend('show');
    
    % Magnetic field
    [data, clock] = loadHDF5Data(filename, 'magnetic_field', 'imu');

    % Note: convert from T to uT
    data = data * 1e6;

    magFieldXData = data(1,:); % X-axis
    magFieldYData = data(2,:); % Y-axis
    magFieldZData = data(3,:); % Z-axis

    subplot(3,1,3);
    hold on;
    title('IMU magnetic field');
    plot(clock, magFieldXData, '-b', 'DisplayName', 'X-axis');
    plot(clock, magFieldYData, '-g', 'DisplayName', 'Y-axis');
    plot(clock, magFieldZData, '-r', 'DisplayName', 'Z-axis');
    xlabel('Time [sec]');
    ylabel('Magnetic field [uT]');
    xlim(xrange);
    legend('show');

end

function plotOdometryPosition(filename)

    figure('color','white');

    [data, ~] = loadHDF5Data(filename, 'position', 'odometry');
    x = data(1,:);
    y = data(2,:);

    subplot(1,1,1)
    hold on;
    scatter(x, y, 10, 'filled');
    scatter(x(end), y(end), 50, [1.0,0.0,0.0], 'filled');

    grid on;
    title('Position')
    xlabel("x [meter]")
    ylabel("y [meter]")

end

function plotOdometryVelocities(filename)

    figure('color','white');

    % NOTE: only show the first 60 seconds
    xrange = [0, 60];

    % Angular velocity data
    [data, clock] = loadHDF5Data(filename, 'angular_velocity', 'odometry');
    angularVelXData = data(1,:); % X-axis
    angularVelYData = data(2,:); % Y-axis
    angularVelZData = data(3,:); % Z-axis

    subplot(2,1,1);
    hold on;
    title('Odometry angular velocity');
    plot(clock, angularVelXData, '-b', 'DisplayName', 'X-axis');
    plot(clock, angularVelYData, '-g', 'DisplayName', 'Y-axis');
    plot(clock, angularVelZData, '-r', 'DisplayName', 'Z-axis');
    xlabel('Time [sec]');
    ylabel('Angular velocity [rad/sec]');
    xlim(xrange);
    legend('show');
    
    % Linear velocity
    [data, clock] = loadHDF5Data(filename, 'linear_velocity', 'odometry');
    linearVelXData = data(1,:); % X-axis
    linearVelYData = data(2,:); % Y-axis
    linearVelZData = data(3,:); % Z-axis

    subplot(2,1,2);
    hold on;
    title('Odometry linear velocity');
    plot(clock, linearVelXData, '-b', 'DisplayName', 'X-axis');
    plot(clock, linearVelYData, '-g', 'DisplayName', 'Y-axis');
    plot(clock, linearVelZData, '-r', 'DisplayName', 'Z-axis');
    xlabel('Time [sec]');
    ylabel('Linear velocity [m/sec]');
    xlim(xrange);
    legend('show');

end

function plotTemperature(filename)

    figure('color','white');

    [data, clock] = loadHDF5Data(filename, 'temperature', 'imu');
    
    subplot(1,1,1);
    plot(clock, data);
    title('IMU temperature');
    xlabel("Time [sec]");
    ylabel("Temperature [celcius]");
    ylim([15.0, 30.0]);

end

function plotPressure(filename)

    figure('color','white');

    % NOTE: only show the first 60 seconds
    xrange = [0, 60];

    [data, clock] = loadHDF5Data(filename, 'pressure', 'imu');
    
    % Note: convert from Pa to kPa
    data = data / 1000.0;

    subplot(1,1,1);
    plot(clock, data);
    title('Atmospheric pressure')
    xlabel("Time [sec]")
    ylabel("Pressure [kPa]")
    xlim(xrange)

end

function plotBattery(filename)

    figure('color','white');

    [data, clock] = loadHDF5Data(filename, 'charge', 'battery');
    voltage = data(1,:);        % Voltage
    current = data(2,:);        % Current
    charge = data(3,:);         % Charge remaining
    capacity = data(4,:);       % Capacity (estimated)
    percentage = data(5,:);     % Percentage remaining

    fprintf('Battery capacity estimated at %0.2f Ah\n', mean(capacity));
    
    subplot(2,2,1);
    plot(clock, voltage);
    title('Battery voltage');
    xlabel("Time [sec]");
    ylabel("Voltage [V]");
    ylim([10.0, 18.0]);

    subplot(2,2,2);
    plot(clock, current);
    title('Battery current');
    xlabel("Time [sec]");
    ylabel("Current [A]");
    ylim([-2.0, 2.0]);

    subplot(2,2,3);
    plot(clock, charge);
    title('Battery charge remaining');
    xlabel("Time [sec]");
    ylabel("Charge [Ah]");
    ylim([2.0, max(capacity)]);

    subplot(2,2,4);
    plot(clock, percentage * 100.0);
    title('Battery percentage remaining');
    xlabel("Time [sec]");
    ylabel("Percentage [%]");
    ylim([75.0, 100.0]);
    
end

function plotCollisionRangers(filename)

    figure('color','white');
    
    % NOTE: only show this small time interval
    xrange = [400, 550];
    
    [data, clock] = loadHDF5Data(filename, 'range', 'collision');
    wall = data(1,:);
    cliffLeft = data(2,:);
    cliffFrontLeft = data(3,:);
    cliffFrontRight = data(4,:);
    cliffRight = data(5,:); 

    subplot(1,1,1);
    hold on;
    plot(clock, wall, '-r', 'DisplayName', 'Wall');
    plot(clock, cliffLeft, '-g', 'DisplayName', 'Cliff left');
    plot(clock, cliffFrontLeft, '-b', 'DisplayName', 'Cliff front-left');
    plot(clock, cliffFrontRight, '-c', 'DisplayName', 'Cliff front-right');
    plot(clock, cliffRight, '-m', 'DisplayName', 'Cliff right');
    title('IR range sensors');
    xlabel("Time [sec]");
    ylabel("Amplitude");
    legend('show');
    xlim(xrange);
end

function plotCollisionSwitches(filename)
    
    figure('color','white');
    
    % NOTE: only show this small time interval
    xrange = [400, 600];
    
    [data, clock] = loadHDF5Data(filename, 'switch', 'collision');
    bumperLeft = data(1,:);
    bumperRight = data(2,:);
    wall = data(10,:);

    % Bumper-related
    subplot(3,1,1);
    hold on;
    plot(clock, bumperLeft, '-r', 'DisplayName', 'Bumper left');
    plot(clock, bumperRight, '-g', 'DisplayName', 'Bumper right');
    plot(clock, wall, '-b', 'DisplayName', 'Wall');
    title('Contact sensors');
    xlabel("Time [sec]");
    ylabel("Activation");
    legend('show');
    xlim(xrange);
    ylim([-0.1, 1.1]);
    yticks([0.0, 1.0]);
    yticklabels({'Off', 'On'});
    
    % Cliff-related
    cliffLeft = data(6,:);
    cliffFrontLeft = data(7,:);
    cliffFrontRight = data(8,:);
    cliffRight = data(9,:);

    subplot(3,1,2);
    hold on;
    plot(clock, cliffLeft, '-r', 'DisplayName', 'Cliff left');
    plot(clock, cliffFrontLeft, '-g', 'DisplayName', 'Cliff front-left');
    plot(clock, cliffRight, '-b', 'DisplayName', 'Cliff front-right');
    plot(clock, cliffFrontRight, '-c', 'DisplayName', 'Cliff right');
    title('Cliff sensors');
    xlabel("Time [sec]");
    ylabel("Activation");
    legend('show');
    xlim(xrange);
    yticks([0.0, 1.0]);
    yticklabels({'Off', 'On'});

    % Wheel-drop related
    caster = data(3,:);
    wheelLeft = data(4,:);
    wheelRight = data(5,:);

    subplot(3,1,3);
    hold on;
    plot(clock, caster, '-r', 'DisplayName', 'Caster');
    plot(clock, wheelLeft, '-g', 'DisplayName', 'Wheel left');
    plot(clock, wheelRight, '-b', 'DisplayName', 'Wheel right');
    title('Wheel drop sensors');
    xlabel("Time [sec]");
    ylabel("Activation");
    legend('show');
    xlim(xrange);
    yticks([0.0, 1.0]);
    yticklabels({'Off', 'On'});
    
end

function plotOpticalFlow(filename)

    figure('color','white');

    [ldata, lclock] = loadHDF5Data(filename, 'left', 'optical_flow');
    [rdata, rclock] = loadHDF5Data(filename, 'right', 'optical_flow');
    
    % NOTE: only show the frames at time 100.0 sec
    frameTime = 100.0;
    [~, lframeIndex] = min(abs(lclock - frameTime));
    [~, rframeIndex] = min(abs(rclock - frameTime));

    lflow = ldata(:,:,:,lframeIndex);
    rflow = rdata(:,:,:,rframeIndex);

    lflow(1,1,:);       % Corresponds to the bottom-left corner of the image
    lflow(end,1,:);     % Corresponds to the top-left corner of the image
    lflow(1,end,:);     % Corresponds to the bottom-right corner of the image
    lflow(end,end,:);   % Corresponds to the top-right corner of the image

    % NOTE: hardcoded image size and border used to compute optical flow
    ih = 240; 
    iw = 320;
    border = 0.1;
    h = size(lflow, 3);
    w = size(lflow, 2);
    [y, x] = meshgrid(linspace(border*ih, (1.0-border)*ih, h), ...
                    linspace(border*iw, (1.0-border)*iw, w));

    % Left channel
    subplot(1,2,1);
    title('Optical flow (left camera)');
    quiver(x, y, squeeze(lflow(1,:,:)), squeeze(-lflow(2,:,:)), 1, ...
           'color', [0 0 0], 'linewidth', 1);
    daspect([1 1 1])
    axis off;

    % Right channel
    subplot(1,2,2);
    title('Optical flow (right camera)');
    quiver(x, y, squeeze(rflow(1,:,:)), squeeze(-rflow(2,:,:)), 1, ...
           'color', [0 0 0], 'linewidth', 1);
    daspect([1 1 1])
    axis off;
    
end
