% =========================================================================
% Project Title   : Multichannel Data Visualization & Smoothing (18 Channels)
% Data Source     : Recorded TXT File
% Language        : MATLAB
% Author          : Madou FALL
% Date            : June 20, 2025
% -------------------------------------------------------------------------
% Description     :
%   This MATLAB script processes and visualizes previously recorded analog
%   data from 18 channels. The data is assumed to be exported from a 
%   Teensy-based acquisition system and saved in a CSV-like `.txt` file.
%
%   Key Features :
%     - Loads multichannel data from a labeled text file
%     - Applies exponential smoothing to selected channels (17 & 18)
%     - Visualizes:
%         • Raw signal from channel 17 (Weighted Sum)
%         • Raw signal from channel 18 (Mean)
%         • Smoothed versions of both channels
%         • All 18 channels in raw and high-pass (zero-mean) form
%     - High-pass filtering simulated by subtracting the mean of each channel
%
%   Functions :
%     - `exponential_smoothing(data, alpha)`:
%         Applies recursive low-pass filtering to smooth noisy data.
%
%   Parameters :
%     - `alpha_M` and `alpha_S`: Smoothing factors for exponential smoothing
%
%   Visualization :
%     - Line plots with labeled axes and legends
%     - Color-coded channel representations using MATLAB’s `lines` colormap
%
%   Usage Notes :
%     - Make sure the data file path is correctly set (see `readmatrix` path)
%     - Compatible with MATLAB R2020 and later
%
% =========================================================================


%%                           DATA LOADING SECTION                         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read data from a text file. The first line (header) is automatically skipped.
% The resulting 'data' matrix contains recordings from 18 channels.
data = readmatrix('data/Pe12/Pe12_SwipeRight.txt');


%%                   EXPONENTIAL SMOOTHING FUNCTION                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function smoothedData = exponential_smoothing(data, alpha)
    % Initialize the smoothed data array with the same size as input data
    smoothedData = zeros(size(data));
    
    % Set the first value of the smoothed data equal to the first raw data point
    smoothedData(1) = data(1);
    
    % Apply exponential smoothing recursively to the rest of the data
    for i = 2:length(data)
        % Smoothing formula:
        % smoothed(i) = alpha * data(i) + (1 - alpha) * smoothed(i - 1)
        smoothedData(i) = alpha * data(i) + (1 - alpha) * smoothedData(i - 1);
    end
end


%%                   CHANNEL 18 VISUALIZATION (MEAN)                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract channel 18, assumed to represent mean values
M = data(:, 18);

% Define the smoothing factor (between 0 and 1)
alpha_M = 0.3;

% Apply exponential smoothing to channel 18
M_filt = exponential_smoothing(M, alpha_M);

% Plot raw channel 18 data
figure;
plot(M, 'LineWidth', 2, 'Color', 'b');
xlabel('Samples');
ylabel('Amplitude');
title('Channel 18 - Raw Data (Mean)');
grid on;

% Plot smoothed channel 18 data
figure;
plot(M_filt, 'LineWidth', 2, 'Color', 'b');
xlabel('Samples');
ylabel('Amplitude');
title('Channel 18 - Smoothed Data (Mean)');
grid on;


%%               CHANNEL 17 VISUALIZATION (WEIGHTED SUM)                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract channel 17, assumed to represent a weighted sum
Sn = data(:, 17);

% Define the smoothing factor
alpha_S = 0.3;

% Apply exponential smoothing to channel 17
S_filt = exponential_smoothing(Sn, alpha_S);

% Plot raw channel 17 data
figure;
plot(Sn, 'LineWidth', 2, 'Color', 'b');
xlabel('Samples');
ylabel('Amplitude');
title('Channel 17 - Raw Data (Weighted Sum)');
grid on;

% Plot smoothed channel 17 data
figure;
plot(S_filt, 'LineWidth', 2, 'Color', 'b');
xlabel('Samples');
ylabel('Amplitude');
title('Channel 17 - Smoothed Data (Weighted Sum)');
grid on;


%%                   ALL CHANNELS VISUALIZATION                           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create a figure to visualize all 18 channels together
figure;
hold on;  % Allow multiple plots on the same figure

% Generate 18 distinct colors using the 'lines' color map
colors = lines(18);

% Plot each channel with a unique color
for i = 1:18
    plot(data(:, i), 'LineWidth', 1.5, 'Color', colors(i, :));
end

% Add labels, title, and legend
xlabel('Samples');
ylabel('Amplitude');
title('All 18 Channels - Raw Data');
legend("Channel " + string(1:18), 'Location', 'eastoutside');
grid on;



%%        ALL CHANNELS VISUALIZATION - HIGH-PASS FILTERED                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create a copy of the data for high-pass filtering (zero-mean)
data_hp = data;

% Remove DC offset (mean) from each channel to apply high-pass effect
for i = 1:18
    data_hp(:, i) = data(:, i) - mean(data(:, i));
end

% Create a new figure for the high-pass filtered data
figure;
hold on;

% Generate 18 distinct colors
colors = lines(18);

% Plot each filtered channel with a unique color
for i = 1:18
    plot(data_hp(:, i), 'LineWidth', 1.5, 'Color', colors(i, :));
end

% Add labels, title, and legend
xlabel('Samples');
ylabel('Amplitude');
title('All 18 Channels - High-Pass Filtered (Zero Mean)');
legend("Channel " + string(1:18), 'Location', 'eastoutside');
grid on;

%%