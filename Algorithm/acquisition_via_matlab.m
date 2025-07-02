% =========================================================================
% Project Title   : Real-Time Labeling and Signal Acquisition (18 Channels)
% Data Source     : Teensy 4.1 via Serial (TXT Format)
% Interface       : Live Serial Stream with Keyboard Labeling
% Sampling Rate   : ~200 Hz (depends on Teensy configuration)
% Author          : Madou FALL
% Date            : June 21, 2025
% -------------------------------------------------------------------------
% Description     :
%   This MATLAB script provides real-time acquisition, visualization, and 
%   labeling of multichannel analog signals streamed from a Teensy 4.1. 
%   The system captures 18 analog channels over a serial connection and 
%   allows the user to label the incoming data in real time via keyboard 
%   shortcuts.
%
%   Key Features :
%     - Live signal display of Channel 17 (e.g., weighted sum)
%     - Keyboard-controlled label switching:
%         [N] Neutral
%         [E] InOut
%         [S] Still
%         [G] SwipeLeft
%         [D] SwipeRight
%         [C] Click
%         [A] Toggle recording (pause/resume)
%     - Real-time TXT recording with associated label
%     - Global TXT recording (all samples, including Neutral)
%     - Participant-based folder structure and automatic file handling
%     - Automatic insertion of TXT headers if file is new
%     - Label override condition: "Neutral" if channels 17 & 18 are in range
%
% =========================================================================

%% --- REAL-TIME LABEL MANAGEMENT ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Declare global variables for label, recording state, and file management
global current_label enregistrement_actif fichier fichier_all folder participant_id;

% Default label at startup
current_label = "Neutral";

% Indicates if recording is active
enregistrement_actif = true;

% File handles (specific and global)
fichier = [];
fichier_all = [];

%% --- PARTICIPANT CONFIGURATION AND FOLDER CREATION ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ask for the participant's ID
participant_id = input("ID du participant : ", 's');

% Create a folder for this participant if it doesn't exist
folder = fullfile("data", participant_id);
if ~exist(folder, 'dir')
    mkdir(folder);
end

% Create (or open) the global data file for this participant
nom_fichier_all = fullfile(folder, participant_id + "_ALL.txt");
fichier_all = fopen(nom_fichier_all, 'a'); % Append mode

% Write the header if the file is empty
if ftell(fichier_all) == 0
    headers = join(["A" + string(1:18), "label"], ',');
    fprintf(fichier_all, '%s\n', headers);
end

%% --- KEYBOARD CALLBACK FUNCTION FOR LABEL CHANGING ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function is called when a key is pressed in the figure
function keyCallback(~, event)
    global current_label enregistrement_actif fichier folder participant_id;

    % Update label depending on key pressed
    switch lower(event.Key)
        case 'n', current_label = "Neutral";
        case 'e', current_label = "InOut";
        case 's', current_label = "Still";
        case 'g', current_label = "SwipeLeft";
        case 'd', current_label = "SwipeRight";
        case 'c', current_label = "Click";
        case 'a' % Toggle recording on/off
            enregistrement_actif = ~enregistrement_actif;
            disp("Recording: " + string(enregistrement_actif));
        otherwise
            disp("Unrecognized key");
            return;
    end

    % Close previous label file if open
    if ~isempty(fichier)
        fclose(fichier);
    end

    % Open a new file for the current label
    nom_fichier = fullfile(folder, participant_id + "_" + current_label + ".txt");
    disp("File opened: " + fullfile(pwd, nom_fichier));
    fichier = fopen(nom_fichier, 'a');

    % Write header if the file is empty
    if ftell(fichier) == 0
        headers = join(["A" + string(1:18), "label"], ',');
        fprintf(fichier, '%s\n', headers);
    end

    disp("Current label: " + current_label);
end


%% --- SERIAL PORT INITIALIZATION ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set up the serial port connection to Arduino or other sensor
s = serialport("COM6", 250000);
configureTerminator(s, "LF"); % End of line character
flush(s); % Clear buffer

%% --- FIGURE AND LIVE PLOT SETUP ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create a figure window for live plotting
fig = figure;
set(fig, 'KeyPressFcn', @keyCallback); % Link key press to the callback

% Set up animated line to display A17 (Sn)
h = animatedline('Color', 'b', 'LineWidth', 2);
xlabel('Samples');
ylabel('Weighted Sum (Sn)');
title('Real-Time Sn');
grid on;

% Sample counter
i = 0;

%% --- MAIN LOOP: DATA READING, LABELING, RECORDING ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Main loop: runs as long as the figure window is open
while ishandle(fig)
    if s.NumBytesAvailable > 0
        line = readline(s); % Read one line from serial port
        values = str2double(split(line, ",")); % Convert to numeric array

        if numel(values) >= 18
            i = i + 1;
            addpoints(h, i, values(17)); % Add A17 to live plot
            drawnow limitrate; % Efficient plot update

            % Force "Neutral" label if values fall within a specific range
            if values(17) >= 3 && values(17) <= 25 && ...
               values(18) >= 0 && values(18) <= 22
                label_to_write = "Neutral";
            else
                label_to_write = current_label;
            end

            % Write to label-specific file if recording is active
            if enregistrement_actif && ~isempty(fichier)
                ligne_avec_label = [string(values'), label_to_write];
                fprintf(fichier, '%s\n', join(ligne_avec_label, ','));
            end

            % Always write to the global file
            if ~isempty(fichier_all)
                ligne_all = [string(values'), label_to_write];
                fprintf(fichier_all, '%s\n', join(ligne_all, ','));
            end
        end
    end
end


%% --- CLEANUP ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Close any open files at the end
if ~isempty(fichier)
    fclose(fichier);
end
if ~isempty(fichier_all)
    fclose(fichier_all);
end
