% Set everything up
clear; clc;
projectPath = fileparts(matlab.desktop.editor.getActiveFilename);
addpath(fullfile(projectPath,'utils')) % custom scripts

% If not done, clone and install buzcode in your computer: https://github.com/buzsakilab/buzcode
% Warning! Look that it needs to be compiled!
pathBuzcode = '/path/to/buzcode/';
% Equally, clone fooof_mat in your computer: https://github.com/fooof-tools
% Warning! Look that it needs some python requirements. It might be useful to check:
% https://github.com/fooof-tools/fooof_mat/issues/30
pathFooof = '/path/to/fooof_mat/';
% Then, add those paths
addpath(genpath(pathBuzcode))
addpath(genpath(pathFooof))

%% Load the example data containing one sleep epoch 

% Path to example data
sessionName = 'ER20_micro1_epoch1';
dataPath = fullfile(fileparts(projectPath),'data');

% lfp: time x 8 micoriwres
% srate: sampling frequency
% detectionChannel: microwire channel used for ripple detection
load(fullfile(dataPath,[sessionName '.mat']))
detectionChannelThisStudy = detectionChannel;

%% 1. Rough ripple detection with all channels

% Set IED parameters
threshIED = [3 10]; 	% lower and upper threshold for IED detection. [3 15] for signals with many false-positives % 
bandIED = [20 80]; 		% bandpass filter frequencies for IED detection, in Hz 
EMGthresh = [];         % 0-1 threshold of EMG to exclude noise
intervals = [];         % interval used to compute normalization (default = all)
alignIED = true; 		% aligning to the peak. It is optional for detection, but recommended for ripmap curation
side = 1; 				% side of the IEDs: 1 for alignment to negative peak, -1 for positive peak 
% Set SWR parameters
threshHFO = [2 5]; 		% lower and upper threshold for SWR detection
bandHFO = [80 250]; 	% bandpass filter frequencies for SWR detection, in Hz
verify = false; 		% visually inspect the detected events
clipping = false; 		% clip the extremes following Norman et al. 
outputstruct = true; 	% generate output structure
% Set snipets parameters
win = [0.1 0.1];
tt = -win(1):1/srate:win(2);
% Initialize structure
allHFOs = struct();
allIEDs = struct();
% Number of channels
nChannels = size(lfp,2);
% Frequencies to perform power spectrum  on
frequencies = 1:200; % Hz

% Loop over the channels
for iChannel = 1:nChannels
    fprintf('Processing channel %d...\n', iChannel)

    % Get lfp from channel
    data = lfp(:,iChannel);

    % IED detection: we use the findRipple function from buzcode with
    % different band (bandIED) and threshold (thresholdIED) to find IEDs
    % and clean them from the signal before ripple detection
    timestamps = (0:1/srate:length(data)/srate-1/srate)';
    IEDs = findIEDs(data, timestamps, srate, threshIED, bandIED, EMGthresh, intervals);
    % Append to structure
    allIEDs(iChannel).peaks = IEDs.peaks;
    if alignIED
        allIEDs(iChannel).peaks = alignpeaks(double(lfp), srate, side, IEDs.peaks);
    end

    % Detect ripples: we use findHFOs (from bz_FindRipples, modified by JS
    % to detect ripples from ECoG data), and we include the IED times to
    % avoid using those intervals for SWR detection
    HFOs = findHFOs( data, 'frequency', srate, 'passband', bandHFO,'thresholds', threshHFO, ...,
        'clipping', clipping, 'verify',verify,'outputstruct', outputstruct,  'IEDs', IEDs.peaks);
    % Append to structure
    allHFOs(iChannel).chan = iChannel;
    allHFOs(iChannel).peaks = HFOs.peaks;

    % Generate events 
    snippets = [];
    powspctrm = [];
    if ~isempty(HFOs.peaks)
        % Get snippets
        [snippets] = getsnippets(data, win, HFOs.peaks,srate, 1);
        % get power spectrum
        [powspctrm] = pwelch(snippets', 100, [], frequencies, srate);
    end

    % Save info
    allHFOs(iChannel).snippets = snippets;
    allHFOs(iChannel).powspctrm = powspctrm;
    allHFOs(iChannel).avg_powspctrm = mean(powspctrm,2);
end
close


%% 2. Extract peak frequency

% Minimum threshold for the power spectrum SWR peak (after foof correction)
% to be considered significant
threshold = 0.2;

% Compute ripple frequencies..
[stats, corrected] = computeripplestats([allHFOs.avg_powspctrm], frequencies, threshold);
rippleFreqs = [stats.max];
% ... and amplitudes
rippleAmps = zeros(size(rippleFreqs));
for iChannel = 1:length(stats)
    idx = ismember(stats(iChannel).freqs, rippleFreqs(iChannel));
    rippleAmps(iChannel) = stats(iChannel).amp(idx);
end


%% 3. Select the channel

minFreq = 60; % Hz
maxFreq = 150; % Hz

% Check what channels have peak frequencies in the ripple band
rippleChannels = find( (rippleFreqs>=minFreq) & (rippleFreqs<maxFreq) );
%.. and what are their amplitudes
rippleChanAmps = rippleAmps(rippleChannels);
% Then, select the one with the highest amplitude
[~, iBestChannel] = max(rippleChanAmps);
detectionChannel = rippleChannels(iBestChannel);
% Show:
disp(['Recommended channel for ripple detection: channel ' num2str(detectionChannel)])
disp(['Channel used for ripple detection in this study: channel ' num2str(detectionChannelThisStudy)])

%%

% Plot the average HFO and powerspectrum of each channel
figure('Units','normalized', 'Position',[0.1 0.2 0.6 0.5])
ts = ((1:size(allHFOs(iChannel).snippets,2)) - size(allHFOs(iChannel).snippets,2)/2)/srate*1000;
for iChannel = 1:8
    % Plot the LFP shape in the left
    subplot(4,4,2*iChannel-1)
    plot(ts, mean(allHFOs(iChannel).snippets,1), 'k', 'LineWidth',1+2*(iChannel==detectionChannel))
    xlim([ts(1)-1, ts(end)+1])
    if iChannel >= 7, xlabel('Time (ms)'), end
    title(['chan ', num2str(iChannel)])
    % Plot the power spectrum in the right
    subplot(4,4,2*iChannel), hold on
    plot(frequencies, log10(allHFOs(iChannel).avg_powspctrm), '--b')
    plot(frequencies,corrected(:,iChannel), 'k','LineWidth',1+2*(iChannel==detectionChannel))
    xline(rippleFreqs(iChannel))
    xlim([frequencies(1), frequencies(end)])
    if iChannel == 1, legend({'raw','corrected','freq'}), end
    if iChannel >= 7, xlabel('Frequency (Hz)'), end
    title(['Peak freq: ', num2str(rippleFreqs(iChannel))])
end
sgtitle(strrep(sessionName,'_','-'))

%% 4. Save the data

% Append the 'detectionChannel' to the structure
save(fullfile(dataPath,[sessionName '.mat']), 'detectionChannel', '-append')

% If wanted, save also the detections
HFOs = allHFOs(detectionChannel).peaks/srate;
IEDs = allIEDs(detectionChannel).peaks';
save(fullfile(dataPath,[sessionName '_automatic_detections.mat']), 'HFOs', 'IEDs')



