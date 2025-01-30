function [ripples, squaredSignal, timestamps] = findHFOs(lfp, varargin)
%findHFOs - Find hippocampal ripples (100~200Hz oscillations).
%
% USAGE
%    [ripples] = findHFOs(lfp,<options>)
%
%    Ripples are detected using the normalized squared signal (NSS) by
%    thresholding the baseline, merging neighboring events, thresholding
%    the peaks, and discarding events with excessive duration.
%    Thresholds are computed as multiples of the standard deviation of
%    the NSS. Alternatively, one can use explicit values, typically obtained
%    from a previous call.  The estimated EMG can be used as an additional
%    exclusion criteria
%
% INPUTS - note these are NOT name-value pairs... just raw values
%    lfp            unfiltered LFP signal (time x 1).
%    <options>      optional list of property-value pairs (see tables below)
%
%    =========================================================================
%     Properties    Values
%    -------------------------------------------------------------------------
%     'thresholds'  thresholds for ripple beginning/end and peak, in multiples
%                   of the stdev (default = [2 5]); must be integer values
%     'durations'   min inter-ripple interval and max ripple duration, in ms
%                   (default = [30 250]). 
%     'restrict'    interval used to compute normalization (default = all)
%     'frequency'   sampling rate (in Hz) (default = 512 Hz)
%     'stdev'       reuse previously computed stdev
%     'passband'    N x 2 matrix of frequencies to filter for ripple detection 
%                   (default = [130 200])
%     'EMGThresh'   0-1 threshold of EMG to exclude noise
%     'minInterval' mininum interval between events. events less than this
%                   value will be merged (default = 30 ms)
%     'verify'      visually inspect the detected events (default = false)
%     'clipping'    clip the extremes following Norman et al. 
%                   (default = false)
%     'IEDs'        replace squared signals around IEDs to NaN
%    =========================================================================
%
% OUTPUT
%
%    ripples        buzcode format .event. struct with the following fields
%                   .timestamps        Nx2 matrix of start/stop times for
%                                      each ripple
%                   .detectorName      string ID for detector function used
%                   .peaks             Nx1 matrix of peak power timestamps 
%                   .stdev             standard dev used as threshold
%                   .noise             candidate ripples that were
%                                      identified as noise and removed
%                   .peakNormedPower   Nx1 matrix of peak power values
%                   .detectorParams    struct with input parameters given
%                                      to the detector
% SEE ALSO
%

% Copyright (C) 2004-2011 by Michaël Zugaro, initial algorithm by Hajime Hirase
% edited by David Tingley, 2017
% edited by Jiyun Shin for ECoG, 2022


% Default values
p = inputParser;
addParameter(p,'thresholds',[2 5],@isvector);
addParameter(p,'durations',[30 250],@isnumeric); % in ms
addParameter(p,'restrict',[],@isnumeric);
addParameter(p,'frequency',512,@isnumeric);
addParameter(p,'stdev',[],@isnumeric);
addParameter(p,'passband',[130 200],@isnumeric);
%addParameter(p,'EMGThresh',.9,@isnumeric);
%addParameter(p,'show','off',@isstr)
%addParameter(p,'noise',[],@ismatrix)
%addParameter(p,'saveMat',false,@islogical);
addParameter(p,'minInterval',30,@isscalar);
%addParameter(p,'min_cyc',2,@isscalar);
%addParameter(p,'cycleTest',false,@islogical);
addParameter(p,'verify',false,@islogical);
addParameter(p,'clipping',false,@islogical);
addParameter(p,'IEDs',[],@isnumeric);
addParameter(p,'outputstruct', true, @islogical);


% assign parameters (either defaults or given)
parse(p,varargin{:})
lowThresholdFactor = p.Results.thresholds(1);
highThresholdFactor = p.Results.thresholds(2);
minRippleDuration = p.Results.durations(1);
maxRippleDuration = p.Results.durations(2);
minInterRippleInterval = p.Results.minInterval;
frequency = p.Results.frequency;
%min_cyc = p.Results.min_cyc;
%cycleTest = p.Results.cycleTest;
verify = p.Results.verify;
passband = p.Results.passband;
sd = p.Results.stdev;
clipping = p.Results.clipping;
IEDs = p.Results.IEDs;
restrict = p.Results.restrict;
outputstruct = p.Results.outputstruct;

global rejected
global ripi
global figh

% Make sure lfp is a column vector
if size(lfp,2) ~= 1
    lfp = lfp(:);
end

%% filter and calculate noise

set(0, 'DefaultFigureWindowStyle', 'normal')
% [bf, af] =  butter(3,[passband(1)/(frequency/2) passband(2)/(frequency/2)]); % 130 - 200 Hz 
% signal = filtfilt(bf, af,lfp);
signal = bz_Filter(lfp, 'filter', 'butter', 'passband', passband, 'order', 3, 'nyquist',frequency/2);
timestamps = (0:1/frequency:length(lfp)/frequency-1/frequency)';


% clipping the extremes (Norman et al.)
if clipping
    extreme = signal > mean(signal) + 4*std(signal);
    signal(extreme) = mean(signal) +4*std(signal);
end


% Parameters
windowLength = frequency/frequency*11;

% Square the signal
squaredSignal = signal.^2;

window = ones(windowLength,1)/windowLength;

% replace the signal around IEDs (+- 500 ms) to NaN
if ~isempty(IEDs)
    %avg = mean(squaredSignal);
    for i = 1:length(IEDs)
        squaredSignal((IEDs(i)-0.5)*frequency:(IEDs(i)+0.5)*frequency) =nan;
    end
end


% use only the specified intervals (e.g., sleep)
keep = [];
if ~isempty(restrict)
    for i=1:size(restrict,1)
        keep = InIntervals(timestamps,restrict);
    end
end
keep = logical(keep); 

% normalize the signal
[normalizedSquaredSignal,sd] = unity(Filter0(window,sum(squaredSignal,2)),sd,keep);
%[normalizedSquaredSignal2,~] = unity(Filter0(window,sum(signal2.^2,2)),[],keep);
%normalizedSquaredSignal=smooth(normalizedSquaredSignal, Fs/4, 'sgolay'); % smoothing 

normalizedSquaredSignal(~keep) = 0;



% Detect ripple periods by thresholding normalized squared signal
thresholded = normalizedSquaredSignal > lowThresholdFactor;
start = find(diff(thresholded)>0);
stop = find(diff(thresholded)<0);
% Exclude last ripple if it is incomplete
if length(stop) == length(start)-1
    start = start(1:end-1);
end
% Exclude first ripple if it is incomplete
if length(stop)-1 == length(start)
    stop = stop(2:end);
end
% Correct special case when both first and last ripples are incomplete
if start(1) > stop(1)
    stop(1) = [];
    start(end) = [];
end
firstPass = [start,stop];
if isempty(firstPass)
    disp('Detection by thresholding failed');
    return
else
    disp(['After detection by thresholding: ' num2str(length(firstPass)) ' events.']);
end


ss = sort(normalizedSquaredSignal); 
ds = diff(ss); 
[~, idx] = max(ds); 
threshold = ss(idx+1)-1; 

wayabove = normalizedSquaredSignal > threshold;
a = find(diff(wayabove)>0);
b = find(diff(wayabove)<0);

%Exclude last ripple if it is incomplete
if length(b) == length(a)-1
    a = a(1:end-1);
end
% Exclude first ripple if it is incomplete
if length(b)-1 == length(a)
    b = b(2:end);
end
% Correct special case when both first and last ripples are incomplete
if a(1) > b(1)
    b(1) = [];
    a(end) = [];
end

[status, ~] = InIntervals(firstPass(:,1),[a-0.5*frequency b+0.5*frequency]);
firstPass = firstPass(~status,:);


% Merge ripples if inter-ripple period is too short
minInterRippleSamples = minInterRippleInterval/1000*frequency;
secondPass = [];

ripple = firstPass(1,:);


for i = 2:size(firstPass,1)
    if firstPass(i,1) - ripple(2) < minInterRippleSamples
        % Merge
        ripple = [ripple(1) firstPass(i,2)];
    else
        secondPass = [secondPass ; ripple];
        ripple = firstPass(i,:);
    end
end
secondPass = [secondPass ; ripple];
if isempty(secondPass)
    disp('Ripple merge failed');
    return
else
    disp(['After ripple merge: ' num2str(length(secondPass)) ' events.']);
end

% Discard ripples with a peak power < highThresholdFactor
thirdPass = [];
peakNormalizedPower = [];
for i = 1:size(secondPass,1)
    [maxValue,maxIndex] = max(normalizedSquaredSignal([secondPass(i,1):secondPass(i,2)]));
    if maxValue > highThresholdFactor % && maxValue < 150
        thirdPass = [thirdPass ; secondPass(i,:)];
        peakNormalizedPower = [peakNormalizedPower ; maxValue];
    end
end

if isempty(thirdPass)
    disp('Peak thresholding failed.');
    return
else
    disp(['After peak thresholding: ' num2str(length(thirdPass)) ' events.']);
end

% Detect negative peak position for each ripple
peakPosition = zeros(size(thirdPass,1),1);
for i=1:size(thirdPass,1)
    [minValue,minIndex] = min(signal(thirdPass(i,1):thirdPass(i,2)));
    peakPosition(i) = minIndex + thirdPass(i,1) - 1;
end

% prepare figure for statistics 
%figure(1),clf
%set(figure(1), 'DefaultFigureWindowStyle', 'normal')


% Discard ripples that are way too long or way too short
ripples = [(thirdPass(:,1)) (peakPosition) ...
    (thirdPass(:,2)) peakNormalizedPower];

duration = (ripples(:,3)-ripples(:,1))./frequency;
[~, soidx] = sort(duration);
pltSize = 0.25*frequency; 


ripples(duration>maxRippleDuration/1000 | duration <minRippleDuration/1000 ,:) = [];

disp(['After duration test: ' num2str(size(ripples,1)) ' events.']);

%% Artifact removal 
% If a noise channel was provided, find ripple-like events and exclude them
% bad = [];
% if ~isempty(noise)
%     if length(noise) == 1 % you gave a channel number
%        noiselfp = getLFP(p.Results.noise,'basepath',p.Results.basepath,'basename',basename);%currently cannot take path inputs
%        squaredNoise = bz_Filter(double(noiselfp.data),'filter','butter','passband',passband,'order', 3).^2;
%     else
%             
% 	% Filter, square, and pseudo-normalize (divide by signal stdev) noise
% 	squaredNoise = bz_Filter(double(noise),'filter','butter','passband',passband,'order', 3).^2;
%     end
%     
% 	window = ones(windowLength,1)/windowLength;
% 	normalizedSquaredNoise = unity(Filter0(window,sum(squaredNoise,2)),sd,[]);
% 	excluded = logical(zeros(size(ripples,1),1));
% 	% Exclude ripples when concomittent noise crosses high detection threshold
% 	previous = 1;
% 	for i = 1:size(ripples,1)
% 		j = FindInInterval([timestamps],[ripples(i,1),ripples(i,3)],previous);
% 		previous = j(2);
% 		if any(normalizedSquaredNoise(j(1):j(2))>highThresholdFactor)
% 			excluded(i) = 1;
%         end
% 	end
% 	bad = ripples(excluded,:);
% 	ripples = ripples(~excluded,:);
% 	disp(['After ripple-band noise removal: ' num2str(size(ripples,1)) ' events.']);
% end
%     %% lets try to also remove EMG artifact?
% if EMGThresh
%     basename = basenameFromBasepath(basepath);
%     EMGfilename = fullfile(basepath,[basename '.EMGFromLFP.LFP.mat']);
%     if exist(EMGfilename)
%         load(EMGfilename)   %should use a bz_load script here
%     else
%         [EMGFromLFP] = bz_EMGFromLFP(basepath,'samplingFrequency',10,'savemat',false,'noPrompts',true);
%     end
%     excluded = logical(zeros(size(ripples,1),1));
%     for i = 1:size(ripples,1)
%        [a ts] = min(abs(ripples(i,1)-EMGFromLFP.timestamps)); % get closest sample
%        if EMGFromLFP.data(ts) > EMGThresh
%            excluded(i) = 1;           
%        end
%     end
%     bad = sortrows([bad; ripples(excluded,:)]);
%     ripples = ripples(~excluded,:);
%     disp(['After EMG noise removal: ' num2str(size(ripples,1)) ' events.']);
% end

%% Verification

if verify

    rejected = zeros(size(ripples,1),1);
    time = -pltSize/frequency:(1/frequency):pltSize/frequency - 1/frequency;

    for ripi = 1:size(ripples,1)

        figh = figure('Visible','on','windowstyle','normal');

        % set its position ([ left bottom width height ])
        set(figh,'Position',[  100   100   1500   700  ])

        % need figure size for later
        figsize = get(figh,'Position');
        figsize = figsize(3:4);

        axh1 = axes('Units','pixels','Position',[50,360,900,300],'tag','axis2draw');
        axh2 = axes('Units','pixels','Position',[50,30,900,300],'tag','axis2draw');

        htext1 = uicontrol('Style','text','String',['event #' num2str(ripi)],...
            'Position',[.25*figsize(1) .95*figsize(2) 200 35], 'FontSize', 14, 'FontWeight', 'bold');

        % draw button
        hredraw_a = uicontrol('Style','pushbutton',...
            'String','ACCEPT!',...
            'Position',[.7*figsize(1) .7*figsize(2) 300 60],...
            'Callback',@accept_event, 'BackgroundColor', 'r', 'FontWeight', 'bold');

        hredraw_r = uicontrol('Style','pushbutton',...
            'String','REJECT!',...
            'Position',[.7*figsize(1) .2*figsize(2) 300 60],...
            'Callback',@reject_event, 'BackgroundColor', [.5 .5 .5], 'FontWeight', 'bold');

        plot(axh1,time, double(lfp(ripples(ripi,2)-pltSize:ripples(ripi,2)+pltSize-1)),'k','LineWidth',2)
        hold(axh1, 'on')
        plot(axh1,(ripples(ripi,1)-ripples(ripi,2):ripples(ripi,3)-ripples(ripi,2))/frequency, lfp(ripples(ripi,1):ripples(ripi,3)), 'g', 'LineWidth',2)

        plot(axh2,time, double(normalizedSquaredSignal(ripples(ripi,2)-pltSize:ripples(ripi,2)+pltSize-1)),'k','LineWidth',2)
        hold on
        yline (lowThresholdFactor, 'r--', 'LineWidth', 2)
        yline (highThresholdFactor, 'r--', 'LineWidth', 2)
        %uicontrol(hredraw_a);
        uiwait(figh);
        close(figh);

    end

    flagged = find (rejected == 1);
    %ripples = ripples(~rejected,:);
else
    flagged = [];
end

%% BUZCODE Struct Output

if outputstruct
    rips = ripples; clear ripples
    
    ripples.timestamps = rips(:,[1 3]);
    ripples.peaks = rips(:,2);            %peaktimes? could also do these as timestamps and then ripples.ints for start/stops?
    ripples.peakNormedPower = rips(:,4);  %amplitudes?
    ripples.stdev = sd;
    ripples.flagged = flagged;
    
    % if ~isempty(bad)
    %     ripples.noise.times = bad(:,[1 3]);
    %     ripples.noise.peaks = bad(:,[2]);
    %     ripples.noise.peakNormedPower = bad(:,[4]);
    % else
    ripples.noise.times = [];
    ripples.noise.peaks = [];
    ripples.noise.peakNormedPower = [];
    % end
    
    %The detectorinto substructure
    detectorinfo.detectorname = 'FindRipplesECoG';
    detectorinfo.detectiondate = now;
    detectorinfo.detectionintervals = restrict;
    detectorinfo.detectionparms = p.Results;
    %detectorinfo.detectionparms = rmfield(detectorinfo.detectionparms,'noise');
    if isfield(detectorinfo.detectionparms,'timestamps')
        detectorinfo.detectionparms = rmfield(detectorinfo.detectionparms,'timestamps');
    end
    %Put it into the ripples structure
    ripples.detectorinfo = detectorinfo;
    
    
end


function y = Filter0(b,x)

if size(x,1) == 1
    x = x(:);
end

if mod(length(b),2)~=1
    error('filter order should be odd');
end

shift = (length(b)-1)/2;

[y0 z] = filter(b,1,x);

y = [y0(shift+1:end,:) ; z(1:shift,:)];


function [U,stdA] = unity(A,sd,restrict)

if ~isempty(restrict)
    meanA = nanmean(A(restrict));
    stdA = nanstd(A(restrict));
else
    meanA = nanmean(A);
    stdA = nanstd(A);
end
if ~isempty(sd)
    stdA = sd;
end

U = (A - meanA)/stdA;


function accept_event(source,eventdata) 
    global figh
    disp('event accepted!');
    uiresume(figh)
 
 return

 function reject_event(source,eventdata) 
     
     global rejected
     global ripi
     global figh

     disp('event rejected!');

     rejected(ripi) = 1;
     uiresume(figh)

 return
