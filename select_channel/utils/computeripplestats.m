function [stats, corrected] = computeripplestats(powspctrm, freqs,  threshold, foi_range)
% powspctrm: freq x n_events
% threshold: peak detection threshold (default: 0.2)

if nargin < 4
    foi_range = [60, 180]; % Default number of events per screen
end

% Correct power spectrum using foof
corrected = correctpowspctrm(powspctrm, freqs);

foi = freqs(freqs >foi_range(1) & freqs <= foi_range(2));
stats = struct;

% Get peaks from power spectrum
for i = 1:size(corrected,2)
    
    [pks,locs, widths, prominences] = findpeaks(corrected(:,i),'Annotate','extents');
    peaks = freqs(locs(pks>threshold));
    high_pks = peaks(ismember(peaks, foi));

    if isempty(high_pks)
        pk_freq = peaks(~ismember(peaks, foi));
    else
        pk_freq = high_pks;
    end


    if length(pk_freq) > 1 % if more than one peak
        k = dsearchn(freqs(locs)', pk_freq'); % choose the one with higher amplitude
        [~, idx] = max(pks(k));
        pk_freq = pk_freq(idx);
    end
    
    if isempty(pk_freq)
        pk_freq = 0;
    end
   
    %stats(i).max = freqs(locs(idx)); % peak frequency
    stats(i).max = pk_freq; % peak frequency
    stats(i).amp = pks;            % peak amplitudes   
    stats(i).freqs = freqs(locs);           % all peak frequencies
    stats(i).widths = widths;               % peak widths
    stats(i).prominences = prominences;
end

