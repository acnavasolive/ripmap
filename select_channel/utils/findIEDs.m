function IEDs = findIEDs(lfp,timestamps, fs, threshold, passband, EMGThresh, intervals)

IEDs = findRipples_a(lfp, timestamps, 'thresholds', threshold, 'passband',passband,...
    'EMGThresh',EMGThresh,'durations',[50 250], 'restrict', intervals, 'saveMat',false,'frequency',fs); 


%% Pick events with the large STD 
validEvents = find(IEDs.peakNormedPower>=5);
IEDs.timestamps = IEDs.timestamps(validEvents,:);
try 
   IEDs.peaks = IEDs.peaks(validEvents,:);
   IEDs.peakNormedPower = IEDs.peakNormedPower(validEvents,:);
end


