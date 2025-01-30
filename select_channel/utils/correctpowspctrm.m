function corrected = correctpowspctrm(powspctrm, freqs)
% powspctrm: output of pwelch, freqs x n (trials, channels)

settings = struct();  % Use defaults

f_range = [freqs(1) freqs(end)];

corrected = nan(size(powspctrm));

for i = 1:size(powspctrm,2)
    try
        results = fooof(freqs', powspctrm(:,i)', f_range, settings, true);
        corrected(:,i) = (results.power_spectrum-results.ap_fit)';
    end
end