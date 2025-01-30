function peaksAlign = alignpeaks(lfp, srLFP, side, peaks)

% window parameters
win = 0.075;
win_min = 0.005; % sec
sm_win = 0.010; % sec

iwin = round(win*srLFP);


for ievent = 1: length(peaks)
    % cut the ripple snippets around specified window
    lfp_win = lfp(round(peaks(ievent)*srLFP + [-iwin:iwin]));
    % - smooth
    lfp_win = side*movmean(movmean(lfp_win, sm_win*srLFP), sm_win*srLFP);
    % - get the smoothed minimum
    [~, imin] = min(lfp_win);
    new_peak = peaks(ievent)-win + imin/srLFP;
    % - gt the raw minimum
    lfp_win = side*lfp(round(new_peak*srLFP + [-win_min*srLFP : win_min*srLFP]));
    [~, imin] = min(lfp_win);

    % Save time and events aligned to channel ch
    peaksAlign(ievent) = new_peak-win_min + imin/srLFP;

end

