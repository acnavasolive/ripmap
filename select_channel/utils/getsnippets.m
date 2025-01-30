function [snippets] = getsnippets (lfp, win, eventidx,fs, mkplt)
% eventidx: indices of the event, not in sec

% Times
tt = -win(1) : 1/fs : win(2);
% Initialize snippets
snippets = zeros(length(eventidx),length(tt));

% Go throught all events
for ievent = 1:length(eventidx)
    
    % Add zeros if event very close to the edge
    if eventidx(ievent) - (win(1)*fs) < 0
        pad = zeros(1, -(eventidx(ievent)-win(1)*fs)+1)';
        snippets(ievent,:) = [pad; lfp(1:eventidx(ievent)+win(2)*fs)];

    % Otherwise, just take the event time +/- win
    else    
        ids = round(eventidx(ievent)-win(1)*fs : eventidx(ievent)+win(2)*fs);
        snippets(ievent,:) = lfp(ids);
    end
end

if mkplt  
    plot(tt, nanmean(snippets,1))
    xlabel('Time(s)')
    ylabel('uV')
end

