function MLproject_createimages(snap,raw,png,f)


% Read snap file
[school,layer,exclude,erased] = LSSSreader_readsnapfiles(snap);

% Read raw file and convert to sv
[raw_header,raw_data] = readEKRaw(raw);
raw_cal = readEKRaw_GetCalParms(raw_header, raw_data);
Sv = readEKRaw_Power2Sv(raw_data,raw_cal);

% Get the main frequency
for ch = 1:length(raw_data.pings)
    F(ch)=raw_data.pings(ch).frequency(1)/1000;
end

ch = find(F==(str2num(f)));

td = double(median(raw_data.pings(ch).transducerdepth));

% Plot result
[fh, ih] = readEKRaw_SimpleEchogram(Sv.pings(ch).Sv, 1:length(Sv.pings(ch).time), Sv.pings(ch).range);

% Plot the interpretation mask
hold on
LSSSreader_plotsnapfiles(layer,school,erased,exclude,f,td)
title([f,'kHz']) 

print(png,'-dpng')
close(gcf)
