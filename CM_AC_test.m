%% Test to write an image and dump data
snap = 'D:\DATA\S2005114_PG.O.Sars[4174]\ACOUSTIC_DATA\LSSS\WORK\tokt2005114-D20051118-T062010.snap';
raw  = 'D:\DATA\S2005114_PG.O.Sars[4174]\ACOUSTIC_DATA\EK60\EK60_RAWDATA\tokt2005114-D20051118-T062010.raw';
png = 'test.png';
datfile = 'test.mat';

% Plotting frequency
par.f='38';

% Generate test sets in in pixels
par.dx = 400;%px
par.dy = 400;%px
par.overlapx = 200;%px
par.overlapy = 200;%px

CM_AC_createimages(snap,raw,png,datfile,par)

