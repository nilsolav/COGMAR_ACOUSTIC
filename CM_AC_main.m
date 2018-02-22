%% MLproject_main2.m
% This script reads the metadata, copies the data into a scratch disk, and
% creates images and subset data for each pair of snap/raw files. After the script is run
% for a given cruise series, the raw and snap files are located in
%
% Figure files and data files output:
%\\ces.imr.no\deep\data\echosounder\akustikk_all\data\DataOverview_North Sea NOR Sandeel cruise in Apr_May\


% Plotting frequency
par.f='38';

% Use the 38kHz as the main freq for the range resolution
par.rangef = 38;

% Generate test sets in in pixels
par.dx = 400;%px
par.dy = 400;%px
par.overlapx = 200;%px
par.overlapy = 200;%px


% Dependencies:
% https://github.com/nilsolav/LSSSreader/src
% https://github.com/nilsolav/NMDAPIreader
% https://github.com/nilsolav/readEKraw

%% Init

% Define data directories
if isunix
    cd /nethome/nilsolav/repos/github/COGMAR_ACOUSTIC
    % Add libraries
    addpath('/nethome/nilsolav/repos/github/LSSSreader/src/')
    addpath('/nethome/nilsolav/repos/hg/matlabtoolbox/echolab/readEKRaw')
    dd =  '/data/deep/data/echosounder/akustikk_all/';
else
    cd  D:\repos\Github\COGMAR_ACOUSTIC
    dd = '\\ces.imr.no\deep\data\echosounder\akustikk_all\';
end

% This gets a list of all the cruise series
DataOverview = dir(fullfile(dd,'dataoverviews','DataOverview*.mat'));

%% Start loop over cruise series
%par.F='200';
k=11; % SandEel
warning off

%%
for k=11:length(DataOverview)
    dd_data = fullfile(dd,'data',DataOverview(k).name(1:end-4));
    % Load the paired files
    dat = load(fullfile(dd,'dataoverviews',['DataPairedFiles',DataOverview(k).name(13:end)]));
    dat2 = load(fullfile(dd,'dataoverviews',['DataOverview',DataOverview(k).name(13:end)]));
    
    % Loop over years witin cruise series
    for i=1:length(dat.pairedfiles)
        % Create survey - year directory
        dd_data_year = fullfile(dd,'data',DataOverview(k).name(1:end-4),dat2.DataStatus{i+1,2});
        
        % Get the file list
        raw0 = dir(fullfile(dd_data_year,'*.raw'));
        
        % Generate status file if it is missing
        statusfile = fullfile(dd_data_year,'datastatus.mat');
        if ~exist(statusfile)
            status = zeros(length(raw0),1);
            save(statusfile,'status')
        end
        % I need column one and three (snap and raw)
        for f=1:length(raw0)
            load(statusfile)
            if status(f)<=0 % Don't rerun files that are ok (positive numbers)
                % Create file names (in and out)
                [~,fn,~]=fileparts(raw0(f).name);
                png = fullfile(dd_data_year,[fn,'.png']);
                raw = fullfile(dd_data_year,[fn,'.raw']);
                snap = fullfile(dd_data_year,[fn,'.snap']);
                cleandatfile = fullfile(dd_data_year,[fn,'.mat']);
                % Generate figure and save clean data file
                try
                    CM_AC_createimages(snap,raw,png,cleandatfile,par);
                    close gcf
                    disp([datestr(now),'; success ; ',fn])
                    status(f)=now;
                catch
                    disp([datestr(now),'; failed  ; ',fn])
                    status(f)=-now;
                end
            end
            save(statusfile,'status')
        end
    end
end
