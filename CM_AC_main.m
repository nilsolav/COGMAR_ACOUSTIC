%% MLproject_main2.m
% This script reads the metadata, copies the data into a scratch disk, and
% creates images for each pair of snap/raw files. After the script is run
% for a given cruise series, the raw and snap files are located in
%
% Output:
%


% Dependencies:
% https://github.com/nilsolav/LSSSreader/src
% https://github.com/nilsolav/NMDAPIreader
% https://github.com/nilsolav/readEKraw


%% Init

% Define data directories
if isunix
    cd /nethome/nilsolav/repos/nilsolav/MODELS/MLprosjekt/
    % Add libraries
    addpath('/nethome/nilsolav/repos/github/LSSSreader/src/')
    addpath('/nethome/nilsolav/repos/hg/matlabtoolbox/echolab/readEKRaw')
    dd =  '/data/deep/data/echsosounder/akustikk_all/';
else
    cd D:\repos\svn\MODELS\MLprosjekt\
    dd = '\\ces.imr.no\deep\data\echsosounder\akustikk_all\';
end

% This gets a list of all the cruise series
DataOverview = dir(fullfile(dd,'dataoverviews','DataOverview*.mat'));

%% Start loop over cruise series
F='200';
warning off
for k=11%1:length(DataOverview)
    dd_data = fullfile(dd,'data',DataOverview(k).name(1:end-4));
    
    % Load the paired files
    dat = load(fullfile(dd,'dataoverviews',['DataPairedFiles',DataOverview(k).name(13:end)]));
    dat2 = load(fullfile(dd,'dataoverviews',['DataOverview',DataOverview(k).name(13:end)]));
    
    % Loop over years witin cruise series
    for i=1%:length(dat.pairedfiles)
        % Create survey - year directory
        dd_data_year = fullfile(dd,'data',DataOverview(k).name(1:end-4),dat2.DataStatus{i+1,2});
        
        % Get the file list
        raw0 = dir(fullfile(dd_data_year,'*.raw'));
        
        % I need column one and three (snap and raw)
        for f=1%:length(raw0)
            
            % Create file names (in and out)
            [~,fn,~]=fileparts(raw0(i).name);
            png = fullfile(dd_data_year,[fn,'.png']);
            raw = fullfile(dd_data_year,[fn,'.raw']);
            snap = fullfile(dd_data_year,[fn,'.snap']);
            
            % Generate figure
            try
                CM_AC_createimages(snap,raw,png,F);
                close gcf
                disp([datestr(now),'; success ; ',fn])
            catch
                disp([datestr(now),'; failed  ;',fn])
            end
        end
    end
end
