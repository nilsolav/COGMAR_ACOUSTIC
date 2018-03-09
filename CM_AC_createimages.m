function CM_AC_createimages(snap,raw,png,datfile,par)

% Plot the raw file and generate a clean datfile for python

f=par.f;

% Read snap file
[school,layer,exclude,erased] = LSSSreader_readsnapfiles(snap);

% Read raw file and convert to data
[raw_header,raw_data] = readEKRaw(raw);
raw_cal = readEKRaw_GetCalParms(raw_header, raw_data);
data = readEKRaw_Power2Sv(raw_data,raw_cal,'Linear',true);

% Get the main frequency
for ch = 1:length(raw_data.pings)
    F(ch)=raw_data.pings(ch).frequency(1)/1000;
end

% Somtetimes there are missing frequencies, if so, give error
br = false;
if isempty(find(F==(str2num(f))))
    disp('Missing main plotting frequency')
    br=true;
end
if isempty(find(F==par.rangef))
    disp('Missing main range frequency')
    br=true;
end
if br
    error('Missing frequencies')
end

%% Plot result
if ~isempty(png)
    ch = find(F==(str2num(f)));
    td = double(median(raw_data.pings(ch).transducerdepth));
    [fh, ih] = readEKRaw_SimpleEchogram(10*log10(data.pings(ch).sv), 1:length(data.pings(ch).time), data.pings(ch).range);
    % Plot the interpretation mask
    hold on
    LSSSreader_plotsnapfiles(layer,school,erased,exclude,f,ch,td)
    title([f,'kHz'])
    print(png,'-dpng')
    close(gcf)
end

%% Extract clean data file
if ~isempty(datfile)
    %% Reshape data
    Fi=find(F==par.rangef);
    % Check if the range and time vectors are different
    tol = min(diff(data.pings(Fi).time));
    t_all=[];
    for ch = 1:length(F)
        range(ch)=length(data.pings(ch).range);
        timerange(ch)=length(data.pings(ch).time);
        % Create a unique timevector
        tround{ch} = round(data.pings(ch).time/tol);
        t_all = [t_all tround{ch}];
    end
    
    % Fix different time vectors (add NaN's)
    t_final = unique(t_all);
    % Initialize the sv structure
    sv = zeros(size(data.pings(Fi).sv,1),length(t_final),length(F));
    % Fill in the missing pings as NaN's (in time)
    for ch = 1:length(F)
        % Keep the range but change the time vector
        sv_dum{ch}=NaN(size(data.pings(ch).sv,1),length(t_final));
        % Find the pings and add to the structure
        [~,LOCB] = ismember(tround{ch},t_final);
        % Add data to new structure
        sv_dum{ch}(:,LOCB)= data.pings(ch).sv;
    end
        
    % Fix different range vectors
    if length(unique(range))==1 % Same range vector length
        for ch = 1:length(F)
            sv(:,:,ch)=sv_dum{ch};
        end
    else
        % Resample/average if the ranges are different between freqs
        for ch = 1:length(F)
            if ch==Fi
                sv(:,:,ch)=sv_dum{ch};
            elseif range(ch)>range(Fi)
               % Average    discretize(x,edges)
               dfe = median(diff(data.pings(Fi).range));
               edges = [data.pings(Fi).range-.5*dfe; data.pings(Fi).range(end)+.5*dfe]; 
               bins = discretize(data.pings(ch).range, edges);
               % If the secondary frequency has data that is outside the
               % edges it needs to be removed
               nonanid=~isnan(bins);
               for p=1:size(sv_dum{ch},2)
                   sv(:,p,ch)=accumarray(bins(nonanid), sv_dum{ch}(nonanid,p), [], @mean);
               end
            else
               % Resample
               sv(:,:,ch)=interp1(data.pings(ch).range, data.pings(ch).sv,data.pings(Fi).range);   
            end
        end
    end
    % Debug plotting
    %     for ch = 1:length(F)
    %         figure(ch)
    %         clf
    %         imagesc(10*(log10(squeeze(sv(:,:,ch)))-log10(squeeze(sv(:,:,Fi)))))
    %     end
    %     imagesc(10*log10(squeeze(sv(:,:,1))))
    
    
    %% Extract the main binary layer
    [X,Y] = meshgrid(1:size(sv,2),data.pings(Fi).range);
    I = zeros(size(X));
    
    if ~isempty(school)
        % Loop over schools
        for i=1:length(school)
            % Plot only non empty schools (since we do not know whether an
            % empty school is assiciated to a frequency)
            if isfield(school(i),'channel')&&~isempty(school(i).channel)
                % Loop over channels
                % Plot only the relevant frequency
                
                % Get the ID string for this patch and freq
                fraction = [];% zeros(length(layer(i).school),length(length(layer(i).school(ch).species)));
                id = [];%zeros(length(layer(i).school),length(length(layer(i).school(ch).species)));
                for ch = 1:length(school(i).channel)
                    if isfield(school(i).channel(ch),'species')
                        for sp=1:length(school(i).channel(ch).species)
                            fraction(ch,sp) =str2num(school(i).channel(ch).species(sp).fraction);
                            id(ch,sp)=str2num(school(i).channel(ch).species(sp).speciesID);
                        end
                    end
                    if length(unique(id(:)))~=1
                        warning('Different IDs in layers for same layer. Using max fraction layer.')
                    end
                    % Set the species ID to the max fraction
                    [~,ind]=max(fraction(:));
                    in=inpolygon(X,Y, school(i).x,school(i).y-td);
                    if ~isempty(ind)%In some case there are no species attributed. 
                        I(in) = id(ind);
                    end
                end
            end
        end
    end
    %% Create training set indices
    ind = overlapind(I,par,sv);
    
    %% Write an NC file that stores both the mask and the data
    t=data.pings.time;
    range = data.pings.range;
    save(datfile,'-v7','I','sv','F','t','range','ind')
end

function ind=overlapind(I,par,sv)
%
% ind(:,1) xindex
% ind(:,2) yindex
% ind(:,3) xstep
% ind(:,4) ystep
% ind(:,5) number of nonzero classes
%
%par.dx = 400;%px
%par.dy = 400;%px
%par.overlapx = 200;%px
%par.overlapy = 200;%px
%%
S=size(I);
N1 = floor((S(1)-par.overlapx)/(par.dx-par.overlapx));
N2 = floor((S(2)-par.overlapy)/(par.dy-par.overlapy));

ind = zeros(N1*N2,5);
%% Get indices
for i=1:N1
    for j=1:N2
        ind((i-1)*N2+j,1:4) = [(par.dx-par.overlapx)*(i-1)+1,(par.dy-par.overlapy)*(j-1)+1,par.dx,par.dy];
    end
end

%% Count the number of non zero classes
for k=1:size(ind,1)
    Idum = I(ind(k,1):(ind(k,1)+ind(k,3)),ind(k,2):(ind(k,2)+ind(k,4)));
    ind(k,5) = sum(Idum(:)~=0);
end

%% Debug
% clf
% for k=1:size(ind,1)
%     if ind(k,5)>0
%         figure(1)
%         clf
%         imagesc(I(ind(k,1):(ind(k,1)+ind(k,3)),ind(k,2):(ind(k,2)+ind(k,4))))
%         figure(2)
%         clf
%         imagesc(10*log10(sv(ind(k,1):(ind(k,1)+ind(k,3)),ind(k,2):(ind(k,2)+ind(k,4)))))
%         pause(1)
%     end
% end



function finaldata=insertNaN(data,diffthreshold)
% FUNCION  insertNaN: Used to insert NaN values into a vector or matrix
% when the difference value of successive points exceed an input threshold.
%  If vector data is provided, NaN's are inserted where the differences
%  exceed the requested threshold. 
%  If matrix data is input, the difference condition is applied to the
%  first column of data, and NaN's are inserted along the entire row.
%
% usage:  output=insertNaN(data,threshold);
%   INPUTS:     data - input data which will be checked for gaps
%               threshold - threshold value to distinguish where NaN's are
%               placed in the data
%   OUTPUTS:    output - input data, with NaN values inserted where
%               differences exceeded the requested threshold
% 
% Example 1:
% output=insertNaN([11:13 15:17 19:21 25:27],1);
%      returns: 
%      output = [11 12 13 NaN 15 16 17 NaN 19 20 21 NaN 25 26 27]
% Example 2:
%  output=insertNaN([[1:2 5:7 9:10].',[1:7].',[11:17].'],1);
%      returns:
%      output =
%         1     1    11
%         2     2    12
%       NaN   NaN   NaN
%         5     3    13
%         6     4    14
%         7     5    15
%       NaN   NaN   NaN
%         9     6    16
%        10     7    17
% 
% Chris Miller
%cwmiller@nps.edu
% 9/14/11

if isvector(data), 
    diffdata=diff(data);
    index=find(diff(data)>diffthreshold);
    if isempty(index),
        finaldata=data;
        return;
    end;
    finaldata=NaN*ones(1,length(data)+length(index));  % preallocate output 
    finaldata(1:index(1))=data(1:index(1));
    if length(index)>1,
        for i=2:length(index),
            finaldata(index(i-1)+i:index(i)+i-1)=data(index(i-1)+1:index(i));
        end;
    else
        i=1;
    end;
    finaldata(index(i)+i+1:length(finaldata))=data(index(i)+1:length(data));
else,
    diffdata=diff(data(:,1));
    index=find(diffdata>diffthreshold);
    if isempty(index),
        finaldata=data;
        return;
    end;
    [n,m]=size(data);
    finaldata=NaN*ones(n+length(index),m);  % preallocate output 
    finaldata(1:index(1),:)=data(1:index(1),:);
    if length(index)>1,
        for i=2:length(index),
            finaldata(index(i-1)+i:index(i)+i-1,:)=data(index(i-1)+1:index(i),:);
        end;
    else
        i=1;
    end;
    finaldata(index(i)+i+1:length(finaldata),:)=data(index(i)+1:length(data),:);
end;