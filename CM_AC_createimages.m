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
    sv = zeros(size(data.pings(ch).sv,1),size(data.pings(ch).sv,2),length(F));
    for ch = 1:length(F)
        sv(:,:,ch)=data.pings(ch).sv;
    end
    
    %% Extract the main binary layer
    [X,Y] = meshgrid(1:size(sv,2),data.pings(ch).range);
    I = zeros(size(data.pings(ch).sv,1),size(data.pings(ch).sv,2));
    
    if ~isempty(school)
        % Loop over schools
        for i=1:length(school)
            % Plot only non empty schools (since we do not know whether an
            % empty school is assiciated to a frequency)
            if ~isempty(school(i).channel)
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
                    I(in) = id(ind);
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



