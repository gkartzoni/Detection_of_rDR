%Test script
%basic preprocessing
warning('off','all')
warning

hold on;
close all;

extension = '*.jpeg';
imagefiles = dir(extension);
nfiles = length(imagefiles);    % Number of files foun


warning('off','all')
warning
k = 0;
for ii=1:nfiles
    currentfilename = imagefiles(ii).name;
    %currentfilename = '2_left.jpeg';
    dir1 = dir(currentfilename);
    if dir1.bytes == 0   
        k = k + 1
        delete currentfilename
    end
end
ex_time = toc

