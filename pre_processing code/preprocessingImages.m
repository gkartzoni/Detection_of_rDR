%Test script
%basic preprocessing

warning('off','all')
warning

hold on;
close all;

% Height for the first resizing of the image.
resizing_height = 512;
% Dimension of final image -> [neural_dim x neural_dim]
neural_dim = 512;

% Store the number of images that didn't preprocessed.
k = 0;

% Path to csv file with the images' level
%table = readtable('C:\\Users\\Ioanna\\Desktop\\thesis\\csv_files\\testLabels.csv');
table = readtable('C:\Users\Ioanna\Desktop\retinopathy_solution.csv');

% Vector with the name of the images 
Y_image_name = table2array(table(:,1));
% Vector with the level of DR of the images 
Y_level = table2array(table(:,2));
tic
%pathTo = 'C:\Users\Ioanna\Desktop\DR_\';
pathTo = 'G:\DatasetsForDR\Kaggle\data kaggle\test\';
%pathTo = 'C:\Users\Ioanna\Desktop\DR_\';

pathExt = strcat(pathTo,'*.jpeg');
% Find all the files tha have [extension] as file extension.
imagefiles = dir(pathExt);
nfiles = length(imagefiles);    % Number of files found
warning('off','all')
warning

for ii=1:nfiles
    % Read first image
    currentfilename = imagefiles(ii).name;
    dir1 = dir(currentfilename);
    
    % Read image
    currentimage = imread(strcat(pathTo,currentfilename));
    % Find the id of current image in [Y_image_name] matrix
    %imshow(currentimage)
    id1 = find(ismember(strcat(Y_image_name,'.jpeg'), currentfilename));
    % Find the level of DR for current image
    y_label = Y_level(id1);             
    % Preprocess image
    [im] = basic_preprocessing_3channel(resizing_height,neural_dim,currentimage);
    % Save preprocessed image in specific folder, depending from DR
    % level. 5 different folders for every DR level.
    if ~isempty(im)
        if y_label == 0
            imwrite((im),sprintf('C:\\Users\\Ioanna\\Desktop\\DR_test\\class0\\%s',currentfilename));
        elseif y_label == 1
            imwrite((im),sprintf('C:\\Users\\Ioanna\\Desktop\\DR_test\\class1\\%s',currentfilename));
        elseif y_label == 2
            imwrite((im),sprintf('C:\\Users\\Ioanna\\Desktop\\DR_test\\class2\\%s',currentfilename));
        elseif y_label == 3
            imwrite((im),sprintf('C:\\Users\\Ioanna\\Desktop\\DR_test\\class3\\%s',currentfilename));
        elseif y_label == 4
            imwrite((im),sprintf('C:\\Users\\Ioanna\\Desktop\\DR_test\\class4\\%s',currentfilename));
        end
    else
        disp(currentfilename)
    end
end

ex_time = toc

