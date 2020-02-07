function [im] = basic_preprocessing_3channel(new_small_dim,neural_dim,image)
%basic_preprossing 1.resize for faster calculations 2.find fundus 3.crop-> KxK and 4.resize -> NxN.
%   First we resize image by a ratio. We use this
%   ratio that makes the small dimension(height)->[new_min_dim]. We make this step
%   to decrease number of pixels for faster calculations.
%   Than convert to grascale image and detect the fundus.
%   If fundus detected, we crop image and we resize to [neural_dim x neural_dim].

% Image dimension
[rows,columns] = size(image(:,:,1));
min_dim = min(rows,columns);
% Find the ratio that makes the small dimension(height) -> [new_min_dim]
ratio = (new_small_dim/min_dim);

% Resize image
im = imresize(image,ratio);
% Find circle with Hough transformation.
% Apply Hough transformation in only one channel. The first channel used here. It can be used, as well, the R or G or B
% channel or gray scale image.

%one_channel_image = im(:,:,1); % Use gray scale image to apply Hough
one_channel_image = rgb2gray(im);
% The image dimensions after the first resizing
[rows,columns] = size(im(:,:,1));

new_max_dim = max(rows,columns);
new_min_dim = min(rows,columns);

% Use the suitable radius range so it finds only the fundus, not small
% circular areas. The radius range defined with experiments.
%radiusRange = [floor((3*new_min_dim)/8) floor((new_max_dim)/2)];
radiusRange = [floor(3.5*(new_min_dim)/8) floor((new_max_dim)/2)];
y_val = 10;

% Apply Hough
[centers,radii,~] = imfindcircles(one_channel_image,radiusRange,'sensitivity',1);

if isempty(radii)
    % Not finding a circle
    im = [];
    disp('Not finding a circle')
else
    % The Hought transformation can find many circles but we keep the most
    % intense that is stored in the first row of centers and radii
    % matrices.
    centers = floor(centers(1,:));
    radii = floor(radii(1));
    
    % Crop image in the way that we keep only the smallest value between
    % 2*(from center to end of the image) or 2*R from height and length.
    % centers(1) -> y, centers(2) ->x
    
    % The first if statement is to check if this image belongs to fundus category 1 or 2.
    % The fundus category 1 is when there is a circular fundus in the image
    % The fundus category 2 is when the fundus is cropped in the upper and
    % bottom
    if (centers(2)-radii > 0 && centers(2)+radii < rows && centers(1)-radii > 0 && centers(1)-radii < columns )
        
        % Fundus category 1
        % Crop a rectangle with center, the center of image.
        im = im(centers(2)-radii:(centers(2)+radii),centers(1)-radii:centers(1)+radii,:);
        im = imresize(im,[neural_dim neural_dim]);
    elseif ((centers(2)-radii <= 0 || centers(2)+radii >= rows) )
        
        % This is for fundus category 2.
        % Add black background upper and below the image. Add more than
        % the expected background and cut it later.
        x_val = max(floor(radii-centers(2)),floor(radii - rows + centers(2)))+6;
        if x_val>=0
            % Add background to x that it is needed to complete fundus but also to the y for the cases that the radius goes out of the image in y direction also. 
            im = padarray(im,[x_val y_val],0,'both');
            % New center(2) -> (centers(2) + x_val). So for the columns we go from
            % (centers(2) + x_val -radii) to (centers(2) + x_val +radii)
            
            % We have to add one pixel in the end, if we loose one from the start(a<1 -> 1)
            a = max(centers(2) + x_val -radii);
            if (a < 1)
                im = im(a:((centers(2)+ x_val+radii))+1,:,:);
            else
                im = im(a:((centers(2)+ x_val+radii)),:,:);
            end
            % Cut the rows
            im = im(:,(centers(1)+ y_val-radii):centers(1)+y_val+radii,:);
            % Final resizing of the image
            im = imresize(im,[neural_dim neural_dim]);
        else
            
            im = [];
            return
        end
    else
        im = [];
        return
        
    end
    
end

end


%