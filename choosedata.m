%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------------------------------------------
%
%                         Michael A. Teti
%
%            Machine Perception and Cognitive Robotics Lab
%
%            Center for Complex Systems and Brain Sciences
%
%                    Florida Atlantic University
%
%--------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------------------------------------------------------
%
% This program is an attempt to detect bird presence in highly noisy 
% and heterogenous images taken by a game camera every three minutes from 
% 6am to 2pm for one week. This particular program is used to select the
% training images for a convolutional neural network.
%
%--------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all; 

cd('/media/mpcr/5EBA074CBA071FDF/bird');

patches=[];
num_patches=0;
files=dir('*.JPG'); % list all .jpg images in your path 
for j=1:length(files); % loop through all images in folder
    im=im2double(imread(files(j).name)); % read each image 
    figure(1)
    imagesc(im)
    fig=gcf();
    [x,y]=getpts(fig);
    x=floor(x);
    y=floor(y);
    if y<1500
        for r=1:numel(x)
            im_patch=im(y(r)-32:y(r)+32, x(r)-32:x(r)+32, :);
            im_patch=im_patch(:); 
            patches=[patches im_patch]; %compile image vectors
            num_patches=num_patches+1;
        end
    elseif x<33
        continue 
    elseif x>(size(im, 2)-33)
        continue
    else 
        continue
    end 
end 


labels=ones(size(patches, 2), 1);
m=length(labels);
pause;

cd('/media/mpcr/5EBA074CBA071FDF/M4_3');

files=dir('*.JPG'); % list all .jpg images in your path 
for j=1:m; % loop through all images in folder
    im=im2double(imread(files(j).name)); % read each image 
    figure(1)
    imagesc(im)
    fig=gcf();
    [x,y]=getpts(fig);
    x=floor(x); y=floor(y);
    if y<1500
        for r=1:numel(x)
            im_patch=im(y(r)-32:y(r)+32, x(r)-32:x(r)+32, :);
            im_patch=im_patch(:);
            labels=[labels; 0];
        end
    elseif x<33
        continue 
    elseif x>(size(im, 2)-33)
        continue
    else
        continue 
    end
    patches=[patches im_patch]; %compile image vectors
end
 

[patches , labels]=add_examples(patches, labels);

cd('/home/mpcr/Desktop/lila_birds');
save('patch_data.mat', 'patches');
save('labels.mat', 'labels');
