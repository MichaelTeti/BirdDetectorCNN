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
labels=[];
files=dir('*.JPG');
for j=1:length(files)-1;
    cd('/media/mpcr/5EBA074CBA071FDF/bird');
    file_name=files(j).name;
    cd('/media/mpcr/5EBA074CBA071FDF/M4_3');
    files2=dir('*.JPG');
    for i=1:length(files2);
        if file_name==files2(i).name
            im=im2double(imread(files2(i).name));
            im2=im2double(imread(files2(i-1).name));
            im3=im2double(imread(files2(i+1).name));
            imagesc(im)
            fig=gcf();
            [x, y]=getpts(fig);
            x=floor(x); y=floor(y);
            cd('/home/mpcr/Desktop/lila_birds');
            if y(end)<1500
                for p=1:numel(y)
                    im_patch=rgb2gray(im(y(p)-32:y(p)+32, x(p)-32:x(p)+32, :));
                    im_patch2=rgb2gray(im2(y(p)-32:y(p)+32, x(p)-32:x(p)+32, :));
                    im_patch3=rgb2gray(im3(y(p)-32:y(p)+32, x(p)-32:x(p)+32, :));
                    diff=im_patch3-2*im_patch+im_patch2;
                    %im_diff=MPCRWhiten_Image2(rgb2gray(abs(im_patch-im_patch2)));
                    %im_diff=imgresize(im_diff, .3, .3);
                    patches=[patches diff(:)];
                    labels=[labels; 1];
                end
            else
                continue
            end
        end
    end
end

pause
m=numel(labels);

for i2=2:100;
    cd('/media/mpcr/5EBA074CBA071FDF/M4_3');
    im=im2double(imread(files2(i2).name));
    im2=im2double(imread(files2(i2-1).name));
    im3=im2double(imread(files2(i2+1).name));
    imagesc(im);
    fig=gcf();
    [x, y]=getpts(fig);
    x=floor(x);  y=floor(y);
    cd('/home/mpcr/Desktop/lila_birds');
    if y(end)<1500
        for p=1:numel(y);
            im_patch=rgb2gray(im(y(p)-32:y(p)+32, x(p)-32:x(p)+32, :));
            im_patch2=rgb2gray(im2(y(p)-32:y(p)+32, x(p)-32:x(p)+32, :));
            im_patch3=rgb2gray(im3(y(p)-32:y(p)+32, x(p)-32:x(p)+32, :));
            diff=im_patch3-2*im_patch+im_patch2;
            %diff=MPCRWhiten_Image2(rgb2gray(abs(im_patch-im_patch2)));
            %diff=imgresize(diff, .3, .3);
            patches=[patches diff(:)];
            labels=[labels; 0];
        end
    else
        continue
    end
end

cd('/home/mpcr/Desktop/lila_birds');
[patches , labels]=add_examples(patches, labels);
patches=patches';
save('data.mat', 'patches');
save('labels.mat', 'labels');
