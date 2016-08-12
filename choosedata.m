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
    else 
        continue
    end 
end 

if num_patches~=size(patches, 2)
    disp('Patches not loaded correctly. Stop now');
end   
labels=ones(size(patches, 2), 1);
m=length(labels);
pause;

cd('/media/mpcr/5EBA074CBA071FDF/M4_3');
while j<400
    files=dir('*.JPG'); % list all .jpg images in your path 
    for j=1:length(files); % loop through all images in folder
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
        else
            continue 
        end
        patches=[patches im_patch]; %compile image vectors
    end
end 

cd('/home/mpcr/Desktop/lila_birds');
save('patch_data.mat', 'patches');
save('labels.mat', 'labels');