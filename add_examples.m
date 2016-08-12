load('labels.mat');
load('patch_data.mat');

x=find(labels==1);
for i=1:length(x)
    im=patches(:, i);
    im2=im+.01*rand(size(im));
    patches=[im2 patches];
    labels=[1; labels];
end 

y=find(labels==0);
for j=1:length(y);
    img=patches(:, i);
    img2=img+.01*rand(size(img));
    patches=[patches im2];
    labels=[labels; 0];
end 

if size(patches, 2)==695*2 
    save('datapatches.mat', 'patches');    
end 

if length(labels)==695*2
    save('lab.mat', 'labels');
end 