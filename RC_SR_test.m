clear; clc;
%%this test demo with matcaffe using caffemodel
addpath('/caffe/matlab/');

%%setup
addpath('/utilities/');
model='/RC_deploy.prototxt';
weights='/RC-Net-SRx2.caffemodel';
savepath='/result';
folderTest='/testset/B100/';

%%noise level
scale  = 2;

showResult  = 0;
useGPU      = 0;
pauseTime   = 1;
imagecolor = 0;
%% use gpu mode
caffe.reset_all(); 
caffe.set_mode_gpu();
caffe.set_device(0);


%% read images
ext  =  {'*.jpg','*.png','*.bmp'};

filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
Times = zeros(1,length(filePaths));

for i = 1:length(filePaths)

    %%% read images
    label = imread(fullfile(folderTest,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    disp([num2str(i),'    ',filePaths(i).name,'    ',num2str(scale)]);
   
    
    if(size(label,3)>1)
     label = rgb2ycbcr(label);
       
     label = im2double(label);
     
     label = modcrop(label,scale);
     
      %% split to three channels
     label_y=label(:, :, 1); 
     label_cb=label(:, :, 2);
     label_cr=label(:, :, 3);  
     
    end
    
    %% convert to double(to match final channel add-on) 
     label_y = im2double(label_y);
     label_cb = im2double(label_cb);
     label_cr = im2double(label_cr);
    
     input = imresize(label_y, 1/scale, 'bicubic');
     input = imresize(input, scale, 'bicubic');
 
    [height, width, channel] = size(input);

    [PSNR_scale, SSIM_scale] = Cal_PSNRSSIM(im2uint8(label_y),im2uint8(input),0,0);
    disp([PSNR_scale, SSIM_scale]);
    disp('bicubic');
    tic;
 
     	%%test
        net = caffe.Net(model,weights,'test');
        net.blobs('data').reshape([height width channel 1]); % reshape blob 'data'
        net.blobs('data').set_data(input);      
        net.forward_prefilled();
        output = net.blobs('sum5').get_data();

    timeCur=toc;

    output=imresize(output,[height width]);
    %%% calculate PSNR and SSIM
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label_y),im2uint8(output),0,0);

   % disp([num2str(PSNR_scale,'%2.2f'),'    ',num2str(SSIM_scale,'%2.4f')]);
    
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
    Times(i) = timeCur;



  %% psnr ssim time output
     %% channel merge, note channel type

    labelout = im2uint8(cat(3,label_y,label_cb,label_cr));
    inputout = im2uint8(cat(3,input,label_cb,label_cr));
    outputout = im2uint8(cat(3,output,label_cb,label_cr));;
  
      
   if length(filePaths)<=1
    %% Save 
   
        imwrite(ycbcr2rgb(labelout),fullfile(savepath,[nameCur '_original.bmp']));
        imwrite(ycbcr2rgb(inputout),fullfile(savepath,[nameCur '_noiseSigma_x' num2str(scale) '.bmp']));
        imwrite(ycbcr2rgb(outputout),fullfile(savepath,[nameCur '_recover_S' num2str(scale) '.bmp']));
   end
end
disp('grey:');
disp([mean(PSNRs),mean(SSIMs),mean(Times)]);
disp('bicubic');
disp([PSNR_scale, SSIM_scale]);

