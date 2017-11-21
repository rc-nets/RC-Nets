
clear; clc;
%%this test demo with matcaffe using caffemodel
addpath('/usr/local/caffe/matlab/');

%%setup
addpath('/data/YJYLi/data/utilities/');
model='/data/YJYLi/data/RC-Net.prototxt';
weights='//data/YJYLi/data/model/RC-Net_N10.caffemodel';
savepath='/result';
folderTest='/data/YJYLi/data/BSD200/';

%%noise level
noiseSigma  = 10;

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


%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
Times = zeros(1,length(filePaths));


for i = 1:length(filePaths)

    %%% read images
    label = imread(fullfile(folderTest,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    disp([num2str(i),'    ',filePaths(i).name,'    ',num2str(noiseSigma)]);
   
    
    if(size(label,3)>1)
     imagecolor = 1; 
     label = rgb2ycbcr(label);  
     label = im2double(label);
          
      %% split to three channels
     label_y=label(:, :, 1); 
     label_cb=label(:, :, 2);
     label_cr=label(:, :, 3);  
     
     
     %% convert to double
     label_y = im2double(label_y);
     label_cb = im2double(label_cb);
     label_cr = im2double(label_cr);
    end


     input = single(label_y + noiseSigma/255*randn(size(label_y)));

 
     [height, width, channel] = size(input);
    [PSNR_noisy, SSIM_noisy] = Cal_PSNRSSIM(im2uint8(label_y),im2uint8(input),0,0);
     disp('noisimage PSNR / SSIM');
     disp ([PSNR_noisy, SSIM_noisy]);
    %%test
    tic;
 
        net = caffe.Net(model,weights,'test');
        net.blobs('data').reshape([height width channel 1]); % reshape blob 'data'
        net.blobs('data').set_data(input);
        net.forward_prefilled();
        output = net.blobs('sum5').get_data();

    timeCur=toc;

   %% calculate PSNR and SSIM
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label_y),im2uint8(output),0,0);

    
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
    Times(i) = timeCur;



  %% psnr ssim time output
     %% channel merge for color image
   if imagecolor == 1 
    labelout = im2uint8(cat(3,label_y,label_cb,label_cr));
    inputout = im2uint8(cat(3,input,label_cb,label_cr));
    outputout = im2uint8(cat(3,output,label_cb,label_cr));
    
      if length(filePaths)<=1
       %% Save
        imwrite(ycbcr2rgb(labelout),fullfile(savepath,[nameCur '_original.bmp']));
        imwrite(ycbcr2rgb(inputout),fullfile(savepath,[nameCur '_noiseSigma_x' num2str(noiseSigma) '.bmp']));
        imwrite(ycbcr2rgb(outputout),fullfile(savepath,[nameCur '_recover_S' num2str(noiseSigma) '.bmp']));
    
      end
      
   
   end
end
disp('greyimage:');
disp([mean(PSNRs),mean(SSIMs),mean(Times)]);


