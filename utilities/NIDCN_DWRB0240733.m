function im_h_y = NIDCN_DWRB0240733(im_l_y,model)
% 
% addpath ../
% load('convfea.mat');
 
 weight = model.weight;
 bias = model.bias;
 
 bnmean= model.bnmean;
 bnvariance=model.bnvariance;
 bnscale=model.bnscale ;
 scaleG=model.scaleG ;
 scaleB=model.scaleB ;
 eps=1e-5;

layer_num = size(weight,2);
 

im_y = single(im_l_y);

disp(size(im_y)); 


convfea = vl_nnconv(im_y,weight{1},bias{1},'Pad',3);
resconv1=convfea; 

bnme=repmat(reshape(bnmean{1},1,1,size(bnmean{1},1)),size(convfea,1),size(convfea,2));
scaG=repmat(reshape(scaleG{1},1,1,size(scaleG{1},1)),size(convfea,1),size(convfea,2));
std=repmat(reshape(sqrt(bnvariance{1}/bnscale{1} + eps),1,1,size(convfea,3)),size(convfea,1),size(convfea,2));
scaB=repmat(reshape(scaleB{1},1,1,size(scaleB{1},1)),size(convfea,1),size(convfea,2));
x=convfea-bnme / bnscale{1};
for i = size(bnmean{1},1):-1:1
    y(:,:,i) = scaG(:,:,i).* x(:,:,i) ./ std(:,:,i) +scaB(:,:,i);
end
 

convfea = vl_nnrelu(y);

for m=2:layer_num-1
    disp(['m=:' num2str(m)]);
    
    disp(['beforeconv:' num2str(size(convfea))]);
    
    pad=0;
    if (m==1 || m==2 || m==3 || m==4 || m==6 || m==9)
        pad=3;
    end
    
    if (m==12 || m==15 || m==18)
        pad=4;
    end
    
    if (m==21 || m==24 || m==27 || m==30)
        pad=5;
    end
    
    convfea = vl_nnconv(convfea,weight{m},bias{m},'Pad',pad);
    disp(['afterconv:' num2str(size(convfea))]);
    if m==4
        convfea=convfea+resconv1;
        sum4=convfea; 
    end
    
    if m==7
          convfea=convfea+sum4;
          sum7=convfea;
    end
   
    
     if m==10
        convfea=convfea+sum7;
     end
    
     
     if m==13
       resconv13=convfea;
     end
     
     
     if m==16
       convfea=convfea+resconv13;
       sum16=convfea;
     end
     
      if m==19
       convfea=convfea+sum16;
      end
     
     if m==22
       resconv22=convfea;
     end
     
      if m==25
        convfea=convfea+resconv22;
        sum25=convfea;
      end
     
      if m==28
        convfea=convfea+sum25;
      end
     
    bnme=repmat(reshape(bnmean{m},1,1,size(bnmean{m},1)),size(convfea,1),size(convfea,2));
    scaG=repmat(reshape(scaleG{m},1,1,size(scaleG{m},1)),size(convfea,1),size(convfea,2));
    std=repmat(reshape(sqrt(bnvariance{m}/bnscale{m} + eps),1,1,size(convfea,3)),size(convfea,1),size(convfea,2));
    scaB=repmat(reshape(scaleB{m},1,1,size(scaleB{m},1)),size(convfea,1),size(convfea,2));
    x=convfea-bnme / bnscale{m};
   
     
    for i = size(bnmean{m},1):-1:1
        y(:,:,i) = scaG(:,:,i).* x(:,:,i) ./ std(:,:,i) +scaB(:,:,i);
    end

    convfea = vl_nnrelu(y); 
    
   
    clear y;
   
    
end
 
 
convfea = vl_nnconv(convfea,weight{layer_num},bias{layer_num},'Pad',5);

y = scaleG{layer_num}*(convfea-bnmean{layer_num}/bnscale{layer_num}) ...
    / sqrt(bnvariance{layer_num}/bnscale{layer_num} + eps)  + scaleB{layer_num};

 
im_h_y = y + im_y;
disp(size(im_h_y)); 
    
    
end
