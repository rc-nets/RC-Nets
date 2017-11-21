function im_h_y = NIDCN_RED30BN(im_l_y,model)

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
[M,N]=size(im_y);
output=cell(1,15);
% disp(layer_num);
convfea=im_y;
for m=1:layer_num
   if m<=15    
         convfea=vl_nnconv(convfea,weight{m},bias{m},'Pad',3);

        bnme=repmat(reshape(bnmean{m},1,1,size(bnmean{m},1)),size(convfea,1),size(convfea,2));
        scaG=repmat(reshape(scaleG{m},1,1,size(scaleG{m},1)),size(convfea,1),size(convfea,2));
        std=repmat(reshape(sqrt(bnvariance{m}/bnscale{m} + eps),1,1,size(convfea,3)),size(convfea,1),size(convfea,2));
        scaB=repmat(reshape(scaleB{m},1,1,size(scaleB{m},1)),size(convfea,1),size(convfea,2));
        x=convfea-bnme / bnscale{m};
        for i = size(bnmean{m},1):-1:1
            y1(:,:,i) = scaG(:,:,i).* x(:,:,i) ./ std(:,:,i) +scaB(:,:,i);
        end

         convfea = vl_nnrelu(y1);
    %      disp(size(convfea));
    %      output{1,i}=mat2cell(convfea);
         output{1,m}=convfea;
   else
        convfea=vl_nnconvt(convfea,weight{m},bias{m},'crop',3);
       
        
         if m<30 
             
            bnme=repmat(reshape(bnmean{m},1,1,size(bnmean{m},1)),size(convfea,1),size(convfea,2));
            scaG=repmat(reshape(scaleG{m},1,1,size(scaleG{m},1)),size(convfea,1),size(convfea,2));
            std=repmat(reshape(sqrt(bnvariance{m}/bnscale{m} + eps),1,1,size(convfea,3)),size(convfea,1),size(convfea,2));
            scaB=repmat(reshape(scaleB{m},1,1,size(scaleB{m},1)),size(convfea,1),size(convfea,2));
            x=convfea-bnme / bnscale{m};
            for i = size(bnmean{m},1):-1:1
                y2(:,:,i) = scaG(:,:,i).* x(:,:,i) ./ std(:,:,i) +scaB(:,:,i);
            end
             convfea = vl_nnrelu(y2);  
             
             if(mod(m,2)==0)
    %            disp(size(output{1,30-i}));
    %            disp(size(convfea));
               convfea=convfea+output{1,30-m};
             end
         else
             
            y2 = scaleG{layer_num}*(convfea-bnmean{layer_num}/bnscale{layer_num}) ...
            / sqrt(bnvariance{layer_num}/bnscale{layer_num} + eps)  + scaleB{layer_num};
            
            convfea=y2;
         end
         
   end
     disp(m);

disp(size(convfea));

end
im_h_y = convfea + im_y;
   
     
end
