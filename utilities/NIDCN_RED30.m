function im_h_y = NIDCN_RED30(im_l_y,model)

 
weight = model.weight;
bias = model.bias;
layer_num = size(weight,2);
im_y = single(im_l_y);
disp(size(im_y));
[M,N]=size(im_y);
output=cell(1,15);
disp(layer_num);
convfea=im_y;
for i=1:layer_num
   if i<=15    
     convfea=vl_nnconv(convfea,weight{i},bias{i},'Pad',3);
     convfea = vl_nnrelu(convfea);
     disp(size(convfea));
%      output{1,i}=mat2cell(convfea);
     output{1,i}=convfea;
   else
     convfea=vl_nnconvt(convfea,weight{i},bias{i},'crop',3);
     if i<30
         convfea = vl_nnrelu(convfea);  
         if(mod(i,2)==0)
           disp(size(output{1,30-i}));
           disp(size(convfea));
           convfea=convfea+output{1,30-i};
         end
     end
   end
     disp(i);

disp(size(convfea));

end
im_h_y = convfea + im_y;
   
     
end
