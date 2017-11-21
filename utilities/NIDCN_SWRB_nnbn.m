function im_h_y = NIDCN_SWRB_nnbn(im_l_y,model)
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
%Y(i,j,k,t) = G(k) * (X(i,j,k,t) - mu(k)) / sigma(k) + B(k)

im_y = single(im_l_y);

disp(size(im_y)); 

convfea = vl_nnconv(im_y,weight{1},bias{1},'Pad',3);

 disp(size(convfea));
 
y=vl_nnbnorm(convfea,scaleG{1},scaleB{1});
 
convfea = vl_nnrelu(y);

disp(size(convfea));

convfea = vl_nnconv(convfea,weight{2},bias{2},'Pad',3);

y=vl_nnbnorm(convfea,scaleG{2},scaleB{2});

convfea = vl_nnrelu(y);

disp(size(convfea));
 
        
         
convfea = vl_nnconv(convfea,weight{3},bias{3},'Pad',3);

y=vl_nnbnorm(convfea,scaleG{3},scaleB{3});

convfea = vl_nnrelu(y);

disp(size(convfea));

convfea = vl_nnconv(convfea,weight{4},bias{4},'Pad',3);

y=vl_nnbnorm(convfea,scaleG{4},scaleB{4});

convfea = vl_nnrelu(y);

disp(size(convfea));
 
convfea = vl_nnconv(convfea,weight{layer_num},bias{layer_num},'Pad',3);
disp(size(convfea));

y=vl_nnbnorm(convfea,scaleG{layer_num},scaleB{layer_num});
 

disp(size(y));
 
im_h_y = y + im_y;
     
    disp(size(im_h_y)); 
    save('output','im_h_y');
end
