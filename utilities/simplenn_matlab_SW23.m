function res = simplenn_matlab_SW23(net, input)

%% If you did not install the matconvnet package, you can use this for testing.

n = numel(net.layers);
%disp(n);
res = struct('x', cell(1,n+1));
res(1).x = input;

for ilayer = 1 : n
    l = net.layers{ilayer};
    switch l.type
        case 'conv'
            for noutmaps = 1 : size(l.weights{1},4)
  %%              disp(['size(l.weights{1},4):' num2str(size(l.weights{1},4))]);
                z = zeros(size(res(ilayer).x,1),size(res(ilayer).x,2),'single');
                tmp=size(res(ilayer).x,3);
                if ilayer==5
                  % disp(size(res(ilayer).x,3)); 
                   tmp=64; 
                end      
              %  disp(['ilayer:' num2str(ilayer)]);       
                for ninmaps = 1 : tmp %size(res(ilayer).x,3)
%                  disp(['tmp:' num2str(tmp)]);
 %                 disp(['ilayer:' num2str(ilayer) ',ninmaps:  ' num2str(ninmaps) ',noutmaps:  ' num2str(noutmaps)]); 
                   %disp( l.weights{1}(:,:,ninmaps,noutmaps));
                    z = z + convn(res(ilayer).x(:,:,ninmaps), l.weights{1}(:,:,ninmaps,noutmaps),'same');                                  
                end
                res(ilayer+1).x(:,:,noutmaps) = z + l.weights{2}(noutmaps);
            end
        case 'relu'
            disp(ilayer);
            res(ilayer+1).x = max(res(ilayer).x,0);
    end
    res(ilayer).x = [];
end

end
