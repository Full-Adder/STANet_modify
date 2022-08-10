close all;
clear all;
fidp = 'F:\wgt\CROP\txt\4.txt';
[path, prd, prob, label] = textread(fidp, '%s%s%s%s', 'delimiter', '+');
ImgIndex = 1;
LengthFiles = length(path);
save_path = 'F:\wgt\Crop_data\smooth\';
rect = [];
V_ALL = [];
for CImgIndex=1:LengthFiles
    imgpath = char(path(CImgIndex, :));
    path_len = length(dir(['F:\wgt\Crop_data\cam' imgpath(1:end-10)]));
    index = str2num(imgpath(end-9:end-6));
    AV_path = ['F:\wgt\Crop_data\cam' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
    path_all = strsplit(imgpath, '\');
    if ~exist([save_path path_all{2}],'dir')   mkdir([save_path path_all{2}]);  end
    if ~exist([save_path path_all{2} '\' path_all{3}],'dir')   mkdir([save_path path_all{2} '\' path_all{3}]);  end

    if index == 3
        for xmm = -2:2
            V_path = ['F:\wgt\Crop_data\cam' imgpath(1:end-10) num2str(index+xmm, '%04d') '_c.jpg'];
                if ~exist(V_path,'file') 
                    continue; 
                end
            V = MatrixNormalization(im2double(imread(V_path)));
            V_ALL{xmm+3} = V;
        end
        VHH = MatrixNormalization((V_ALL{1}+V_ALL{2}+V_ALL{3}+V_ALL{4}+V_ALL{5})./5);
        V_ALL{5} = VHH;
    end
    if index>=4 & index<path_len-5
        V_path = ['F:\wgt\Crop_data\cam' imgpath(1:end-10) num2str(index+2, '%04d') '_c.jpg'];
        index
        if ~exist(V_path,'file') 
            continue; 
        end
        V = MatrixNormalization(im2double(imread(V_path)));
        V_ALL{1} = V_ALL{2};V_ALL{2}=V_ALL{3};V_ALL{3}=V_ALL{4};V_ALL{4}=V_ALL{5};V_ALL{5}=V;
        VHH = MatrixNormalization((V_ALL{1}+V_ALL{2}+V_ALL{3}+V_ALL{4}+V_ALL{5})./5);
        V_ALL{5} = VHH;
        VHH(find(VHH<3*mean(mean(VHH))))=0;
%                 figure(3),imshow(VHH);
%                 ALL = zeros(356,356,3);
%                 ALL(:,:,1) = VHH;ALL(:,:,2) = VHH;
%                 figure(4),imshow(imadd(RGB*0.5, 0.5*ALL));
        imwrite(VHH, [save_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
    end
end
fprintf('done!\n');
fclose(fidp);