clear all
close all
%% 最终三个结果融合

testpath = 'E:\STAViS-master\data\video_frames\';
imgDataPathSTANet = 'E:\resultds\STRANet\';
imgDataPathCross = 'E:\resultds\Cross\';
imgDataPathShort = 'E:\resultds\Short\';

savepath    = 'E:\resultds\FuseMy\';
imgDataDir = dir(testpath); % 遍历所有文件
for i = 1:length(imgDataDir)
    if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
    isequal(imgDataDir(i).name,'..')||...
    ~imgDataDir(i).isdir) % 去除遍历中不是文件夹的
        continue;
    end
    imgDir = dir([testpath '\' imgDataDir(i).name]); 
    for j =1:length(imgDir)
        if(isequal(imgDir(j).name,'.')||...
        isequal(imgDir(j).name,'..')||...
        ~imgDir(j).isdir)
            continue;
        end
        imgs = dir([testpath imgDataDir(i).name '\' imgDir(j).name '\*.jpg']);
        for k=1:length(imgs)
            STAP = [imgDataPathSTANet imgDataDir(i).name '\' imgDir(j).name '\' imgs(k).name(1:end-4) '.jpg'];
            CrossP = [imgDataPathCross imgDataDir(i).name '\' imgDir(j).name '\' imgs(k).name(1:end-4) '.png'];
            ShortP = [imgDataPathShort imgDataDir(i).name '\' imgDir(j).name '\' imgs(k).name(1:end-4) '.png'];
            midsta = [];midcross = [];midshort=[];
            if exist(STAP,'file')~=0
                midsta = im2double(imresize(imread(STAP),[356,356]));midsta=MatrixNormalization(midsta);
            end         
            if exist(CrossP,'file')~=0
                midcross = im2double(imresize(imread(CrossP),[356,356]));midcross=MatrixNormalization(midcross);
            end
            if exist(ShortP,'file')~=0
                midshort = im2double(imresize(imread(ShortP),[356,356]));midshort=MatrixNormalization(midshort);
            end
            if isempty(midsta)==0 && isempty(midcross)==0 && isempty(midshort)==0
                if exist([savepath '\' imgDataDir(i).name],'dir')==0
                    mkdir([savepath '\' imgDataDir(i).name]);
                end
                if exist([savepath '\' imgDataDir(i).name '\' imgDir(j).name],'dir')==0
                    mkdir([savepath '\' imgDataDir(i).name '\' imgDir(j).name]);
                end
                midsta = reshape(midsta,[],1);midcross = reshape(midcross,[],1);midshort = reshape(midshort,[],1);
                final=MatrixNormalization(reshape(max([midsta,midcross,midshort],[],2),[356,356]));
                % MatrixNormalization(MatrixNormalization(midsta+midcross+midshort) + MatrixNormalization(midcross.*midsta.*midshort));
                imwrite(final, [savepath '\' imgDataDir(i).name '\' imgDir(j).name '\' imgs(k).name(1:end-4) '.png']);
            end
        end
    end
end
