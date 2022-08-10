close all;
clear all;
fidpS = 'G:\ECCV_test\ClassS.txt';
[pathS, prdS, probS, labelS] = textread(fidpS, '%s%s%s%s', 'delimiter', '+');
fidpA = 'G:\ECCV_test\ClassA.txt';
[pathA, prdA, probA, labelA] = textread(fidpA, '%s%s%s%s', 'delimiter', '+');
fidpT = 'G:\ECCV_test\ClassT.txt';
[pathT, prdT, probT, labelT] = textread(fidpT, '%s%s%s%s', 'delimiter', '+');
LengthFiles = length(pathS);
crop_path = 'F:\crop\crop\';
rgb_path = 'F:\crop\rgb\';
txt_path = 'F:\crop\txt\';

rect = [];
ImgIndex = 300001;
while(ImgIndex<=400000)
    CImgIndex = ImgIndex;
        if(CImgIndex<=LengthFiles)
            imgpath = char(pathS(CImgIndex, :));
            path_len = length(dir(['G:\ECCV_test\result' imgpath(1:end-10)]));
            index = str2num(imgpath(end-9:end-6));
            path_all = strsplit(imgpath, '\');
            if ~exist([crop_path path_all{2}],'dir')   mkdir([crop_path path_all{2}]);  end
            if ~exist([rgb_path path_all{2}],'dir')   mkdir([rgb_path path_all{2}]);  end
            if ~exist([txt_path path_all{2}],'dir')   mkdir([txt_path path_all{2}]);  end
            if ~exist([crop_path path_all{2} '\' path_all{3}],'dir')   mkdir([crop_path path_all{2} '\' path_all{3}]);  end
            if ~exist([rgb_path path_all{2} '\' path_all{3}],'dir')   mkdir([rgb_path path_all{2}  '\' path_all{3}]);  end
            if ~exist([txt_path path_all{2}  '\' path_all{3}],'dir')   mkdir([txt_path path_all{2}  '\' path_all{3}]);  end

            V_path = ['G:\ECCV_test\Vresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            A_path = ['G:\ECCV_test\Aresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            T_path = ['G:\ECCV_test\Tresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            
            index
            V = MatrixNormalization(im2double(imread(V_path)));
            A = MatrixNormalization(im2double(imread(A_path)));
            T = MatrixNormalization(im2double(imread(T_path)));
            Up = V.*str2double(probS(CImgIndex))+A.*str2double(probA(CImgIndex))+T.*str2double(probT(CImgIndex))+0.0001;
            Down = str2double(probS(CImgIndex))+str2double(probA(CImgIndex))+str2double(probT(CImgIndex))+0.0001;
            F = MatrixNormalization(Up./Down);
            [row, col]  = find(F(:,:) > 2*mean(mean(F)));
            max_col = max(col);
            min_col = min(col);
            max_row = max(row);
            min_row = min(row);
            if ~(isempty(max_col)|isempty(min_col)|isempty(max_row)|isempty(min_row))
                rect = [min_row, max_row, min_col, max_col];
                RGB_path = ['G:\AVE-ECCV18-master\AVE_Dataset\img2' imgpath(1:end-6) '.jpg'];
                RGB = im2double(imresize(imread(RGB_path), [356, 356]));
                result = RGB(min_row:max_row, min_col:max_col,:);
                fid=fopen([txt_path path_all{2}  '\' path_all{3} '\' path_all{4}(1:end-6) '.txt'],'w');
                fprintf(fid,'%d %d %d %d\n', min_row, max_row, min_col, max_col);
                imwrite(result, [crop_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
                fclose(fid);
            end
       end
ImgIndex = ImgIndex + 1;
end
fprintf('done!\n');
fclose(fidp);