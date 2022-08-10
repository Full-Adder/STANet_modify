%% 需要SAT,ST-short,SAT-long,SAT-Cross帧已经对齐。
close all;
clear all;
% Coarse-Fine
fidpS = 'G:\ECCV_test\ClassS.txt';
[pathS, prdS, probS, labelS] = textread(fidpS, '%s%s%s%s', 'delimiter', '+');
fidpA = 'G:\ECCV_test\ClassA.txt';
[pathA, prdA, probA, labelA] = textread(fidpA, '%s%s%s%s', 'delimiter', '+');
fidpT = 'G:\ECCV_test\ClassT.txt';
[pathT, prdT, probT, labelT] = textread(fidpT, '%s%s%s%s', 'delimiter', '+');
% Short
fidpSs = 'G:\ECCV_test\ClassSshort.txt';
[pathSs, prdSs, probSs, labelSs] = textread(fidpSs, '%s%s%s%s', 'delimiter', '+');
fidpTs = 'G:\ECCV_test\ClassTshort.txt';
[pathTs, prdTs, probTs, labelTs] = textread(fidpTs, '%s%s%s%s', 'delimiter', '+');
% Long
fidpSl = 'G:\ECCV_test\ClassSlong.txt';
[pathSl, prdSl, probSl, labelSl] = textread(fidpSl, '%s%s%s%s', 'delimiter', '+');
fidpAl = 'G:\ECCV_test\ClassAlong.txt';
[pathAl, prdAl, probAl, labelAl] = textread(fidpAl, '%s%s%s%s', 'delimiter', '+');
fidpTl = 'G:\ECCV_test\ClassTlong.txt';
[pathTl, prdTl, probTl, labelTl] = textread(fidpTl, '%s%s%s%s', 'delimiter', '+');
% Cross
fidpSc = 'G:\ECCV_test\ClassScross.txt';
[pathSc, prdSc, probSc, labelSc] = textread(fidpSc, '%s%s%s%s', 'delimiter', '+');
fidpAc = 'G:\ECCV_test\ClassAcross.txt';
[pathAc, prdAc, probAc, labelAc] = textread(fidpAc, '%s%s%s%s', 'delimiter', '+');
fidpTc = 'G:\ECCV_test\ClassTcross.txt';
[pathTc, prdTc, probTc, labelTc] = textread(fidpTc, '%s%s%s%s', 'delimiter', '+');

LengthFiles = length(pathS);

ImgIndex = 1200001;
while(ImgIndex<=1300000)
    CImgIndex = ImgIndex;
        if(CImgIndex<=LengthFiles)
            imgpath = char(pathS(CImgIndex, :));
            path_len = length(dir(['G:\ECCV_test\result' imgpath(1:end-10)]));
            index = str2num(imgpath(end-9:end-6));
            path_all = strsplit(imgpath, '\');

            V_path = ['G:\ECCV_test\Vresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            A_path = ['G:\ECCV_test\Aresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            T_path = ['G:\ECCV_test\Tresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            VS_path = ['G:\ECCV_test\VSresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            TS_path = ['G:\ECCV_test\TSresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            VL_path = ['G:\ECCV_test\VLresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            AL_path = ['G:\ECCV_test\ALresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            TL_path = ['G:\ECCV_test\TLresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            VC_path = ['G:\ECCV_test\VCresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            AC_path = ['G:\ECCV_test\ACresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            TC_path = ['G:\ECCV_test\TCresult' imgpath(1:end-10) num2str(index, '%04d') '_c.jpg'];
            
            index
            V = MatrixNormalization(im2double(imread(V_path)));
            A = MatrixNormalization(im2double(imread(A_path)));
            T = MatrixNormalization(im2double(imread(T_path)));

            VS = MatrixNormalization(im2double(imread(VS_path)));
            TS = MatrixNormalization(im2double(imread(TS_path)));

            VL = MatrixNormalization(im2double(imread(VL_path)));
            AL = MatrixNormalization(im2double(imread(AL_path)));
            TL = MatrixNormalization(im2double(imread(TL_path)));

            VC = MatrixNormalization(im2double(imread(VC_path)));
            AC = MatrixNormalization(im2double(imread(AC_path)));
            TC = MatrixNormalization(im2double(imread(TC_path)));

            Up = V.*str2double(probS(CImgIndex))+A.*str2double(probA(CImgIndex))+T.*str2double(probT(CImgIndex))...
                                    +VS.*str2double(probSs(CImgIndex))+TS.*str2double(probTs(CImgIndex))...
                                    +VL.*str2double(probSl(CImgIndex))+AL.*str2double(probAl(CImgIndex))+TL.*str2double(probTl(CImgIndex))...
                                    +VC.*str2double(probSc(CImgIndex))+AC.*str2double(probAc(CImgIndex))+TC.*str2double(probTc(CImgIndex))+0.0001;
            Down = str2double(probS(CImgIndex))+str2double(probA(CImgIndex))+str2double(probT(CImgIndex))...
                                    str2double(probSs(CImgIndex))+str2double(probTs(CImgIndex))...
                                    str2double(probSl(CImgIndex))+str2double(probAl(CImgIndex))+str2double(probTl(CImgIndex))...
                                    str2double(probSc(CImgIndex))+str2double(probAc(CImgIndex))+str2double(probTc(CImgIndex))+0.0001;
            F = MatrixNormalization(Up./Down);
            imwrite(F, [crop_path path_all{2}  '\' path_all{3} '\' path_all{4}]);
        end
ImgIndex = ImgIndex + 1;
end
fprintf('done!\n');
fclose(fidp);