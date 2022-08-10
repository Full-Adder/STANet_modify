close all;
clear all;
fidp = 'G:\ECCV_test\class.txt';
[path, prd, prob, label] = textread(fidp, '%s%s%s%s', 'delimiter', '+');
ImgIndex = 1;
LengthFiles = length(path);
crop_path = 'F:\crop\crop\';
rgb_path = 'F:\crop\rgb\';
txt_path = 'F:\crop\txt\';

rect = [];
while(ImgIndex<=LengthFiles)
  parfor i=1:6
      if(i==1)
          CImgIndex = ImgIndex;
          if(CImgIndex<=LengthFiles)
              rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
              CImgIndex
          end
      end
      if(i==2)
          CImgIndex = ImgIndex+1;
          if(CImgIndex<=LengthFiles)
              rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
              CImgIndex
          end
      end
      if(i==3)
          CImgIndex = ImgIndex+2;
          if(CImgIndex<=LengthFiles)
               rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
               CImgIndex
          end
      end
      if(i==4)
          CImgIndex = ImgIndex+3;
          if(CImgIndex<=LengthFiles)
               rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
               CImgIndex
          end
      end
      if(i==5)
          CImgIndex = ImgIndex+4;
          if(CImgIndex<=LengthFiles)
               rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
               CImgIndex
          end
      end
      if(i==6)
          CImgIndex = ImgIndex+5;
          if(CImgIndex<=LengthFiles)
               rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
               CImgIndex
          end
      end
%        if(i==7)
%           CImgIndex = ImgIndex+6;
%           if(CImgIndex<=LengthFiles)
%               rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
%           end
%       end
%       if(i==8)
%           CImgIndex = ImgIndex+7;
%           if(CImgIndex<=LengthFiles)
%               rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
%           end
%       end
%       if(i==9)
%           CImgIndex = ImgIndex+8;
%           if(CImgIndex<=LengthFiles)
%                rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
%           end
%       end
%       if(i==10)
%           CImgIndex = ImgIndex+9;
%           if(CImgIndex<=LengthFiles)
%                rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
%           end
%       end
%       if(i==11)
%           CImgIndex = ImgIndex+10;
%           if(CImgIndex<=LengthFiles)
%                rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
%           end
%       end
%       if(i==12)
%           CImgIndex = ImgIndex+11;
%           if(CImgIndex<=LengthFiles)
%                rect = computecrop(rect, CImgIndex, path, crop_path, rgb_path, txt_path);
%           end
%       end
  end
ImgIndex = ImgIndex + 6;
end
fprintf('done!\n');
fclose(fidp);