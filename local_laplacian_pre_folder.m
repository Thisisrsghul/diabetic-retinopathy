clear; clc; close all;

%% 1. Input/Output Directory Parameters
inputFolder  = 'D:\diabetic-retinopathy\EyePACS\train\left';  % Directory containing input images
outputFolder = 'D:\diabetic-retinopathy\EyePACS\train\left_local_laplacian';    % Directory to save enhanced results
outputSize   = 512;                                           % Desired target size (512x512)

%% 2. Folder Verification & Setup
if ~exist(inputFolder, 'dir')
    error('Input folder "%s" not found! Check the path and try again.', inputFolder);
end

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
    fprintf('Created output folder: %s\n', outputFolder);
end

%% 3. Retrieve Image Files
% Search for common image formats
imgExtensions = {'*.jpeg', '*.jpg', '*.png', '*.tif', '*.bmp'};
imgFiles = [];

for ext = imgExtensions
    imgFiles = [imgFiles; dir(fullfile(inputFolder, ext{1}))]; %#ok<AGROW>
end

numFiles = length(imgFiles);
if numFiles == 0
    error('No compatible image files found in folder: %s', inputFolder);
end

fprintf('Found %d image(s) to process.\n\n', numFiles);

%% 4. Batch Processing Loop
tic;
successful = 0;
failed = 0;

parfor i = 1:numFiles
    fileName = imgFiles(i).name;
    currentInputPath  = fullfile(inputFolder, fileName);
    currentOutputPath = fullfile(outputFolder, fileName);
    
    % --- RESUME CHECK: Skip if file already exists ---
    if exist(currentOutputPath, 'file')
        fprintf('[%d/%d] Skipping (already processed): %s\n', i, numFiles, fileName);
        continue;
    end
    
    fprintf('[%d/%d] Processing: %s ... ', i, numFiles, fileName);
    
    try
        % Run the enhancement pipeline
        enhancedImg = fundusEnhancePipeline(currentInputPath, outputSize);
        
        % Save output image
        imwrite(enhancedImg, currentOutputPath);
        fprintf('Saved.\n');
        
    catch ME
        % Handle processing errors gracefully without breaking the batch loop
        fprintf('FAILED! Error: %s\n', ME.message);
    end
end

elapsedTime = toc;
fprintf('\n========================================================\n');
fprintf('Batch Processing Finished in %.2f seconds!\n', elapsedTime);
fprintf('Successfully Processed: %d | Failed: %d\n', successful, failed);
fprintf('Enhanced images saved to: %s\n', outputFolder);
fprintf('========================================================\n');


%% ========================================================================
% HELPER FUNCTION
% ========================================================================
function enhanced = fundusEnhancePipeline(imgPath, outSize)
% FUNDUSENHANCEPIPELINE Image enhancement pipeline for fundus images
if nargin < 2
    outSize = 512;
end

%% 1. Load & standardize
img = imread(imgPath);
img = im2double(img);

%% 2. FOV mask extraction (threshold on green channel)
greenCh = img(:,:,2);
fovMask = greenCh > graythresh(greenCh) * 0.3;   % low threshold
fovMask = imfill(fovMask, 'holes');
fovMask = bwareafilt(fovMask, 1);                % keep largest connected component

%% 3. Crop to FOV bounding box
stats = regionprops(fovMask, 'BoundingBox');
if isempty(stats)
    % Fallback if mask detection fails
    fovMask = true(size(greenCh));
    bbox = [1, 1, size(img, 2)-1, size(img, 1)-1];
else
    bbox = round(stats(1).BoundingBox);
end

img = imcrop(img, bbox);
fovMask = imcrop(fovMask, bbox);

%% Resize to fixed resolution
img = imresize(img, [outSize outSize]);
fovMask = imresize(fovMask, [outSize outSize], 'nearest');

%% 4. Channel selection - use green channel (best vessel contrast)
greenCh = img(:,:,2);

%% 5. Denoising
denoised = greenCh;

%% 6. Illumination correction (divisive, using Gaussian blur as background)
bgSigma = 30;
background = imgaussfilt(denoised, bgSigma);
meanGray = mean(denoised(fovMask));
corrected = (denoised ./ (background + eps)) * meanGray;
corrected = mat2gray(corrected);

%% 7. Contrast & Detail Enhancement - Local Laplacian Filter
sigma = 0.2;
alpha = 0.3;
beta  = 1.0;

correctedSingle = single(corrected);
enhancedDetail = locallapfilt(correctedSingle, sigma, alpha, beta);
enhancedDetail = double(enhancedDetail);

%% 8. Gamma correction
gamma = 0.9;
gammaImg = imadjust(enhancedDetail, [], [], gamma);

%% 9. Re-mask with FOV (remove border artifacts)
gammaImg(~fovMask) = 0;

%% 10. Normalization
enhanced = mat2gray(gammaImg);
end