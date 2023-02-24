% input root directory with the images

inputPath='C:\Users\pteneder\Documents\photo_enhancer\data\images\test_image_2';

% actual files to read (just jpegs)
inputfiles=dir('C:\Users\pteneder\Documents\photo_enhancer\data\images\test_image_2\*.jpg');

%output path
outputFiles='C:\Users\pteneder\Documents\photo_enhancer\data\images\test_image_2';

disp('Hello world')

tic
% read all files
for k=1:length(inputfiles)
    disp('Hello world')
    fn=fullfile(inputfiles(k).folder, inputfiles(k).name);

    [folder, baseFileNameNoExt, extension] = fileparts(fn);

    % create the output directory if it doesn't exist

    newFilePath = (strcat(outputFiles,extractAfter(folder,inputPath)));

    if ~exist(newFilePath, 'dir')
       mkdir(newFilePath)
    end

    % get the bits of the filke name to create the output file name, this
    % bit needs cleaning up
    adaptHistName = strcat(newFilePath,'\',baseFileNameNoExt, '_enh_m.jpg');
    disp(adaptHistName)
    oName = adaptHistName;
    outputFileName = fullfile(adaptHistName);
    baseFileName = inputfiles(k).name;
    inputFileName = fullfile(inputfiles(k).folder, baseFileName);
    iName = strcat(inputfiles(k).folder,'\',baseFileName);

    % check if the output file exists
    if isfile(outputFileName)

        fprintf(1, 'Skipping: File already exists - %s\n', outputFileName);

    elseif contains(inputfiles(k).name, '_enh')

        fprintf(1, 'Skipping: not deemed as input file %s\n', inputfiles(k).name)

    else

        % if not then read the exif data from the input file to get the
        % altitude
        fprintf(1, 'Now reading %s\n', fn);

        % get the exif altitude data from the file if prtesent

        try
            exifInfo = imfinfo(fn);
            altitude = (exifInfo.GPSInfo.GPSAltitude);
            if altitude > 20
                altitude = altitude / 10.0;
            end
            disp(altitude);
        catch

            % if not set to 6m as an average
            altitude = 6.0;
        end

        % colour balance and altitude equation done empirically - this is
        % what we will need to re-do in Python

        bRedClipLimit = -0.0000011296 * altitude^4 + 0.0000575441 * altitude^3 - 0.0009774864 * altitude^2 + 0.0056842405 * altitude - 0.0017444152;
        bGreenClipLimit = 0.0000038958 * altitude^3 - 0.0001131430 * altitude^2 + 0.0004288439 * altitude + 0.0064228875;
        bBlueClipLimit = 0.0000050696 * altitude^3 - 0.0001263203 * altitude^2 + 0.0005117638 * altitude + 0.0049041834;


        beta = 0.0001931321* altitude^3 - 0.0071039255* altitude^2 + 0.0850709324* altitude;

        imageFile = imread(fn);



        % extract the colour channels
        redChannel = imageFile(:,:,1); % Red channel
        greenChannel = imageFile(:,:,2); % Green channel
        blueChannel = imageFile(:,:,3); % Blue channel

        % remove noise from each channel
        redChannel_denoise = wiener2(redChannel,[6 6]);
        greenChannel_denoise = wiener2(greenChannel,[5 5]);
        blueChannel_denoise = wiener2(blueChannel,[4 4]);

        % do an adaptive histogram stretch using the clip limits from above
        adaptImageRed = adapthisteq(redChannel_denoise,'NumTiles',[8, 6], ClipLimit=bRedClipLimit, Range="full", NBins=2048, Distribution='uniform');
        adaptImageGreen = adapthisteq(greenChannel_denoise,'NumTiles',[8, 6], ClipLimit=bGreenClipLimit, Range="full", NBins=2048, Distribution='uniform');
        adaptImageBlue = adapthisteq(blueChannel_denoise,'NumTiles',[8, 6], ClipLimit=bBlueClipLimit, Range="full", NBins=2048, Distribution='uniform');

        % merge the channnel back into an image
        adaptImage = cat(3, adaptImageRed, adaptImageGreen, adaptImageBlue);



        % sharpen the image
        sharpImage = imsharpen(adaptImage,'Radius',2,'Amount',1.2);

        % dehaze the image - use the altoitude to de-noise more for deeper
        % images
        dehaze_amount = altitude / 120;

        AInv = imcomplement(sharpImage);
        BInv = imreducehaze(AInv, dehaze_amount, 'Method','approxdcp', 'ContrastEnhancement','none');
        deHazeImage = imcomplement(BInv);

        % optinally brighten the final image as the de-hazing can make it
        % darker

        finalImage = imlocalbrighten(deHazeImage,beta);

        % write the EXIF dat from the opriginal to the new file
        % in this case I am just using the non-brightenede image as this
        % was not working that well...

        status = putexif(deHazeImage, oName, iName);

        % so final image is the deHazeImage image
      end
end
toc

function [exifdata, nf] = getexif(fname)
%[status] = getexif(fname) read Exif data from an image file
% fname   = name of image file, e.g., 'myfile.jpg'
%
%Needs ExifTool.exe, written by Phil Harvey
% http://www.sno.phy.queensu.ca/~phil/exiftool/
% 1. Download the file, exiftool(-k).exe
% 2. Rename this file, exiftool.exe
% 3. Save this file in a folder on your Matlab path, e.g. .../matlab/
%Peter Burns, 28 May 2013
%             22 July 2013, following suggestions from jhh and Jonathan via
%                           Matlab Central.
test = which('exiftool.exe');
if isempty(test)
    disp('ExifTool not available:');
    disp('Please download from,')
    disp('   http://www.sno.phy.queensu.ca/~phil/exiftool/')
    disp('make sure that the installed exiftool.exe is on your Matlab path')
    beep
    exifdata=[];
    nf = 0;
    return
else
    
TS=[ '"' test '" -s "' fname '"']; 
[status, exifdata] = system(TS); 
       
nf = find(exifdata==':');
nf = length(nf);
end
end

function [status] = putexif(dat,fname, refname)
%[status] = putexif(dat,fname, refname) save and image with Exif data from
%                  a reference image file
% dat     = array of image data
% fname   = name of image file to be saved, e.g., 'myfile.jpg'
% refname = name (path) of reference image file
%
%Needs ExifTool.exe, written by Phil Harvey
% http://www.sno.phy.queensu.ca/~phil/exiftool/
% 1. Download the file, exiftool(-k).exe
% 2. Rename this file, exiftool.exe
% 3. Save this file in a folder on your Matlab path, e.g. .../matlab
%Peter Burns, 28 May 2013
%             22 July 2013, following suggestions from jhh and Jonathan via
%                           Matlab Central.
test = which('exiftool.exe');
if isempty(test)
    disp('ExifTool not available:');
    disp('Please download from,')
    disp('   http://www.sno.phy.queensu.ca/~phil/exiftool/')
    disp('or make sure that the installed exiftool.exe is on your Matlab path')
    beep
    status =1;
    return
else
    
    [exifdata, nf] = getexif(refname);
   % Save data as image file with Matlab metadata
    imwrite(dat, fname, 'quality', 100);
    [exifdata2, nf2] = getexif(fname);
   % Replace matadata from this file with desired tags from reference file
   % temp1=['exiftool -m -tagsfromfile ',refname,' -all:all ', fname];
    temp1=['"' test '" -m -tagsfromfile "',refname,'" -all:all "', fname, '"'];
    
    %[status, junk] = system(temp1);
    %Second output variable appears to suppress '1 image files updated'
    % being returned to Command window (~ also works).
    [status, junk] = system(temp1);
   
   % Approximate check
    [exifdata3, nf3] = getexif(fname);
    
   % Delete extra copy of reference image file
    temp3 = ['del "',fname,'_original"'];
    status = system(temp3);
    
   % Test approx. number of tags
    if abs(nf3-nf2<10)
        disp('Warning: Exif tags may not have been copied');
        status = 1;
    end
    
end
end
