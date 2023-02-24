
import os
import sys

import numpy as np
import scipy
import cv2
from dehaze import Dehaze

import time

import multiprocessing
from joblib import Parallel, delayed

import exifreader
from PIL import Image

def getClippingLimits(altitude):
    r_limit = -0.0000011296 * altitude**4 + 0.0000575441 * altitude**3 - 0.0009774864 * altitude**2 + 0.0056842405 * altitude - 0.0017444152
    g_limit = 0.0000038958 * altitude**3 - 0.0001131430 * altitude**2 + 0.0004288439 * altitude + 0.0064228875
    b_limit = 0.0000050696 * altitude**3 - 0.0001263203 * altitude**2 + 0.0005117638 * altitude + 0.0049041834
    return r_limit, g_limit, b_limit

def getClipippingLimitsY(altitude):
    r, g, b = getClippingLimits(altitude)
    # from the formula Y = 0.299R + 0.587G + 0.114B
    return 0.299*r + 0.587*g + 0.114*b

def getBeta(altitude):
    return 0.0001931321* altitude**3 - 0.0071039255* altitude**2 + 0.0850709324* altitude


def denoise_1ch(y, filter_size):
    timer = Timer()
    y = y.astype('float64') / 255
    print(y)
    timer.start()
    y = scipy.signal.wiener(y, filter_size)
    timer.stop_and_disp('Wiener')
    return (y*255).astype(np.uint8)

def denoise(img, filter_size):
    print(type(img))
    print(len(img))
    print(type(img[2]))
    print(len(img[2]))
    print(type(img[:,:,0]))
    print(len(img[:,:,0]))


    for i in range(0,3):
        img[:,:,i] = scipy.signal.wiener(img[:,:,i].astype('float64')/255, filter_size).astype(np.uint8)
    return img

# def denoise(r, g, b, filter_size):
#     r = r/255.0
#     g = g/255.0
#     b = b/255.0
#     r_denoise = (scipy.signal.wiener(r, filter_size[0])*255).astype(np.uint8)
#     g_denoise = (scipy.signal.wiener(g, filter_size[1])*255).astype(np.uint8)
#     b_denoise = (scipy.signal.wiener(b, filter_size[2])*255).astype(np.uint8)
#     # r_denoise = r
#     # g_denoise = g
#     # b_denoise = b
#     # r_denoise = (r_denoise*255).astype(np.uint8)
#     # g_denoise = (g_denoise*255).astype(np.uint8)
#     # b_denoise = (b_denoise*255).astype(np.uint8)
#     return r_denoise, g_denoise, b_denoise

def adaptive_histeq_1ch(img, filter_size, clip_limit=4.0):
    # print(img)
    clahe = cv2.createCLAHE(tileGridSize=filter_size, clipLimit=clip_limit)
    return clahe.apply(img)

def adaptive_histeq(r, g, b, filter_size, clip_limits):
    r_histeq = adaptive_histeq_1ch(r, filter_size[0], clip_limits[0])
    g_histeq = adaptive_histeq_1ch(g, filter_size[1], clip_limits[1])
    b_histeq = adaptive_histeq_1ch(b, filter_size[2], clip_limits[2])
    return r_histeq, g_histeq, b_histeq

# uses 'unsharp masking' technique
def sharpen_image(img, sigma, kernel_size=(0,0), strength=0.8):
    smoothed = cv2.GaussianBlur(img, kernel_size, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    sharpened = cv2.addWeighted(img, 1+strength, smoothed, -strength, 0.0)
    return sharpened

def add_brightness(img, brightness):
    return cv2.convertScaleAbs(img, beta=(33.0*brightness))

def image_complement(img):
    return cv2.bitwise_not(img)

def image_complementY(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y,u,v = cv2.split(yuv)
    y = cv2.bitwise_not(y)
    yuv = cv2.merge([y,u,v])
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return bgr

def dehaze_from_original_example(img, dehaze_amount):
    # A_inv = image_complementY(img)
    # B_inv = Dehaze(A_inv, dehaze_amount)
    # return image_complementY(B_inv)

    return Dehaze(img, dehaze_amount)

def read_image_rgb(imgpath):
    # opencv extracts channels in the bgr format
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    # print(img)
    b, g, r = cv2.split(img)
    return r, g, b


def get_index_of_extension(name):
    return name.find(f".{name.split('.')[-1]}")

def get_filename_noext(name):
    return name[:get_index_of_extension(name)]

class Timer():
    def __init__(self):
        self.startTime = 0
        self.timeTaken = 0

    def start(self):
        self.startTime = time.process_time()

    def stop(self):
        self.timeTaken = time.process_time() - self.startTime

    def stop_and_disp(self, label):
        self.stop()
        self.print(label)

    def print(self, label):
        print(f'{label}: {self.timeTaken}')

def processImage(input_imgpath, output_imgpath):
    overallTimer = Timer()
    overallTimer.start()

    timer = Timer()

    PILImage = Image.open(input_imgpath)

    # get altitude metadata
    timer.start()
    altitude = exifreader.getAltitude(PILImage)
    if altitude <= 0.0:
        altitude = 6
    timer.stop_and_disp('getAltitude')


    print(f'Altitude={altitude}')


    # histeq_size = ((8,6),(8,6),(8,6))
    histeq_size = ((6,8),(6,8),(6,8))

    clip_limits = getClippingLimits(altitude) 
    # Scale by 255 as possible values are [0,1] in matlab and [0,255] in opencv
    clip_limits = [255*i for i in clip_limits]   
    
    timer.start()
    # r, g, b = read_image_rgb(input_imgpath)
    img = cv2.imread(input_imgpath, cv2.IMREAD_COLOR)
    timer.stop_and_disp('Read')

    b, g, r = cv2.split(img)


    # 1. DENOISING
    timer.start()
    b = denoise_1ch(b, (4, 4))
    g = denoise_1ch(g, (5, 5))
    r = denoise_1ch(r, (6, 6))
    timer.stop_and_disp('Denoise')

    # 2. ADAPTIVE HIST EQ
    print(f'ClipLimits={clip_limits}')
    timer.start()
    r, g, b = adaptive_histeq(r, g, b, filter_size=histeq_size, clip_limits=clip_limits)
    timer.stop_and_disp('Adaptive HistEq')

    # img = cv2.merge([b, g, r])
    # cv2.imwrite(output_imgpath+'equalized.jpg', img)

    # 3. SHARPENING
    timer.start()
    img = sharpen_image(img, 2, strength=1.2)
    timer.stop_and_disp('Sharpen')

    # 4. DEHAZING
    timer.start()
    dehaze_amount = altitude / 120
    final_img = dehaze_from_original_example(img, dehaze_amount)
    timer.stop_and_disp('Dehaze')

    # 5. ADDING BRIGHTNESS
    timer.start()
    print(f'beta={getBeta(altitude)}')
    final_img = add_brightness(final_img, getBeta(altitude))
    timer.stop_and_disp('Add Brightness')

    print(final_img)

    # cv2.imwrite(output_imgpath, final_img)

    # convert to rgb image, save with exif data from PILImage
    output_img_data = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    output_img_data.save(output_imgpath, quality=100, exif=PILImage.info['exif'])


    overallTimer.stop_and_disp('OVERALL')

def processImage_batch(batch):
    start = time.process_time()
    for img in batch:
        processImage(img[0], img[1])
    print(f'Time elapsed: {time.process_time() - start}')


def processImage_multi(inputs, cpu_load=0.9):
    len_inputs = len(inputs)
    print(f'len_inputs={len_inputs}')
    n_batches = np.minimum(len_inputs, multiprocessing.cpu_count())

    # assign batches based on load
    n_batches = np.maximum(1, np.minimum(int(n_batches*cpu_load), n_batches-2))
    print(f'n_batches={n_batches}')

    # Divide batches evenly
    batch_size = len_inputs // n_batches
    excess = len_inputs % n_batches
    cursor = 0
    batches = []
    for i in range(n_batches):
        start = cursor
        # end = (i + 1) * batch_size if i != n_batches - 1 else len_inputs
        if excess >= 0:
            end = start + batch_size + 1
            excess = excess - 1
        else:
            end = start + batch_size if (start + batch_size < len_inputs) else len_inputs
        cursor = end
        batch = inputs[start:end]
        print(f'Batch = inputs[{start}:{end}]')
        batches.append(batch)

    return Parallel(n_jobs=n_batches)(
        delayed(processImage_batch)(batch) for batch in batches
    )

def isImageFile(name):
    ext = name[get_index_of_extension(name)+1:]
    valid_ext = ['jpg','jpeg','png','bmp']
    print(ext.lower())
    return ext.lower() in valid_ext

def photoenhance(target=''):
    if target:
        imgdir=target
        inputs = []
        for imgfile in os.listdir(imgdir):
                input_imgpath = os.path.join(imgdir, imgfile)
                if os.path.isfile(input_imgpath) and isImageFile(input_imgpath):

                    output_imgpath = get_filename_noext(input_imgpath) + '_enh.jpg'
                    
                    # if not os.path.exists(output_imgpath):
                    if '_enh' not in input_imgpath:
                        print(f'\nProcessing {input_imgpath} -> {output_imgpath}')
                        # processImage(input_imgpath, output_imgpath)
                        inputs.append([input_imgpath, output_imgpath])

        processImage_multi(inputs)
    else:
        print('INVALID TARGET: Directory does not exist')

def main(**kwargs):
    for key, value in kwargs.items():
        print(f'Arg: {key}={value}')
    photoenhance(**kwargs)


if __name__=='__main__':
    sysargs = [arg.replace("--","") for arg in sys.argv[1:]]
    main(**dict(arg.split('=') if '=' in arg else [arg, 'True'] for arg in sysargs))