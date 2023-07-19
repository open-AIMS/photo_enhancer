
import os
import sys

import numpy as np
import scipy
import cv2
from PIL import Image

import time
import multiprocessing
from joblib import Parallel, delayed
from threading import Thread

from utils import exifreader

if __name__ == "__main__":
    from dehaze.dehaze import Dehaze
    from utils.exifreader import getAltitude
    from utils.perftimer import Timer
else:
    from photoenhancer.dehaze.dehaze import Dehaze
    from photoenhancer.utils.exifreader import getAltitude
    from photoenhancer.utils.perftimer import Timer  

overallTimer = Timer()

class EnhancerParameters():
    enable_stronger_contrast_deep: bool = False
    denoising: bool = True

class BatchMonitor():
    n_images_completed: int = 0
    n_images_total: int = 0
    ontick_callback_func = lambda x, y: None
    n_batches: int = 1
    cancelled: bool = False
    finished: bool = False

    def set_cancelled(self):
        self.cancelled = True

    def get_total_images(self):
        return self.n_images_total

    def set_total_images(self, n_images_total):
        self.n_images_total = n_images_total

    def get_completed_images(self):
        return self.n_images_completed

    def initialise_tick(self):
        if self.n_images_total > 0:
            self.on_tick(0, self.n_images_total)

    def tick(self):
        self.n_images_completed += 1
        self.on_tick(self.n_images_completed, self.n_images_total)

    def set_callback_on_tick(self, callback_func):
        self.ontick_callback_func = callback_func

    def on_tick(self, completed, total):
        self.ontick_callback_func(completed, total)


def getScipySignalPath():
    return scipy.signal

def getScipyVersion():
    return scipy.__version__

def getScipy():
    return scipy

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
    y = y.astype('float64') / 255
    y = scipy.signal.wiener(y, filter_size)
    return (y*255).astype(np.uint8)

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
    return cv2.convertScaleAbs(img, beta=2.55*brightness)

def image_complement(img):
    # return cv2.bitwise_not(img)
    return 255-img

def image_complementY(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y,u,v = cv2.split(yuv)
    y = cv2.bitwise_not(y)
    yuv = cv2.merge([y,u,v])
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return bgr

def dehaze_from_original_example(img, dehaze_amount):
    A_inv = image_complement(img)
    B_inv = Dehaze(A_inv, dehaze_amount)
    return image_complement(B_inv)

    # return Dehaze(img, dehaze_amount)

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


def processImage(input_imgpath, output_imgpath, enhancer_params=EnhancerParameters(), batch_monitor=BatchMonitor(), stepTimer=Timer()):
    overallTimer.start()

    PILImage = Image.open(input_imgpath)

    # get altitude metadata
    stepTimer.start()
    exif_data = PILImage._getexif()
    altitude = exifreader.getAltitude(exif_data)
    if altitude <= 0.0:
        print('Using default altitude of 6')
        altitude = 6
    stepTimer.stop_and_disp('getAltitude')

    print(f'Processing file {input_imgpath}')
    print(f'Altitude={altitude}')


    # histeq_size = ((8,6),(8,6),(8,6))
    histeq_size = ((6,8),(6,8),(6,8))

    clip_limits = getClippingLimits(altitude) 
    # Scale by 255 as possible values are [0,1] in matlab and [0,255] in opencv
    clip_limits = [255*i for i in clip_limits]   
    
    stepTimer.start()
    # r, g, b = read_image_rgb(input_imgpath)
    img = cv2.imread(input_imgpath, cv2.IMREAD_COLOR)
    stepTimer.stop_and_disp('Read')

    b, g, r = cv2.split(img)

    # Check if photoenhance was cancelled by some top-level process
    if batch_monitor.cancelled:
        return

    # 1. DENOISING
    if enhancer_params.denoising:
        stepTimer.start()
        b = denoise_1ch(b, (4, 4))
        g = denoise_1ch(g, (5, 5))
        r = denoise_1ch(r, (6, 6))
        stepTimer.stop_and_disp('Denoise')

    # Check if photoenhance was cancelled by some top-level process
    if batch_monitor.cancelled:
        return

    # 2. ADAPTIVE HIST EQ
    print(f'ClipLimits={clip_limits}')
    stepTimer.start()
    if altitude < 6:
        clip_limits = [altitude**3.5 / 529 * i for i in clip_limits]
    r, g, b = adaptive_histeq(r, g, b, filter_size=histeq_size, clip_limits=clip_limits)
    stepTimer.stop_and_disp('Adaptive HistEq')

    img = cv2.merge([b, g, r])
    # cv2.imwrite(output_imgpath+'equalized.jpg', img)

    # Check if photoenhance was cancelled by some top-level process
    if batch_monitor.cancelled:
        return

    # 3. SHARPENING
    stepTimer.start()
    img = sharpen_image(img, 2, strength=1.2)
    stepTimer.stop_and_disp('Sharpen')

    # Check if photoenhance was cancelled by some top-level process
    if batch_monitor.cancelled:
        return

    # 4. DEHAZING
    stepTimer.start()
    # dehaze_amount = altitude / 120
    dehaze_amount = altitude**2 / 600
    print(f'dehaze_amount={dehaze_amount}')
    final_img = dehaze_from_original_example(img, dehaze_amount)
    stepTimer.stop_and_disp('Dehaze')

    # final_img = img

    # Check if photoenhance was cancelled by some top-level process
    if batch_monitor.cancelled:
        return

    # 5. ADDING BRIGHTNESS
    stepTimer.start()
    print(f'beta={getBeta(altitude)}')
    final_img = add_brightness(final_img, getBeta(altitude))
    stepTimer.stop_and_disp('Add Brightness')

    # 6. POST-PROCESSING: CUSTOM LAB-SPACE HISTEQ
    if altitude < 6:
        postproc_clip_limit = 0.0125 * altitude**2
    else:
        if enhancer_params.enable_stronger_contrast_deep:
            print('Using stronger contrasting')
            postproc_clip_limit = 0.03 * altitude + 0.3
            multiplier = 0.9 + altitude / 50
            postproc_clip_limit = np.minimum(postproc_clip_limit*(multiplier**4), 1.5)
        else:
            postproc_clip_limit = 0.45

    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=postproc_clip_limit, tileGridSize=(32,32))
    final_img[:,:,0] = clahe.apply(final_img[:,:,0])
    final_img = cv2.cvtColor(final_img, cv2.COLOR_Lab2RGB)

    # convert to rgb image, save with exif data from PILImage
    output_img_data = Image.fromarray(final_img)
    output_img_data.save(output_imgpath, quality=100, exif=PILImage.info['exif'])
    
    overallTimer.stop_and_disp('OVERALL')

def processImage_batch(batch, enhancer_params=EnhancerParameters(), batch_monitor=BatchMonitor()):
    stepTimer = Timer()
    start = time.process_time()
    for img in batch:
        if not batch_monitor.cancelled:
            processImage(img[0], img[1], enhancer_params=enhancer_params, batch_monitor=batch_monitor, stepTimer=stepTimer)
            batch_monitor.tick()
    stepTimer.print_labels()
    print(f'Time elapsed: {(time.process_time() - start) / batch_monitor.n_batches}')


def processImage_multi(inputs, cpu_load=0.9, enhancer_params=EnhancerParameters(), batch_monitor=BatchMonitor()):
    len_inputs = len(inputs)
    batch_monitor.set_total_images(len_inputs)
    print(f'len_inputs={len_inputs}')

    # assign batches based on load
    n_batches = np.minimum(len_inputs, int(cpu_load * multiprocessing.cpu_count())-2)
    n_batches = np.maximum(1, n_batches)
    print(f'n_batches={n_batches}')

    # Divide batches evenly
    batch_size = len_inputs // n_batches
    excess = len_inputs % n_batches
    cursor = 0
    batches = []
    for i in range(n_batches):
        start = cursor
        # end = (i + 1) * batch_size if i != n_batches - 1 else len_inputs
        if excess > 0:
            end = start + batch_size + 1
            excess = excess - 1
        else:
            end = start + batch_size if (start + batch_size < len_inputs) else len_inputs
        cursor = end
        batch = inputs[start:end]
        print(f'Batch = inputs[{start}:{end}]')
        batches.append(batch)

    return Parallel(n_jobs=n_batches)(
        delayed(processImage_batch)(batch, enhancer_params, batch_monitor) for batch in batches
    )

def processImage_multi_2(inputs, cpu_load=0.9, enhancer_params=EnhancerParameters(), batch_monitor=BatchMonitor()):


    threads = []

    len_inputs = len(inputs)
    batch_monitor.set_total_images(len_inputs)
    batch_monitor.initialise_tick()
    print(f'len_inputs={len_inputs}')

    # assign batches based on load
    n_batches = np.minimum(len_inputs, int(cpu_load * multiprocessing.cpu_count())-2)
    n_batches = np.maximum(1, n_batches)
    print(f'n_batches={n_batches}')

    batch_monitor.n_batches = n_batches

    # Divide batches evenly
    batch_size = len_inputs // n_batches
    excess = len_inputs % n_batches
    cursor = 0
    batches = []
    for i in range(n_batches):
        start = cursor
        # end = (i + 1) * batch_size if i != n_batches - 1 else len_inputs
        if excess > 0:
            end = start + batch_size + 1
            excess = excess - 1
        else:
            end = start + batch_size if (start + batch_size < len_inputs) else len_inputs
        cursor = end
        batch = inputs[start:end]
        print(f'Batch = inputs[{start}:{end}]')
        batches.append(batch)

    # return Parallel(n_jobs=n_batches)(
    #     delayed(processImage_batch)(batch, enhancer_params, batch_monitor) for batch in batches
    # )

    for batch in batches:
        thread = Thread(target=processImage_batch, args=(batch, enhancer_params, batch_monitor))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    batch_monitor.finished = True

def isImageFile(name):
    ext = name[get_index_of_extension(name)+1:]
    valid_ext = ['jpg','jpeg','png','bmp']
    print(ext.lower())
    return ext.lower() in valid_ext

def str2bool(mystr):
    return False if mystr.lower() in ['0','false'] else True

def photoenhance(target='', output_folder='enhanced', time_profiling='False', load=0.9, use_suffix='False', suffix='', stronger_contrast_deep='True', disable_denoising='False', batch_monitor=None):
    enhancer_params = EnhancerParameters()

    output_path = os.path.join(target, output_folder)
    os.makedirs(output_path, exist_ok=True)


    if str2bool(stronger_contrast_deep):
        enhancer_params.enable_stronger_contrast_deep = True

    if str2bool(disable_denoising):
        enhancer_params.denoising = False

    useSuffix = True if str2bool(use_suffix) else False

    if not str2bool(time_profiling):
        overallTimer.disable()

    if target:
        imgdir=target
        inputs = []
        for imgfile in os.listdir(imgdir):
                input_imgpath = os.path.join(imgdir, imgfile)
                if os.path.isfile(input_imgpath) and isImageFile(input_imgpath):

                    output_suffix = '_enh.jpg' if not suffix else f'_{suffix}.jpg'

                    output_imgfile = imgfile
                    if useSuffix:
                        output_imgfile = get_filename_noext(imgfile) + output_suffix
                    
                    # output_imgpath = get_filename_noext(input_imgpath) + output_suffix
                    
                    output_imgpath = os.path.join(output_path, output_imgfile)

                    # if not os.path.exists(output_imgpath):
                    if output_suffix not in input_imgpath:
                        print(f'\nProcessing {input_imgpath} -> {output_imgpath}')
                        # processImage(input_imgpath, output_imgpath)
                        inputs.append([input_imgpath, output_imgpath])

        mybatchmonitor = BatchMonitor()

        def printfunc(completed, total):
            print(f'Progress: {completed} / {total}')

        mybatchmonitor.set_callback_on_tick(printfunc)

        monitor = batch_monitor if batch_monitor is not None else mybatchmonitor

        processImage_multi_2(inputs, cpu_load=float(load), 
                           enhancer_params=enhancer_params,
                           batch_monitor=monitor)

    else:
        print('INVALID TARGET: Directory does not exist')

def main(**kwargs):
    for key, value in kwargs.items():
        print(f'Arg: {key}={value}')
    photoenhance(**kwargs)


if __name__=='__main__':
    sysargs = [arg.replace("--","") for arg in sys.argv[1:]]
    if 'target' not in sysargs[0]:
        sysargs[0] = f'target={sysargs[0]}'

    main(**dict(arg.split('=') if '=' in arg else arg for arg in sysargs))
