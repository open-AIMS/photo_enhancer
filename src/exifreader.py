from PIL import Image
from PIL import ExifTags

import numpy as np

def getAltitude(pil_img):
    # img = Image.open(imgpath)
    if pil_img._getexif() is None:
        return 0.0

    overall_exifdata = { ExifTags.TAGS[k]: v for k, v in pil_img._getexif().items() if k in ExifTags.TAGS }
    
    # Check if 'GPSInfo' key is in the exif metadata
    if 'GPSInfo' in overall_exifdata.keys():
        gpsdata = { ExifTags.GPSTAGS[k]: v for k, v in overall_exifdata['GPSInfo'].items() if k in ExifTags.GPSTAGS}

        # Check if GPSAltitude is in the key
        if 'GPSAltitude' in gpsdata.keys():
            # print(f'Type of gpsdata[GPSAltitude]: {type(gpsdata["GPSAltitude"])}')
            # print(f'Value of gpsdata[GPSAltitude]: {gpsdata["GPSAltitude"]}')

            # According to their docs, exif file metadata allow nan (0/0) values to be put on their 
            # IDFRational values to account for things like digital zoom. Normally if Altitude is not 
            # entered it does not resolve as NaN but a legal 0/0 so a typical 'is' check does not work.
            # Therefore check if denominator is zero for altitude
            if gpsdata['GPSAltitude']._denominator == 0:
                print('Altitude metadata is NaN')
                return 0.0            
            else:   
                return float(gpsdata['GPSAltitude'])
        else:
            return 0.0
    else:
        return 0.0
    