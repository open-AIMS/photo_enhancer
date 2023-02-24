from PIL import Image
from PIL import ExifTags

def getAltitude(pil_img):
    # img = Image.open(imgpath)
    if pil_img._getexif() is None:
        return 0.0

    overall_exifdata = { ExifTags.TAGS[k]: v for k, v in pil_img._getexif().items() if k in ExifTags.TAGS }
    
    # Check if 'GPSInfo' key is in the exif metadata
    if 'GPSInfo' in overall_exifdata.keys():
        gpsdata = { ExifTags.GPSTAGS[k]: v for k, v in overall_exifdata['GPSInfo'].items() if k in ExifTags.GPSTAGS}
        return float(gpsdata['GPSAltitude'])
    else:
        return 0.0
    