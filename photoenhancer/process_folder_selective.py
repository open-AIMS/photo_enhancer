import fnmatch
import os

import pandas as pd

from photoenhance import processImage_multi
from photoenhance import BatchMonitor
from photoenhance import EnhancerParameters

# input_folder = "C:\\Users\\pteneder\\Downloads\\reefscan\\reefscan\\photos_from_pearl\\surveys\\REEFSCAN"
# output_folder = "C:\\Users\\pteneder\\Downloads\\reefscan\\reefscan\\enhanced_photos\\surveys\\REEFSCAN"


input_folder="C:\\Users\\pteneder\\Downloads\\reefscan\\reefscan\\photos_from_pearl"
output_folder="C:\\Users\\pteneder\\Downloads\\reefscan\\reefscan\\enhanced"

refdf = pd.read_csv('C:\\Users\\pteneder\\Downloads\\reefscan\\reefscan\\export-human-classified.csv')

inputs=[]
found=0
not_found=0
not_jpg=0

for index, row in refdf.iterrows():
    file = os.path.join(input_folder, row['image_path'])
    if file.lower().endswith(".jpg"):
        input = file
        output = input.replace(input_folder, output_folder)
        if os.path.isfile(output):
            found += 1
            print(f"{output} exists")
        else:
            not_found += 1
            inputs.append([input, output])
            print(f"{output} does not exist")
            os.makedirs(os.path.dirname(output), exist_ok=True)
    else:
        print(f"{file} is not a jpg")
        not_jpg+=1

# for root, dir, files in os.walk(input_folder):
#         for file in files:
#                 if file.lower().endswith(".jpg"):
#                         input = f"{root}/{file}"
#                         output=input.replace(input_folder, output_folder)
#                         if os.path.isfile(output):
#                                 found += 1
#                                 print(f"{output} exists")
#                         else:
#                                 not_found += 1
#                                 inputs.append([input, output])
#                                 print(f"{output} does not exists")
#                                 os.makedirs(os.path.dirname(output), exist_ok=True)

#                 else:
#                         print(f"{file} is not a jpg")
#                         not_jpg+=1

print(f"not_jpg: {not_jpg}")
print(f"not_found: {not_found}")
print(f"found: {found}")

print(inputs)


mybatchmonitor = BatchMonitor()

def printfunc(completed, total):
    print(f'Progress: {completed} / {total}')

mybatchmonitor.set_callback_on_tick(printfunc)

enhancer_params = EnhancerParameters()
enhancer_params.denoising = False

processImage_multi(inputs, 0.9, enhancer_params=enhancer_params, batch_monitor=mybatchmonitor)
