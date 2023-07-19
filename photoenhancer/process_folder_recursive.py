import fnmatch
import os

from photoenhance import processImage_multi

input_folder = "D:/reefcloud-copy/reefscan\photos_from_pearl\surveys\REEFSCAN"
output_folder = "D:/reefcloud-copy/reefscan\enhanced_photos\surveys\REEFSCAN"
inputs=[]
found=0
not_found=0
not_jpg=0
for root, dir, files in os.walk(input_folder):
        for file in files:
                if file.lower().endswith(".jpg"):
                        input = f"{root}/{file}"
                        output=input.replace(input_folder, output_folder)
                        if os.path.isfile(output):
                                found += 1
                                print(f"{output} exists")
                        else:
                                not_found += 1
                                inputs.append([input, output])
                                print(f"{output} does not exists")

                else:
                        print(f"{file} is not a jpg")
                        not_jpg+=1

print(f"not_jpg: {not_jpg}")
print(f"not_found: {not_found}")
print(f"found: {found}")

print(inputs)
processImage_multi(inputs, 0.9)
