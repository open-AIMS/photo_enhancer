## photo_enhancer

Python program for enhancing images based on a reference MATLAB code

## Usage

```
cd photoenhancer
```
```
python photoenhance.py target=<your directory of images>
```

### Additional arguments
| argument          | definition | example |
|-------------------|------------|---------|
| replace_existing  | Replace the existing output enhanced folder. <br/> <br/> Possible values: ["True", "False" *(default)*]            | replace_existing="True"         |
| output_folder  | Customize the name of the output folder. | output_folder="enhanced_new"         |
| disable_denoising | Disable denoising step for faster performance: <br/> <br/> Possible values: ["True", "False"*(default)*]            |  disable_denoising="True"       |
| disable_dehazing  | Disable dehazing step for faster performance: <br/> <br/> Possible values: ["True", "False"*(default)*]            |  disable_dehazing="True"       |
| stronger_contrast_deep  | Add another post-processing step that performs stronger contrast. Generally used for deeper images   <br/> <br/> Possible values: ["True"*(default)*, "False"]       |  stronger_contrast_deep="True"       |
| use_suffix  |   Tags the output images with an extra suffix. If not used, then the default suffix is **_enh**. <br/> <br/> Possible values: ["True", "False"*(default)*] <br/> <br/> **NOTE: If you are using a suffix, it is should be followed up by `suffix="{your own suffix}"`. See example for when suffix is "abc".**         |  use_suffix="True" suffix="abc"       | 