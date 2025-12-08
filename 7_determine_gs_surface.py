#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import nibabel as nib
import nrrd
import numpy as np


main_path = r'MAIN PATH'
gs_path = r'GS/RS PATH'
    
pixelsize_dict = {'Volume 67':0.7109, 'Volume 83':0.7421875,'Volume 67_2':0.7109, 'Volume 83_2':0.7421875,  'Volume 84':0.7578125, 'Volume 86':0.6835938, 'Volume 87':0.854, 'Volume 89':0.6370, 'Volume 131':1.3672, 'Volume 10':0.7695312}

volume_order_dict = {10:7, 67:2, 83:4, 84:5, 86:3, 87:6, 89:1, 131:8}

# %%
surface_df = pd.DataFrame()
for dir in os.listdir(gs_path):
    volume_path = os.path.join(gs_path, dir)
    volume = dir
    for file in os.listdir(volume_path):
        if file.endswith('.seg.nrrd'):
            muscle = file.split(' ')[1].split('.')[0]
            segmentation_path = os.path.join(volume_path, file)
            segmentation, _ = nrrd.read(segmentation_path)
            segmentation = np.sum(segmentation)
            if volume in pixelsize_dict.keys():
                pixelsize = pixelsize_dict[volume]
            else:
                pixelsize = 1
            pixel_surface = pixelsize**2
            surface = segmentation * pixel_surface
            input_dict = {'Volume':volume, 'Muscle':muscle, 'Surface':surface}
            surface_df = pd.concat([surface_df, pd.DataFrame(input_dict, index=[0])])

surface_df.to_csv(os.path.join(main_path, 'surface_df.csv'), index=False)
# %%
