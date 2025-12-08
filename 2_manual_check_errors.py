#%%
'''
Docstring for 2_manual_check_errors
This script is used to manually check for errors in the segmentations
by visualizing the scans and segmentations for specific cases where
there are known issues.
'''

import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib

scans = r'SCAN PATH'
observers_folder = r'OBSERVER PATH'

#%%
#China, 10, rectus, gb in overzicht participanten
#Finland, 89, rectus,gb in overzicht participanten

scan = nib.load(os.path.join(scans,'volume-10.nii.gz'))
seg = nrrd.read(os.path.join(observers_folder,'China','Volume 10','5 Rectus.seg.nrrd'))
seg = nrrd.read(os.path.join(observers_folder,'Finland','Volume 89','5 Rectus.seg.nrrd'))

#Geen rectus segmentatie?
# %%
#Check DICE == 0
#België en India

input_dict ={'België': ['Volume 131','volume-131', '1 Psoas'],
             'India': ['Volume 87','volume-87',  '2 Quadratus'],
             'Colombia': ['Volume 10','volume-10', '5 Rectus'],}

for country, [seg_name, scan_name, muscle] in input_dict.items():
    scan = nib.load(os.path.join(scans,f'{scan_name}.nii.gz'))
    seg = nrrd.read(os.path.join(observers_folder,country,seg_name,f'{muscle}.seg.nrrd'))[0]

    trigger = 0
    for i in range(seg.shape[2]):
        if np.sum(seg[:,:,i]) != 0:
            plt.imshow(scan.get_fdata()[:,:,i], cmap='gray')
            plt.imshow(seg[:,:,i], alpha=0.5)
            plt.title(f'{country} - {muscle} - {i}')
            plt.show()
            trigger = 1
    if trigger == 0:
        print(f'{country} heeft geen segmentatie voor {muscle}')

#I quadratus error
#C rectus to large, 
#B no psoas?
# %%
#Load B segmentation - to check
scan = nib.load(os.path.join(scans,'volume-131.nii.gz'))
seg = nrrd.read(os.path.join(observers_folder,'België','Volume 131','1 Psoas.seg.nrrd'))[0]

#%%
# volume_order_dict = {10:7, 67:2, 83:4, 84:5, 86:3, 87:6, 89:1, 131:8}
#F 89 Rectus
#I 87 Quadratus
#C 10 Rectus

input_dict = {'F': ['Volume 89','volume-89', '5 Rectus'],
              'I': ['Volume 87','volume-87',  '2 Quadratus'],
              'C': ['Volume 10','volume-10', '5 Rectus']}

input_dict = {
              'I': ['Volume 87','volume-87',  '2 Quadratus'],
}

slice_dict = {'Volume 89': 323,
              'Volume 87': 381,
              'Volume 10': 279}

for country, [seg_name, scan_name, muscle] in input_dict.items():

    scan = nib.load(os.path.join(scans,f'{scan_name}.nii.gz'))
    for file in os.listdir(os.path.join(observers_folder,country,seg_name)):
        if file.endswith('.seg.nrrd'):
            seg = nrrd.read(os.path.join(observers_folder,country,seg_name,file))[0]
            seg = np.sum(seg, axis=2)
            seg = np.where(seg == 1, 1, np.nan)
            
        slice_idx = slice_dict.get(seg_name, 0)
        plt.imshow(scan.get_fdata()[:,:,slice_idx], cmap='gray')
        try:
            plt.imshow(seg, alpha=0.5)
        except:
            pass
        plt.title(f'{country} - {file} - {slice_idx}')
        #Plt axes off
        plt.axis('off')
        plt.show()
        
    
# %%
