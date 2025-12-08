#%%
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import nrrd

input_gs = r'RF Segmentations 83'
ct_input = r'CT Images 83'

stack_list = []
idx_not_zero = False

for i, seg in enumerate(os.listdir(input_gs)):
    seg_path = os.path.join(input_gs, seg)
    seg_data, seg_header = nrrd.read(seg_path)
    if not idx_not_zero:
        for j in range(seg_data.shape[2]):
            if np.sum(seg_data[:,:,j])>0:
                idx_not_zero = j
                break
        
    seg_data = seg_data.astype(np.uint8)
    seg_data = np.sum(seg_data, axis=2)
    seg_data = np.rot90(seg_data,3)
    seg_data = seg_data*(i+1)
    stack_list.append(seg_data)

stack = np.stack(stack_list, axis=2)
stack = np.sum(stack, axis=2)
stack = np.where(stack>0, stack, np.nan)

ct_slice = np.rot90(nib.load(ct_input).get_fdata()[:,:,idx_not_zero],3)
#%%
plt.figure(figsize=(20,20))
plt.imshow(ct_slice, cmap='gray')
#plt.imshow(stack, cmap='Pastel2', alpha=0.3)
plt.axis('off')
plt.show()
# %%
