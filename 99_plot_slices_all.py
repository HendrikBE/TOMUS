#%%
import os
import matplotlib.pyplot as plt
from skimage import io
input_gs = r'MAIN PAT GS/RS'

volume_slice_dict = {10:280, 67:103, 83:394, 84:400, 86:400, 87:380, 89:323, 131:412}
volume_order_dict =  {67:1, 83:2, 89:3, 86:4, 87:5, 84:6, 10:7, 131:8}

slice_dict = {}

for volume in volume_slice_dict.keys():
    print(volume)
    slice_idx = volume_slice_dict[volume]
    position = volume_order_dict[volume]
    scan_stack = io.imread(os.path.join(input_gs, f'volume-{volume}.tif'))
    slice_dict[position] = scan_stack[slice_idx,:,:]

#%%
fig, ax = plt.subplots(2, 4, figsize=(40, 20))
#increase font size
plt.rcParams.update({'font.size': 35})
for i in range(8):
    ax[i//4, i%4].imshow(slice_dict[i+1], cmap='gray')
    ax[i//4, i%4].set_title(f'CT Org volume {list(volume_order_dict.keys())[list(volume_order_dict.values()).index(i+1)]}\n Study slice ID {i+1}')
for a in ax.flat:
    a.axis('off')
plt.rcParams.update({'font.size': 20})
plt.show()

# %%
#Snip for focus on differentiation of muscles
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
for i in range(3):
    im_idx = {0:1, 1:5, 2:8}[i]
    ax[i].imshow(slice_dict[im_idx], cmap='gray')
    ax[i].set_title(f'Volume {list(volume_order_dict.keys())[list(volume_order_dict.values()).index(im_idx)]}')
for a in ax.flat:
    a.axis('off')
plt.show()

# %%
