#%%
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import pandas as pd
import pickle
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import nrrd
from unidecode import unidecode

main_path = r'MAIN PATH' 

input_gs = os.path.join(main_path, r'GS\GS_complete\Volume 83')
#Originele 83
ct_input = os.path.join(main_path, r'included\volume-83.nii.gz')


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
window_low = -150
window_high = 250
y_1, y_2, x1, x2 = 90, 370, 45, 512-75
stack = np.where(stack!=9, stack, np.nan)
ct_slice_crop = np.fliplr(ct_slice[y_1:y_2, x1:x2])
stack_crop = np.fliplr(stack[y_1:y_2, x1:x2])
plt.figure(figsize=(20,20))
plt.imshow(ct_slice_crop, cmap='gray', vmin=window_low, vmax=window_high, alpha=0.5)
plt.imshow(stack_crop, cmap='rainbow', alpha=0.6)
plt.axis('off')

plt.show()

class Registration:
    def __init__(self, muscle, img, observer):
        self.muscle = muscle
        self.stack = np.expand_dims(img, axis=2)
        self.observer_keys = {0: observer}

    def add_slice(self, img, observer):
        self.observer_keys[self.stack.shape[2]] = observer
        self.stack = np.dstack((self.stack, img))
    
    def average_stack(self):
        self.mean_stack = self.stack.mean(axis=2)
        return self.mean_stack

# %%


#Compare 84 and 86 for expertise groups
ct_input_84 = os.path.join(main_path, r'included\volume-84.nii.gz')
ct_input_86 = os.path.join(main_path, r'included\volume-86.nii.gz')

ct_input_84 = nib.load(ct_input_84).get_fdata()
ct_input_86 = nib.load(ct_input_86).get_fdata()
slice_84 = np.rot90(ct_input_84[:,:,400],3)
slice_86 = np.rot90(ct_input_86[:,:,400],3)

slice_dict = {'Volume 84': slice_84, 'Volume 86': slice_86}

#Open registration dict from pickle
pkl_path = os.path.join(main_path, 'registration_dict_v4.pkl')
with open(pkl_path, 'rb') as f:
    registration_dict = pickle.load(f)

combined_df = pd.read_csv(os.path.join(main_path, 'final_df_v4.csv'))
#%%
#List observers and Experience binary
observers = combined_df['Observer'].unique()
obs_exp_dict = {}
for obs in observers:
    obs_exp = combined_df[combined_df['Observer']==obs]['Experience binary'].unique()
    obs_exp_dict[obs] = obs_exp[0]

segmentation_dict = {}

for slice_n in slice_dict.keys():

    volume = registration_dict[slice_n]
    volume_segmentation_dict = {}
    for muscle in volume:
        muscle_obj = volume[muscle]
        observer_keys = muscle_obj.observer_keys
        muscle_stack = muscle_obj.stack
        exp_stack_0 = None
        exp_stack_1 = None
        for idx in range(muscle_stack.shape[2]):
            segmentation_slice = muscle_stack[:,:,idx]
            observer = unidecode(observer_keys[idx]).replace(' ', '')
            expertise = obs_exp_dict[observer]

            if expertise == 0:
                if exp_stack_0 is None:
                    exp_stack_0 = segmentation_slice
                else:
                    exp_stack_0 = np.dstack((exp_stack_0, segmentation_slice))
            else:
                if exp_stack_1 is None:
                    exp_stack_1 = segmentation_slice
                else:
                    exp_stack_1 = np.dstack((exp_stack_1, segmentation_slice))
            volume_segmentation_dict[muscle] = (exp_stack_0, exp_stack_1)
    segmentation_dict[slice_n] = volume_segmentation_dict   
#%%
for volume_key in segmentation_dict.keys():
    vol_exp0_stack = None
    vol_exp1_stack = None
    for muscle in segmentation_dict[volume_key].keys():
        exp_stack_0, exp_stack_1 = segmentation_dict[volume_key][muscle]
        if vol_exp0_stack is None:
            vol_exp0_stack = exp_stack_0
        else:
            vol_exp0_stack = np.add(vol_exp0_stack, exp_stack_0)
        if vol_exp1_stack is None:
            vol_exp1_stack = exp_stack_1
        else:
            vol_exp1_stack = np.add(vol_exp1_stack, exp_stack_1)
    
    vol_exp0_stack = np.mean(vol_exp0_stack, axis=2)
    vol_exp1_stack = np.mean(vol_exp1_stack, axis=2)

    #normalise stacks between 0 and 1
    vol_exp0_stack = (vol_exp0_stack - np.min(vol_exp0_stack))/(np.max(vol_exp0_stack) - np.min(vol_exp0_stack))
    vol_exp1_stack = (vol_exp1_stack - np.min(vol_exp1_stack))/(np.max(vol_exp1_stack) - np.min(vol_exp1_stack))

    vol_exp0_stack = np.where(vol_exp0_stack>0, vol_exp0_stack, np.nan)
    vol_exp1_stack = np.where(vol_exp1_stack>0, vol_exp1_stack, np.nan)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(slice_dict[volume_key], cmap='gray')
    ax[0].imshow(vol_exp0_stack, cmap='rainbow')
    ax[0].set_title(f'Volume {volume_key} Expertise 0')
    ax[1].imshow(slice_dict[volume_key], cmap='gray')
    ax[1].imshow(vol_exp1_stack, cmap='rainbow')
    ax[1].set_title(f'Volume {volume_key} Expertise 1')
    for a in ax.flat:
        a.axis('off')
    plt.show()

        
# %%
#list observers and and first and second moment of the muscle for each observer and scan 83
ct_input_83 = os.path.join(main_path, r'included\volume-83.nii.gz')
ct_input_83 = nib.load(ct_input_83).get_fdata()
slice_83 = np.rot90(ct_input_83[:,:,394],3)

slice_dict = {'Volume 83': slice_83}

#Open registration dict from pickle
pkl_path = os.path.join(main_path, 'registration_dict_v4.pkl')
with open(pkl_path, 'rb') as f:
    registration_dict = pickle.load(f)

slice_83_t1 = registration_dict['Volume 83']
slice_83_t2 = registration_dict['Volume 83_2']

muscle_stack_t1 = None
muscle_stack_t2 = None
for t, reg_dict in {'t1':slice_83_t1, 't2':slice_83_t2}.items():
    muscle_stack = None
    for muscle in reg_dict:
        muscle_obj = reg_dict[muscle]
        if muscle_stack is None:
            muscle_stack = muscle_obj.stack
        else:
            muscle_stack = np.dstack((muscle_stack, muscle_obj.stack))
    if t == 't1':
        muscle_stack_t1 = muscle_stack
    else:
        muscle_stack_t2 = muscle_stack

def process_stack(stack):
    mean_stack = np.mean(stack, axis=2)
    norm_stack = (mean_stack - np.min(mean_stack))/(np.max(mean_stack) - np.min(mean_stack))
    return norm_stack

norm_stack_t1 = process_stack(muscle_stack_t1)
norm_stack_t2 = process_stack(muscle_stack_t2)

norm_stack_t1 = np.where(norm_stack_t1>0, norm_stack_t1, np.nan)
norm_stack_t2 = np.where(norm_stack_t2>0, norm_stack_t2, np.nan)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(slice_83, cmap='gray')
ax[0].imshow(norm_stack_t1, cmap='rainbow')
ax[0].set_title('Slice id 4 first moment')
ax[1].imshow(slice_83, cmap='gray')
ax[1].imshow(norm_stack_t2, cmap='rainbow')
ax[1].set_title('Slice id 4 second moment')
for a in ax.flat:
    a.axis('off')
#colorbar
plt.tight_layout()
plt.colorbar()
plt.show()
            
# %%
#Plot all abdo muscles for each observer and all scans
#Open registration dict from pickle
pkl_path = os.path.join(main_path, 'registration_dict_v4.pkl')
with open(pkl_path, 'rb') as f:
    registration_dict = pickle.load(f)

volume_include_list =[131, 83]

for volume_key in registration_dict.keys():
    print(f'****************{volume_key}****************')
    stack = registration_dict[volume_key]['Zijspieren'].stack
    mean_stack = np.mean(stack, axis=2)
    norm_stack = (mean_stack - np.min(mean_stack))/(np.max(mean_stack) - np.min(mean_stack))
    norm_stack = np.where(norm_stack>0, norm_stack, np.nan)
    plt.imshow(norm_stack, cmap='rainbow')
    plt.title(f'{volume_key} Zijspieren')
    plt.axis('off')
    plt.show()
    
    slice_id = volume_key.split(' ')[-1]

    if int(slice_id) not in volume_include_list:
        continue

    elif int(slice_id) in volume_include_list:

        for id in range(stack.shape[2]):
            
            slice = stack[:,:,id]
            slice = np.where(slice>0, slice, np.nan)
            plt.imshow(slice, cmap='rainbow')
            plt.title(f'{volume_key} slice {id}')
            plt.axis('off')
            plt.show()
    
# %%

from matplotlib.colors import ListedColormap
#Open registration dict from pickle
pkl_path = os.path.join(main_path, 'registration_dict_v4.pkl')
with open(pkl_path, 'rb') as f:
    registration_dict = pickle.load(f)

volume_list = {'volume-10.nii.gz':280,
                'volume-131.nii.gz':412,
                'volume-67.nii.gz':103,
                'volume-83.nii.gz':394,
                'volume-84.nii.gz':400,
                'volume-86.nii.gz':400,
                'volume-87.nii.gz':380,
                'volume-89.nii.gz':323}
#Used to measure intensities at specific coordinates to determine n observers deviating
measurement_dict = {'10':[(130,200),(370,212)],
                    '131':[(60, 50), (35, 158)],
                    '67':[(355, 260)],
                    '83':[(215, 145), (265, 145), (275, 230)],
                    '84':[(142, 343),(130, 300)],
                    '86':[],
                    '87':[(145, 125)], #Minimalal deviation
                    '89':[(270, 395)] #Much exlusion
                    }

break_idx = 10
measurement_df = pd.DataFrame(columns=['Volume', 'Coordinates'])
n_observers = 0

volume_list = {'volume-83.nii.gz':394,}

for volume in volume_list.keys(): 

    complete_stack = None
    slice_idx = volume_list[volume]

    volume_n = volume.split('-')[1].split('.')[0]

    ct_input = os.path.join(main_path, f'included\\{volume}')
    ct_input = nib.load(ct_input).get_fdata()
    if volume_n != '10':
        ct_slice = np.rot90(ct_input[:,:,slice_idx],3)
    else:
        ct_slice = np.rot90(ct_input[:,:,slice_idx],1)

    segmentation_slice = registration_dict[f'Volume {volume_n}']

    #muscle_stack = None
    for muscle in segmentation_slice:
        muscle_obj = segmentation_slice[muscle]
        temp_stack = muscle_obj.stack
        #Remove i 1, 29 and 36 due to artefacts
        temp_stack = np.delete(temp_stack, [1, 29, 36], axis=2) #remove observers with more experience, these are not students.
        
        if temp_stack.shape[2] > n_observers:
            n_observers = temp_stack.shape[2]
        
        temp_stack = np.sum(temp_stack, axis=2)

        print(temp_stack.max())

        if complete_stack is None:
            complete_stack = temp_stack
        else:
            complete_stack = np.add(complete_stack, temp_stack)

    #Convert complete_stack to %
    complete_stack = (complete_stack / 42) * 100 #42 observers total
    complete_stack = np.where(complete_stack>0, complete_stack, np.nan) #0.02 due to artefacts

    #flip complete stack and slice_83 for better visualisation
    if volume_n != '10':
        ct_slice = np.fliplr(ct_slice)
    complete_stack = np.fliplr(complete_stack)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    window_low = -150
    window_high = 250
    
    if volume_n == '131':
        y_1, y_2, x1, x2 = 210, 350, 130, 380
        ct_slice = ct_slice[y_1:y_2, x1:x2]
        complete_stack = complete_stack[y_1:y_2, x1:x2]

    elif volume_n == '83':
        y_1, y_2, x1, x2 = 90, 370, 60, 512-40
        ct_slice = ct_slice[y_1:y_2, x1:x2]
        complete_stack = complete_stack[y_1:y_2, x1:x2]
    
    plt.figure()
    ax = plt.gca()

    colors = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
    '#8B4513',  # Dark Brown
    '#FFD700',  # Gold
    '#00CED1'   # Teal
    ]


    cmap = ListedColormap(colors)
    cmap = plt.cm.get_cmap('viridis', 15)

    colors_last_10 = cmap(np.linspace(5/14, 1, 11))  # start from index 5 to 14
    cmap = ListedColormap(colors_last_10)

    im = ax.imshow(ct_slice, cmap='gray', vmin=window_low, vmax=window_high, alpha=0.5)
    im = ax.imshow(complete_stack, cmap=cmap, alpha=0.7)
    plt.axis('off')
    #Include white grid lines alpha 0.5 at ticks 50 intervals
    ax.set_xticks(np.arange(0, ct_slice.shape[1], 50))
    ax.set_yticks(np.arange(0, ct_slice.shape[0], 50))

    plot_grid = False

    for coord in measurement_dict[volume_n]:
        y, x = coord
        if plot_grid:
            ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.5)
            #add a dot based on measurement dict
            ax.plot(x, y, marker='x', color='red', markersize=4, alpha=0.6)
        intensity = complete_stack[y, x]
        input_dict = {'Volume': volume_n, 'Coordinates': coord, 'n_incorrect': intensity}
        measurement_df = pd.concat([measurement_df, pd.DataFrame([input_dict])], ignore_index=True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.suptitle(f'Volume {volume_n} Slice {slice_idx}', fontsize=16)
    plt.colorbar(im, cax=cax)
    plt.show()

    break_idx -= 1
    if break_idx == 0:
        break
print(measurement_df)

# %%
