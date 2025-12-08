#%%
import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import pandas as pd
import tqdm

from scipy.spatial import distance

main_path = r'MAIN PATH'
gs_folder = r'Reference standard PATH'
#%%
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

#Muscle name dictionary used to rename muscles to match GS/RS filenames
muscle_dict = {'Psoas':'1 Psoas',
               'Quadratus':'2 Quadratus',
               'Erector': '3 Erector',
               'Zijspieren': '4 Zijspieren',
               'Rectus': '5 Rectus',}

#Invert DICE function, as scipy returns non-overlapping area.
def dice_coefficient(y_true, y_pred):
    return 1 - distance.dice(y_true.flatten(), y_pred.flatten())

#%%
#Build a registration dict from scratch, iterating over all observers, volumes and muscles.
#The dict will have the following structure:
#registration_dict = {volume: {muscle: Registration object}}
#The Registration object will contain the stack of all slices for that muscle and volume, as well
#as the observer keys.

analyse_path = os.path.join(main_path, 'observers')
registration_dict = {}
for dir in os.listdir(analyse_path):
    observer = dir
    print('Starting with observer:', observer)
    for sub_dir in os.listdir(os.path.join(analyse_path, dir)):
        volume = sub_dir
        for file in os.listdir(os.path.join(analyse_path, dir, sub_dir)):
            if file.endswith('.seg.nrrd'):
                try:
                    muscle = file.split(' ')[1].split('.')[0]
                except:
                    continue
                file_path = os.path.join(analyse_path, dir, sub_dir, file)
                #Check if file_path is a list
                if isinstance(file_path, list):
                    file_path = file_path[0]
                img = nrrd.read(file_path)
                img = np.rot90(img, 3)
                img = img.sum(axis=2)
                try:
                    if volume not in registration_dict:
                        registration_dict[volume] = {muscle: Registration(muscle, img, observer)}
                        
                    elif muscle not in registration_dict[volume]:
                        registration_dict[volume][muscle] = Registration(muscle, img, observer)
                        
                    else:
                        registration_dict[volume][muscle].add_slice(img, observer)
                except:
                    img = np.zeros((512, 512), dtype=np.uint32)
                    registration_dict[volume][muscle].add_slice(img, observer)

                    print('Error with observer:', observer, 'volume:', volume, 'muscle:', muscle)
                    continue


#Save registration dict to pickle
with open('registration_dict_v4.pkl', 'wb') as f:
    pickle.dump(registration_dict, f)

#%%
#Open registration dict from pickle
pkl_path = os.path.join(main_path, 'registration_dict_v4.pkl')
with open(pkl_path, 'rb') as f:
    registration_dict = pickle.load(f)


#Analyse data using DICE coefficient per muscle, volume and observer
# Create dataframe
analysis_df = pd.DataFrame(columns=['Volume', 'Muscle', 'Observer', 'DICE'])

# Iterate over registration dict
for volume in registration_dict:
    for muscle in registration_dict[volume]:
        mean_stack = registration_dict[volume][muscle].average_stack()
        muscle_stack = registration_dict[volume][muscle].stack

        #GS Trigger
        gs_volume = volume
        if volume.split('_')[-1] == '2':
            gs_volume = volume.split('_')[0]
        muscle_filename = f'{muscle_dict[muscle.split("_")[0]]}.seg.nrrd'
        gs = nrrd.read(os.path.join(gs_folder, gs_volume, muscle_filename))[0]
        gs = np.rot90(gs, 3)
        gs = gs.sum(axis=2)

        for i in range(muscle_stack.shape[2]):
            i_slice = muscle_stack[:, :, i]
            observer = registration_dict[volume][muscle].observer_keys[i]
            dice_coeff = dice_coefficient(gs, i_slice)
            haussdorf = distance.directed_hausdorff(gs, i_slice)[0]

            input_df = pd.DataFrame(data={'Volume': volume,
                                        'Muscle': muscle,
                                        'Observer': observer,
                                        'Observer idx': int(i),
                                        'Dice': dice_coeff,
                                        'Hausdorff': haussdorf},
                                    index=[0])
            

            analysis_df = pd.concat([analysis_df, input_df], ignore_index=True)

muscle_dummie_dict = {k:i for i, k in enumerate(analysis_df['Muscle'].unique())}
 
analysis_df['Muscle idx'] = analysis_df['Muscle'].apply(lambda x: muscle_dummie_dict[x])

analysis_df.to_csv(os.path.join(main_path,'analysis_df.csv'), index=False)
