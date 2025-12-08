#%%
'''
Docstring for 4_screen_combined_dfs
Code used to screen the combined dataframe to check if all observers have 50 segmentations
'''
import pandas as pd
import os
from unidecode import unidecode

main_path = r'MAIN PATH'

combined_df = pd.read_csv(os.path.join(main_path, 'final_df_v4.csv'))

#%%
#dataframe containing number of segmentations per observer derived from combined_df to check if all observers have 50 segmentations
observers_df = pd.DataFrame()
observers_df['Observer'] = combined_df['Observer'].unique()
observers_df['Number of segmentations'] = [len(combined_df[combined_df['Observer'] == observer]) for observer in observers_df['Observer']]
observers_df[observers_df['Number of segmentations'] != 50]


# %%
