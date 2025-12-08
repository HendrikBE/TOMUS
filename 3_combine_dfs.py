#%%
'''
Combine observer data with participant overview data.
Mening (opinion) data is not used in the final version of this analysis, but kept for potential future use.
'''
import pandas as pd
import os
from unidecode import unidecode


main_path = r'MAIN PATH'

df1_path = os.path.join(main_path, 'analysis_df.csv')
df2_path = os.path.join(main_path, 'Overzicht participanten.xlsx')

df_1 = pd.read_csv(df1_path)
df_2 = pd.read_excel(df2_path)



df_2['Proefpersoon '] = df_2['Proefpersoon '].apply(unidecode)
df_2['Proefpersoon '] = df_2['Proefpersoon '].apply(lambda x: x.replace(' ', ''))
df_1['Observer'] = df_1['Observer'].apply(unidecode)
df_1['Observer'] = df_1['Observer'].apply(lambda x: x.replace(' ', ''))
column_translate_dict = {'Mening slice 10': 'Volume 10', 
                         'Mening slice 67': 'Volume 67',
                        'Mening slice 83': 'Volume 83',
                        'Mening slice 84': 'Volume 84', 
                        'Mening slice 86': 'Volume 86',
                        'Mening slice 87': 'Volume 87', 
                        'Mening slice 89': 'Volume 89', 
                        'Mening slice 83-2': 'Volume 83_2',
                        'Mening slice 131': 'Volume 131', 
                        'Mening slice 67-2': 'Volume 67_2'}

#Rename columns in df_2
df_2.rename(columns=column_translate_dict, inplace=True)

df_2 = df_2.dropna(subset=['Volume 10'])
#Gender binary

df_2['Geslacht '] = df_2['Geslacht '].apply(lambda x: 1 if x == 'man' else 0)
df_2['MBRT gediplomeerd '] = df_2['MBRT gediplomeerd '].apply(lambda x: 1 if x == 'ja' else 0)

#df_2 replace nan and - with x
df_2 = df_2.replace('-', 'x')
df_2 = df_2.fillna('x')
def transfer_data(row):
    volume = row['Volume']
    observer = row['Observer']

    df2_row = df_2.loc[df_2['Proefpersoon '] == observer]


    try:    
        gender = df2_row['Geslacht '].iloc[0]
        opinion = df2_row[volume].iloc[0]
        diploma = df2_row['MBRT gediplomeerd '].iloc[0]
        experience = df2_row['Werkjaren/opleidingsjaren '].iloc[0]
        rt_experience = df2_row['Jaren ervaring radiotherapie '].iloc[0]
        ct_experience = df2_row['Jaren ervaring CT'].iloc[0]
        expert_group = df2_row['Expertise groep'].iloc[0]
        experience_binary = df2_row['Ervaring'].iloc[0]
    except:
        return print('Observer {} not found'.format(observer))

    return gender, opinion, diploma, experience, rt_experience, ct_experience, expert_group, experience_binary

df_1['Gender'], df_1['Opinion'], df_1['Diploma'], df_1['Experience'], df_1['RT Experience'], df_1['CT Experience'], df_1['Expertise group'], df_1['Experience binary'] = zip(*df_1.apply(lambda x: transfer_data(x), axis=1))
#Create new columns in df_1
#df_final = df_1[df_1['Threshold']==0.7].drop(columns=['Threshold'])

#%%
#Insert C and F data: [C, 10, rectus, dice=0] and [F, 89, rectus, dice=0]
#China
input_dict = {'Observer': 'C', 'Volume': 'Volume 10', 'Muscle': 'Rectus', 'Dice':0}
df_1 = pd.concat([df_1, pd.DataFrame(input_dict, index=[0])], ignore_index=True)
#Finland
input_dict = {'Observer': 'F', 'Volume': 'Volume 89', 'Muscle': 'Rectus', 'Dice':0}
df_1 = pd.concat([df_1, pd.DataFrame(input_dict, index=[0])], ignore_index=True)
df_final = df_1
df_final.to_csv(os.path.join(main_path, 'final_df_v4.csv'), index=False)




