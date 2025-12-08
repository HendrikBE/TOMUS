#%%
'''
Docstring for 5_DICE_analysis
This script analyses the DICE similarity coefficients calculated in 1_DICE_assessment.py.
It creates violin plots per muscle and volume, calculates median and IQR per muscle and volume,
and performs statistical tests between experience groups.
Observers were given a Pseudonym, this pseudonym is abbreviated in the code for publication.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

main_path = r'MAIN PATH'

combined_df = pd.read_csv(os.path.join(main_path, 'final_df_v4.csv'))
rename_dict = {'Psoas': 'm. psoas major', 
               'Quadratus': 'm. quadratus lumborum', 
               'Erector': 'm. erector spinae', 
               'Zijspieren': 'abdominal wall muscles', 
               'Rectus': 'm. rectus abdominis'}

combined_df['Muscle'] = combined_df['Muscle'].replace(rename_dict)
#combined_df['Muscle'] = combined_df['Muscle'].replace('Zijspieren', 'Abdominal wall muscles')
combined_df_6783 = combined_df[(combined_df['Volume'] == 'Volume 83_2') | (combined_df['Volume'] == 'Volume 67_2') | (combined_df['Volume'] == 'Volume 83') | (combined_df['Volume'] == 'Volume 67')]

#n per group
n_observer = len(combined_df['Observer'].unique())

gender_experience_0 = combined_df[combined_df['Experience binary'] == 0]['Gender'].value_counts()
n_total_experience_0 = len(combined_df[combined_df['Experience binary'] == 0])
gender_experience_1 = combined_df[combined_df['Experience binary'] == 1]['Gender'].value_counts()
n_total_experience_1 = len(combined_df[combined_df['Experience binary'] == 1])

print(f'Female in experience group 0: {gender_experience_0[0]} with a total of {n_total_experience_0} observations')
print(f'Female in experience group 1: {gender_experience_1[0]} with a total of {n_total_experience_1} observations')

missing_rows = (combined_df['Experience binary'] == 0) == (combined_df['Experience binary'] == 1)
print(f'Missing rows: {missing_rows.value_counts()}')

#Pixel sizes derived from Dicom headers.
pixelsize_dict = {'Volume 67':0.7109, 'Volume 83':0.7421875,'Volume 67_2':0.7109, 'Volume 83_2':0.7421875,  'Volume 84':0.7578125, 'Volume 86':0.6835938, 'Volume 87':0.854, 'Volume 89':0.6370, 'Volume 131':1.3672, 'Volume 10':0.7695312}

def calculate_hd_mm(volume, hausdorff):
    return round(hausdorff * pixelsize_dict[volume],2)
combined_df['Hausdorff_mm'] = combined_df.apply(lambda x: calculate_hd_mm(x['Volume'], x['Hausdorff']), axis=1)

surface_df = pd.read_csv(os.path.join(main_path, 'surface_df.csv'))
surface_df['Muscle'] = surface_df['Muscle'].replace('Zijspieren', 'Side muscles')

#Rename Volume row using the dictonary below, sort by volume number and merge volume and second scan into one column
volume_order_dict =  {10:1, 67:2, 83:3, 84:4, 86:5, 87:6, 89:7, 131:8}

combined_df['Volume'] = combined_df['Volume'].apply(lambda x: x.split(' ')[1])
combined_df['Second scan'] = combined_df['Volume'].apply(lambda x: '_2' if x.split('_')[-1] == '2' else '')
combined_df['Volume'] = combined_df['Volume'].apply(lambda x: int(x.split('_')[0]))
combined_df['Volume'] = combined_df['Volume'].apply(lambda x: volume_order_dict[int(x)]) 
#Sort by volume
combined_df = combined_df.sort_values(by='Volume')
#Merge volume and second scan into one column
combined_df['Volume'] = combined_df['Volume'].astype(str) + combined_df['Second scan'].astype(str)
#exclude _2 scans
combined_df['Second scan exclude'] = combined_df['Second scan'].apply(lambda x: 1 if x == '_2' else 0)
combined_df = combined_df[combined_df['Second scan exclude'] == 0]

#Drop B Psoas DICE 0.0
combined_df = combined_df.drop(combined_df[(combined_df['Observer'] == 'B') & (combined_df['Muscle'] == 'Psoas') & (combined_df['Dice'] == 0.0)].index)
#Exclude DICE 0.0
combined_df = combined_df[combined_df['Dice'] >= 0.01]
print(len(combined_df['Observer'].unique()))

#Exclude I, F, A - observers not falling within the expertise group.
combined_df = combined_df[~combined_df['Observer'].isin(['I', 'F', 'A'])]
print(len(combined_df['Observer'].unique()))

#%%
#Mean dice and standard deviation per scan and expert level in a dataframe per volume and muscle
median = combined_df['Dice'].median()
iqr = [combined_df['Dice'].quantile(0.75), combined_df['Dice'].quantile(0.25)]
mean_dice_df = pd.DataFrame()
for volume in combined_df['Volume'].unique():
    df_volume = combined_df[combined_df['Volume'] == volume]
    for muscle in df_volume['Muscle'].unique():
        df_muscle = df_volume[df_volume['Muscle'] == muscle]
        mean_dice = df_muscle['Dice'].median()
        std_dice = df_muscle['Dice'].std()
        iqr_dice = [df_muscle['Dice'].quantile(0.75), df_muscle['Dice'].quantile(0.25)]
        input_dict = {'Volume': volume, 'Muscle': muscle, 'Median DICE': mean_dice, 'Std DICE': std_dice, 'IQR DICE': iqr_dice}
        mean_dice_df = pd.concat([mean_dice_df, pd.DataFrame([input_dict])], ignore_index=True)

#DSC per volume
mean_dice_per_volume = pd.DataFrame()
for volume in combined_df['Volume'].unique():
    df_volume = combined_df[combined_df['Volume'] == volume]
    median_dice = df_volume['Dice'].median()
    std_dice = df_volume['Dice'].std()
    iqr_dice = [df_volume['Dice'].quantile(0.75), df_volume['Dice'].quantile(0.25)]
    input_dict = {'Volume': volume, 'Median DICE': median_dice, 'Std DICE': std_dice, 'IQR DICE': iqr_dice}
    mean_dice_per_volume = pd.concat([mean_dice_per_volume, pd.DataFrame([input_dict])], ignore_index=True)

#Sort mean_dice_per_volume by median DICE
mean_dice_per_volume = mean_dice_per_volume.sort_values(by='Median DICE', ascending=False)
mean_dice_per_volume['IQR_sub'] = mean_dice_per_volume['IQR DICE'].apply(lambda x: x[0] - x[1])

median_dice_per_muscle = pd.DataFrame()
for muscle in combined_df['Muscle'].unique():
    df_muscle = combined_df[combined_df['Muscle'] == muscle]
    median_dice = df_muscle['Dice'].median()
    iqr_dice = [df_muscle['Dice'].quantile(0.75), df_muscle['Dice'].quantile(0.25)]
    input_dict = {'Muscle': muscle, 'Median DICE': median_dice, 'Std DICE': std_dice, 'IQR DICE': iqr_dice}
    median_dice_per_muscle = pd.concat([median_dice_per_muscle, pd.DataFrame([input_dict])], ignore_index=True)
median_dice_per_muscle['IQR_sub'] = median_dice_per_muscle['IQR DICE'].apply(lambda x: x[0] - x[1])

#Translate with rows for volume and columns for muscle include column for iqr
median_dice_pivot = mean_dice_df.pivot(index='Volume', columns='Muscle', values='Median DICE')

iqr_dice_pivot = mean_dice_df.pivot(index='Volume', columns='Muscle', values='IQR DICE').round(3)

#Merge median and iqr pivot tables into one table per muscle
median_dice_pivot.columns = pd.MultiIndex.from_product([['Median DICE'], median_dice_pivot.columns])
iqr_dice_pivot.columns = pd.MultiIndex.from_product([['IQR DICE'], iqr_dice_pivot.columns])
combined_pivot = pd.concat([median_dice_pivot, iqr_dice_pivot], axis=1).round(3)

combined_pivot.to_csv(os.path.join(main_path, 'median_dice_pivot.csv'))

#%%
#Violin plots per muscle for DSC per volume
#Dictionary for plot titles and variable names

plot_dict = {'Dice': 'DICE Similarity index', 'Hausdorff_mm': 'Hausdorff distance (mm)', 'Surface deviation': 'Muscle surface (cm^2)'}

rename_dict = {'m. psoas major': 'psoas major',
               'm. quadratus lumborum': 'quadratus lumborum', 
               'm. erector spinae': 'erector spinae', 
               'abdominal wall muscles': 'abdominal wall muscles', 
               'm. rectus abdominis': 'rectus abdominis'}

#Dice similarity coefficient and Hausdorff distance violin plots per muscle
for choice in ['Dice', 'Hausdorff_mm']:
    plt.rcParams.update({'font.size': 30})
    for muscle in combined_df['Muscle'].unique():
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        sns.violinplot(x='Volume', 
                    y=choice, 
                    data=combined_df[combined_df['Muscle'] == muscle], 
                        hue='Volume', 
                        palette=sns.color_palette("colorblind"), 
                        inner='quartile', 
                        cut=0, 
                        ax=ax)
        #Remove top and right spine
        sns.set_style("whitegrid")
        plt.title(f'{plot_dict[choice]} \n {rename_dict[muscle]}')
        #Increase font size
        plt.ylabel(plot_dict[choice])
        plt.xlabel('Slice ID')
        if choice =='Dice':
            plt.ylim(0.3, 1)
        else:
            plt.ylim(1, 9)
        plt.show()
#%%
#Overall violin plot per volume for all muscles combined
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sns.violinplot(x='Volume', y=choice, data=combined_df, palette=sns.color_palette("colorblind"), inner='quartile', cut=0)
plt.ylabel(plot_dict[choice])
plt.title(f'Overall deviation per volume')
plt.show()

vol_median_df = pd.DataFrame()
for volume in combined_df['Volume'].unique():
    median_vol = combined_df[combined_df['Volume'] == volume][choice].median()
    iqr_vol = [combined_df[combined_df['Volume'] == volume][choice].quantile(0.75), combined_df[combined_df['Volume'] == volume][choice].quantile(0.25)]
    iqr_vol = iqr_vol[0] - iqr_vol[1]
    input_dict = {'Volume': volume, 'Median': median_vol, 'IQR': iqr_vol}
    vol_median_df = pd.concat([vol_median_df, pd.DataFrame([input_dict])], ignore_index=True)

muscle_median_df = pd.DataFrame()
for muscle in combined_df['Muscle'].unique():
    median_muscle = combined_df[combined_df['Muscle'] == muscle][choice].median()
    iqr_muscle = [combined_df[combined_df['Muscle'] == muscle][choice].quantile(0.75), combined_df[combined_df['Muscle'] == muscle][choice].quantile(0.25)]
    iqr_muscle = iqr_muscle[0] - iqr_muscle[1]
    input_dict = {'Muscle': muscle, 'Median': median_muscle, 'IQR': iqr_muscle}
    muscle_median_df  = pd.concat([muscle_median_df , pd.DataFrame([input_dict])], ignore_index=True)



#%%
#df median and IQR per muscle, round to 3 decimals for registration in manuscript
#Print median and IQR overall
#Code sub-optimal due to need to switch between Hausdorff and Dice.
median = combined_df['Hausdorff_mm'].median()
Hausdorff_mm = [combined_df['Hausdorff_mm'].quantile(0.75), combined_df['Hausdorff_mm'].quantile(0.25)]
#Hausdorff_mm = Hausdorff_mm[0] - Hausdorff_mm[1]
print(f'Median Hausdorff: {median} with IQR: {Hausdorff_mm}')
#%%
#Print median and IQR per volume
median_df = pd.DataFrame()
for volume in combined_df['Volume'].unique():
    df_volume = combined_df[combined_df['Volume'] == volume]
    median = df_volume['Hausdorff_mm'].median()
    iqr = [df_volume['Hausdorff_mm'].quantile(0.75), df_volume['Hausdorff_mm'].quantile(0.25)]
    #iqr = iqr[0] - iqr[1]
    input_dict = {'Volume': volume, 'Median': median, 'IQR': iqr}
    median_df = pd.concat([median_df, pd.DataFrame([input_dict])], ignore_index=True)
#%%
median_df = pd.DataFrame()
for muscle in combined_df['Muscle'].unique():
    df_muscle = combined_df[combined_df['Muscle'] == muscle]
    median = df_muscle['Dice'].median()
    iqr = [df_muscle['Dice'].quantile(0.75), df_muscle['Dice'].quantile(0.25)]
    #iqr = iqr[0] - iqr[1]
    input_dict = {'Muscle': muscle, 'Median': median, 'IQR': iqr}
    median_df = pd.concat([median_df, pd.DataFrame([input_dict])], ignore_index=True)
#%%
median_hd = pd.DataFrame()
for volume in combined_df['Volume'].unique():
    for muscle in combined_df['Muscle'].unique():
        df_filter = combined_df[(combined_df['Volume'] == volume) & (combined_df['Muscle'] == muscle)]
        median = df_filter['Hausdorff_mm'].median()
        iqr = [df_filter['Hausdorff_mm'].quantile(0.25), df_filter['Hausdorff_mm'].quantile(0.75)]
        #iqr = iqr[0] - iqr[1]
        input_dict = {'Volume': volume, 'Muscle': muscle, 'Median': median, 'IQR': iqr}
        median_hd = pd.concat([median_hd, pd.DataFrame([input_dict])], ignore_index=True)
#%%
#Test segnificant difference between first and second annotation per muscle

register_df = pd.DataFrame()
#filter_df = combined_df[combined_df['Second scan'] != ""]
#Filter for volume containing 4 or 2
filter_df = combined_df[(combined_df['Volume'] == '4') | (combined_df['Volume'] == '2') | (combined_df['Volume'] == '4_2') | (combined_df['Volume'] == '2_2')]
filter_df['Base volume'] = filter_df['Volume'].apply(lambda x: x.split('_')[0])

#filter for volume containing 4 or 2
for volume in filter_df['Base volume'].unique():
    print(f'Volume: {volume}')
    df_volume_filter = filter_df[filter_df['Base volume'] == volume]  
    for muscle in df_volume_filter['Muscle'].unique():
        print(f'Muscle: {muscle}')
        df_volume_muscle_filter = df_volume_filter[df_volume_filter['Muscle'] == muscle]
        df_1 = df_volume_muscle_filter[df_volume_muscle_filter['Second scan'] == '']
        df_2 = df_volume_muscle_filter[df_volume_muscle_filter['Second scan'] != '']
        wilcoxon = pg.wilcoxon(df_1['Dice'], df_2['Dice'])
        input_dict = {'Volume': volume, 'Muscle': muscle, 'W-val': wilcoxon['W-val'].values[0], 'p-val': wilcoxon['p-val'].values[0]}
        
        register_df = pd.concat([register_df, pd.DataFrame([input_dict])], ignore_index=True)
            #print(f'***** \n ICC for {muscle} in group {exp}: \n \n {icc}')


# %%
#Overall ttest between experience groups
from scipy.stats import ttest_ind
group_0 = combined_df[combined_df['Experience binary'] == 0]['Dice']
group_1 = combined_df[combined_df['Experience binary'] == 1]['Dice']
ttest = ttest_ind(group_0, group_1)
print(ttest.pvalue)
#%%

#mann whitney u test (per volume and muscle)
from scipy.stats import mannwhitneyu
mannwhitney_df = pd.DataFrame()
#mu per volume
mannwhitney_df_volume = pd.DataFrame()
for volume in combined_df['Volume'].unique():
    df_volume = combined_df[combined_df['Volume'] == volume]
    for muscle in df_volume['Muscle'].unique():
        df_muscle = df_volume[df_volume['Muscle'] == muscle]
        group_0 = df_muscle[df_muscle['Experience binary'] == 0]['Surface deviation']
        group_1 = df_muscle[df_muscle['Experience binary'] == 1]['Surface deviation']
        group_0_median_deviation = group_0.median()
        group_1_median_deviation = group_1.median()
        mannwhitney = mannwhitneyu(group_0, group_1)
        input_dict = {'Volume': volume, 'U-statistic': mannwhitney.statistic, 'p-value': mannwhitney.pvalue, 'Muscle': muscle, 'Median deviation group 0': group_0_median_deviation, 'Median deviation group 1': group_1_median_deviation}
        mannwhitney_df_volume = pd.concat([mannwhitney_df_volume, pd.DataFrame([input_dict])], ignore_index=True)

mannwhitney_df_volume_filtered = mannwhitney_df_volume[mannwhitney_df_volume['p-value'] < 0.05]
mannwhitney_df_volume_filtered.to_excel(os.path.join(main_path, 'mannwhitney_df_volume_filtered.xlsx'))
#%%
from scipy.stats import mannwhitneyu

mannwhitney_df_volume = pd.DataFrame()
for muscle in combined_df['Muscle'].unique():
    df_muscle = combined_df[combined_df['Muscle'] == muscle]
    
    for volume in df_muscle['Volume'].unique():
        df_volume = df_muscle[df_muscle['Volume'] == volume]
        group_0 = df_volume[df_volume['Experience binary'] == 0]['Dice']
        group_1 = df_volume[df_volume['Experience binary'] == 1]['Dice']
        group_0_median_dsc = group_0.median()
        group_1_median_dsc = group_1.median()
        mannwhitney = mannwhitneyu(group_0, group_1)
        input_dict = {'Volume': volume, 'Muscle': muscle, 'Group0medianDSC':group_0_median_dsc,'Group1medianDSC':group_1_median_dsc, 'p-value': mannwhitney.pvalue}
        mannwhitney_df_volume = pd.concat([mannwhitney_df_volume, pd.DataFrame([input_dict])], ignore_index=True)

mannwhitney_df_volume.to_excel(os.path.join(main_path, 'mannwhitney_df_volume.xlsx'))

#%%
#p < 0.05
mu_p = mannwhitney_df_volume[mannwhitney_df_volume['p-value'] < 0.05]

#Overall
group_0_complete = combined_df[combined_df['Experience binary'] == 0]['Dice']
group_1_complete = combined_df[combined_df['Experience binary'] == 1]['Dice']
mannwhitney_complete = mannwhitneyu(group_0_complete, group_1_complete)
mannwhitney_complete.pvalue

for row in mu_p.iterrows():
    vol = row[1]['Volume']
    muscle = row[1]['Muscle']

    df_vol = combined_df[combined_df['Volume'] == vol]
    df_muscle = df_vol[df_vol['Muscle'] == muscle]
    group_0 = df_muscle[df_muscle['Experience binary'] == 0]['Dice']
    group_1 = df_muscle[df_muscle['Experience binary'] == 1]['Dice']
    print(f'Volume: {vol}, Muscle: {muscle}')
    print(f'Group 0 median: {group_0.median()}')
    print(f'Group 1 median: {group_1.median()}')
    
# %%
#Merge expertise binary on Observer
repeated_dsc_df = pd.read_csv(os.path.join(main_path, 'repeated_dsc_df.csv'))
combined_df = pd.read_csv(os.path.join(main_path, 'final_df_v4.csv'))

repeated_dsc_df['Expertise binary'] = repeated_dsc_df['Observer'].apply(lambda x: 1 if x in combined_df[combined_df['Experience binary'] == 1]['Observer'].unique() else 0) 
len(repeated_dsc_df)

median_1 = repeated_dsc_df[repeated_dsc_df['Expertise binary'] == 1]['DSC'].median().round(3)
median_0 = repeated_dsc_df[repeated_dsc_df['Expertise binary'] == 0]['DSC'].median().round(3)
iqr_1 = [repeated_dsc_df[repeated_dsc_df['Expertise binary'] == 1]['DSC'].quantile(0.75).round(3), repeated_dsc_df[repeated_dsc_df['Expertise binary'] == 1]['DSC'].quantile(0.25).round(3)]
iqr_0 = [repeated_dsc_df[repeated_dsc_df['Expertise binary'] == 0]['DSC'].quantile(0.75).round(3), repeated_dsc_df[repeated_dsc_df['Expertise binary'] == 0]['DSC'].quantile(0.25).round(3)]
print(f'Median DSC for expert group: {median_1} with IQR: {iqr_1}')
print(f'Median DSC for non-expert group: {median_0} with IQR: {iqr_0}')
