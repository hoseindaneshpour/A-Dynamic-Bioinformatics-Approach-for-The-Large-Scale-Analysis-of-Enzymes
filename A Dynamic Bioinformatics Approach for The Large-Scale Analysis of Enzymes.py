#!/usr/bin/env python
# coding: utf-8


get_ipython().system('pip install LogoData')


import numpy as np
import ruptures as rpt
import pandas as pd
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import requests
import time
from Bio.PDB import PDBParser
import os
import seaborn as sns
import plotly.express as px



logofile = "./data/....txt"
msafile = "./data/....aln"
alignment = AlignIO.read(msafile, "clustal") 
print(f"Length of alignment file {len(alignment)} records")
lencheck=[]
for record in alignment:
    lencheck.append(len(record.seq))
print(f"Unique seq lengths in MSA: {set(lencheck)}")
if len(set(lencheck)) > 1:
    raise Exception(f"Sequences of inequal length in {msafile}")
    
dfweights0 = pd.read_csv(logofile, sep="\t", header=7) 
dfweights0.rename(columns={"#": "position"}, inplace=True)
logo_width = len(dfweights0)
print(f"MSA columns in logofile {logo_width}")
if lencheck[0] != logo_width:
    raise Exception(f"Mismatch between {logofile} and {msafile}")        



SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



dfweights0 = pd.read_csv(logofile, sep="\t", header=7)  
dfweights0.rename(columns={"#": "position"}, inplace=True)
fig, ax1 = plt.subplots(figsize=(16, 5))
ax2 = ax1.twinx()
dfweights0.plot(x='position', y='Entropy', ax=ax1, legend=False,color='g')
dfweights0.plot(x='position', y='Weight', ax=ax2, legend=False, color='r')
ax1.set_xlabel('Position')
ax1.set_ylabel('Entropy', color='g')
ax2.set_ylabel('Weight', color='r')
plt.show()



# running average of n residues
fig, ax1 = plt.subplots(figsize=(16, 5)) 
ax2 = ax1.twinx()
dfweights0['Entropy_mean'] = dfweights0['Entropy'].rolling(window=10, center=True).mean() 
dfweights0.plot(x='position', y='Entropy_mean', ax=ax1, legend=False,color='g')
dfweights0.plot(x='position', y='Weight', ax=ax2, legend=False, color='r')
ax1.set_xlabel('Position')
ax1.set_ylabel('Entropy', color='g')
ax2.set_ylabel('Weight', color='r')
ax1.set_title('with running average of 10 residues for Entropy')
plt.show()



# Normalizing the weight and entropy to the range 0 to 1
weight_norm = (dfweights0['Weight'] - dfweights0['Weight'].min()) / (dfweights0['Weight'].max() - dfweights0['Weight'].min())
entropy_norm = (dfweights0['Entropy_mean'] - dfweights0['Entropy_mean'].min()) / (dfweights0['Entropy_mean'].max() - dfweights0['Entropy_mean'].min())
# the average of the normalized values
joint_quantity = (weight_norm + entropy_norm) / 2
# Adding the joint_quantity column to the df
dfweights0['Average'] = joint_quantity
# Plotting the joint_quantity as a line
fig, ax = plt.subplots(figsize=(16, 5))
dfweights0.plot(x='position', y='Average', ax=ax, legend=False, color='b')
ax.set_xlabel('Position')
ax.set_ylabel('Average', color='b')
ax.set_title('Normalized Average of weight and entropy of residues')
ax.grid()
plt.show()




algorithm = rpt.Pelt(model="l1",min_size=1).fit(np.asanyarray(dfweights0["Weight"]))
result = algorithm.predict(pen=2)
rpt.display(np.asanyarray(dfweights0["Weight"]), result, figsize=(16, 5))
#plt.xlim(left=300, right=400)
plt.xlabel('Position')
plt.ylabel('Weight')
plt.show()





dfweights= dfweights0.loc[:, ["position", 'Weight']]



#downloading of alphafold models for each sequence in an MSA
#if they are not previously in model_dir rewrites the alignment if models were missing and could not be downloaded
def download_file(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception if the request was not successful
    with open(save_path, "wb") as file:
        file.write(response.content)

model_dir = "./data/Models/"
data1= []
uniprot_ids = [record.id.split("|")[1] for record in alignment]
found = 0
not_found = 0
dl_ok = 0
dl_fail = 0

if os.path.isdir(model_dir):
    #checking both that the pathname exist and that it's a directory
    print(f"Using folder {model_dir} to save models")
else:
    raise FileNotFoundError(f"The folder {model_dir} does not exist!")
    # crashes with this error if model_dir is not present
    
for uniprot_id in uniprot_ids:
    file_name = f"{model_dir}AF-{uniprot_id}-F1-model_v4.pdb"
    if os.path.isfile(file_name) == False:
        not_found +=1
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        save_path = file_name
        try:
            download_file(url, save_path)
            dl_ok +=1
        except:
            dl_fail +=1
            print(f"{file_name} download failed. Manual download link:")
            print (url)
    else:
        found +=1
        
print(f"Found {found} models, {not_found} were missing; Downloaded {dl_ok}, failed {dl_fail}.")

if (dl_fail > 0):
    #creating a new MSA if not all models could be downloaded
    print("Rewriting new alignment file of the found records.")
    new_msa = MultipleSeqAlignment([]) #empty MSA object
    seqs = 0

    for record in alignment: 
        uniprot_id = record.id.split("|")[1]
        file_name = f"{model_dir}AF-{uniprot_id}-F1-model_v4.pdb"
        if os.path.isfile(file_name): #if the model is there we write it into the new MSA
            new_msa.append(record)
            seqs += 1
            #would be better if the previous had written a list of record indices 
            #as the files were retrieved

    new_msafile = f"{msafile}.new{seqs}.fasta"
    AlignIO.write(new_msa, new_msafile, "fasta")
    print(seqs, "sequences as", new_msafile)
    print("Now realign the MSA, make a new logo and start with new msafile and logofile")




alignment = AlignIO.read(msafile, "clustal")
data1= []
for record in alignment:
    for i, residue in enumerate(record.seq):
        if residue != "-":
            sequence_id = record.id.split("|")[1]
            data1.append({"Uniprot ID": sequence_id, "Residue": residue, "position": i+1})
df1 = pd.DataFrame(data1)
aa_dict = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL"
}
df1["Residue"] = df1["Residue"].apply(lambda x: aa_dict.get(x))
#df1 # data frame for Uniprot ID & Residue & column number



new_df = pd.DataFrame(columns=['weights_collection'])
positions = df1['position']
for position in positions:
    value = dfweights.loc[dfweights['position'] == position].drop('position', axis=1).values[0][0]
    new_df = pd.concat([new_df, pd.DataFrame({'weights_collection': [value]})], ignore_index=True)




df2 = pd.merge(df1, new_df, left_index=True, right_index=True, suffixes=('_df1', '_dfweights'))

# In[15]:

uniprot_ids = [record.id.split("|")[1] for record in alignment]
model_dir="./data/Models/"
parser = PDBParser()
data = []
for uniprot_id in uniprot_ids:
    file_name = f"{model_dir}AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        structure = parser.get_structure(f"AF-{uniprot_id}-F1-model_v4", file_name)

        for model in structure:
            for chain in model:
                for residue in chain:
                    b_factor = residue["CA"].get_bfactor()
                    data.append({"Uniprot ID": uniprot_id, "Residue": residue.get_resname(),
                                 "Residue ID": residue.get_id()[1], "B Factor": b_factor})
    except FileNotFoundError:
        print(f"File {file_name} does not exist. Continuing to the next sequence.")
        continue
df3 = pd.DataFrame(data)



ready_df = pd.concat([df3, df2.iloc[:, [2,3]]], axis=1)
ready_df



ready_df.to_csv('ready_ML.csv', index=False)




x = ready_df['B Factor']
y = ready_df['weights_collection']
sns.regplot(x=x, y=y,line_kws={'color':'r'}, scatter_kws={'s': 5})
plt.xlabel('pLDDT')
plt.ylabel('MSA weights')
plt.title('Whole MSA positions')
A = [20, 50, 50, 20]
B = [0.6, 0.6, 1, 1]
plt.fill(A, B, 'orange',alpha=0.2)
plt.gcf().set_size_inches(16, 5)
plt.show()




sns.displot(ready_df['B Factor'], kde=True, height=5, aspect=4, bins=39,color='green')
plt.title('Whole MSA positions')
plt.show()




print("correlation of whole MSA:",ready_df['B Factor'].corr(ready_df['weights_collection']))




filtered_RED_box = ready_df[(ready_df['weights_collection'] > 0.6) & (ready_df['B Factor'] < 50)]
fig, ax = plt.subplots(figsize=(16,5))
ax.scatter(filtered_RED_box['position'], filtered_RED_box['B Factor'])
ax.set_ylabel('pLDDT')
ax.set_facecolor('blanchedalmond')
plt.show()


fig, ax = plt.subplots(figsize=(16,5))
ax.hist(filtered_RED_box['position'], bins=220)
ax.set_xlabel('MSA position')
ax.set_ylabel('Frequency')
ax.set_facecolor('blanchedalmond')
plt.show()




dca_start = 360
dca_end = 800
df_dcadom = ready_df[(ready_df['position'] >= dca_start) & (ready_df['position'] <= dca_end)]
x = df_dcadom['B Factor']
y = df_dcadom['weights_collection']
fig, ax = plt.subplots(figsize=(16,5))
sns.regplot(x=x, y=y,line_kws={'color':'red'}, scatter_kws={'s': 5})
plt.xlabel('pLDDT')
plt.ylabel('MSA weight')
plt.title(f'DCA domain {dca_start} to {dca_end}')




df_dcadom=ready_df[(ready_df['position'] >= dca_start) & (ready_df['position'] <= dca_end)]
sns.displot(df_dcadom['B Factor'], kde=True, bins=39, height=5, aspect=4,color='green') 
plt.title(f'Histogram of DCA domain {dca_start} to {dca_end}')




ready_df_copy = ready_df.copy()
df_360_800=ready_df_copy.loc[(ready_df_copy['position'] >= 360) & (ready_df_copy['position'] <= 800)]
print("correlation of 360-800:", df_360_800['B Factor'].corr(df_360_800['weights_collection']))




ready_df_copy = ready_df.copy()




ready_df_copy.groupby('position')['B Factor'].mean().plot(figsize=(16,5),color='green') 
#average pLDDT across the MSA 
plt.ylabel('Mean pLDDT')
# the aggregated “bfactor” column by mean based on the “position” column id in the dataframe




fig, ax = plt.subplots(figsize=(16, 5))
ax2 = ax.twinx()
dfweights.plot(x="position", y=["Weight"], kind="line", ax=ax, color='red', ylim=[0,1.25])
ready_df_copy.groupby('position')['B Factor'].mean().plot(ax=ax2, color='b')
ax1.set_ylabel('pLDDT', color='b')
ax2.set_ylabel('MSA weight', color='r')
plt.xlim(330,710)




normalized_b_factors = (ready_df_copy.groupby('position')['B Factor'].
                        mean()-min(ready_df_copy.groupby('position')['B Factor'].
                                   mean()))/(max(ready_df_copy.groupby('position')['B Factor'].
                                                 mean())-min(ready_df_copy.groupby('position')['B Factor'].mean()))
fig, ax = plt.subplots(figsize=(16, 5))
dfweights['pLDDT times MSA weight'] = normalized_b_factors * dfweights['Weight']
dfweights.plot(x="position", y=["pLDDT times MSA weight"], kind="line", ax=ax, color='g')
ax.set_ylabel('mean pLDDT * MSA weight', color='g')
plt.xlabel("position")
plt.xlim(330,710)
plt.show()
plt.plot(normalized_b_factors) 


# Statistics of good-quality residues in AF models


limit0 = pd.DataFrame(ready_df_copy.groupby('position')['B Factor'].mean())
limits = limit0.reset_index()
limits.columns = ['position', 'B Factor']
column_values1=limits.loc[(limits['position'] >= dca_start)
                          &(limits['position'] <= dca_end)
                          & (limits['B Factor'] >= 70) ]['position'].values
len(column_values1) 




column_values2= limits.loc[(limits['position'] >= dca_start)
                           &(limits['position'] <= dca_end)
                           &(limits['B Factor'] >= 90) ]['position'].values
len(column_values2) 




column_values3 = limits.loc[(limits['position'] >= dca_start)
                           &(limits['position'] <= dca_end) 
                            & (limits['B Factor'] < 70) ]['position'].values
len(column_values3) 




column_values4= limits.loc[(limits['position'] >= dca_start)
                           &(limits['position'] <= dca_end)
                           &(limits['B Factor'] >= 95) ]['position'].values
len(column_values4) 




filtered_df1 = pd.DataFrame(columns=['Uniprot ID', 'Column Values'])
for uniprot_id in ready_df['Uniprot ID'].unique():
    column_values = ready_df.loc[(ready_df['position'] >= dca_start) 
                                 & (ready_df['position'] <= dca_end)
                                 & (ready_df['B Factor'] >= 70) 
                                 & (ready_df['Uniprot ID'] == uniprot_id)]['position'].values
    filtered_df1 = pd.concat([filtered_df1, pd.DataFrame({'Uniprot ID': [uniprot_id], 
                                                          'Column Values': [column_values]})])
    residue_count = ready_df[(ready_df['position'] >= dca_start) 
                             & (ready_df['position'] <= dca_end) &
                             (ready_df['Uniprot ID'] == uniprot_id)].groupby('Uniprot ID')['Residue'].count()
    filtered_df1.loc[filtered_df1['Uniprot ID'] == uniprot_id, 
                     f"{dca_start} < Residue Count < {dca_end}"] = residue_count[0]
filtered_df1 = filtered_df1.reset_index(drop=True)
filtered_df1['Len of Column Values no. of > 70'] = filtered_df1['Column Values'].apply(len)



filtered_df2 = pd.DataFrame(columns=['Uniprot ID', 'Column Values'])
for uniprot_id in ready_df['Uniprot ID'].unique():
    column_values = ready_df.loc[(ready_df['position'] >= dca_start) 
                                 & (ready_df['position'] <= dca_end) 
                                 & (ready_df['B Factor'] >= 90) 
                                 & (ready_df['Uniprot ID'] == uniprot_id)]['position'].values
    filtered_df2 = pd.concat([filtered_df2, pd.DataFrame({'Uniprot ID': [uniprot_id], 'Column Values': [column_values]})])
filtered_df2 = filtered_df2.reset_index(drop=True)
filtered_df2['Len of Column Values no. of > 90'] = filtered_df2['Column Values'].apply(len) 



filtered_df3 = pd.DataFrame(columns=['Uniprot ID', 'Column Values', 'Residue'])
for uniprot_id in ready_df['Uniprot ID'].unique():
    column_values = ready_df.loc[(ready_df['position'] >= dca_start) &
                                 (ready_df['position'] <= dca_end) &
                                 (ready_df['B Factor'] < 70) &
                                 (ready_df['Uniprot ID'] == uniprot_id)]['position'].values
    residues = ready_df.loc[(ready_df['position'] >= dca_start) &
                            (ready_df['position'] <= dca_end) &
                            (ready_df['B Factor'] < 70) &
                            (ready_df['Uniprot ID'] == uniprot_id)]['Residue'].values
    filtered_df3 = pd.concat([filtered_df3, pd.DataFrame({'Uniprot ID': [uniprot_id],
                                                          'Column Values': [column_values],
                                                          'Residue': [residues]})])

filtered_df3 = filtered_df3.reset_index(drop=True)
filtered_df3['Len of Column Values no. of < 70'] = filtered_df3['Column Values'].apply(len)



filtered_df4 = pd.DataFrame(columns=['Uniprot ID', 'Column Values'])
for uniprot_id in ready_df['Uniprot ID'].unique():
    column_values = ready_df.loc[(ready_df['position'] >= dca_start) 
                                 & (ready_df['position'] <= dca_end) &
                                 (ready_df['B Factor'] >= 70) & (ready_df['B Factor'] <  80) &
                                 (ready_df['Uniprot ID'] == uniprot_id)]['position'].values
    filtered_df4 = pd.concat([filtered_df4, pd.DataFrame({'Uniprot ID': [uniprot_id], 'Column Values': [column_values]})])
filtered_df4 = filtered_df4.reset_index(drop=True)
filtered_df4['80>Len of Column Values no. of > 70'] = filtered_df4['Column Values'].apply(len) 



filtered_df5 = pd.DataFrame(columns=['Uniprot ID', 'Column Values'])
for uniprot_id in ready_df['Uniprot ID'].unique():
    column_values = ready_df.loc[(ready_df['position'] >= dca_start) 
                                 & (ready_df['position'] <= dca_end) &
                                 (ready_df['B Factor'] >= 80) & (ready_df['B Factor'] <  90) &
                                 (ready_df['Uniprot ID'] == uniprot_id)]['position'].values
    filtered_df5 = pd.concat([filtered_df5, pd.DataFrame({'Uniprot ID': [uniprot_id], 'Column Values': [column_values]})])
filtered_df5 = filtered_df5.reset_index(drop=True)
filtered_df5['90>Len of Column Values no. of > 80'] = filtered_df5['Column Values'].apply(len) 


filtered_df = pd.concat([filtered_df1.drop('Column Values', axis=1),
                         filtered_df4.drop(['Column Values', "Uniprot ID"],axis=1),
                         filtered_df5.drop(['Column Values', "Uniprot ID"],axis=1),
                         filtered_df2.drop(['Column Values', "Uniprot ID"], axis=1),
                         filtered_df3.drop(["Uniprot ID"], axis=1)], axis=1)
# filtered_df
filtered_df.style.set_table_styles([{'selector': 'th:nth-child(8), th:nth-child(9), th:nth-child(10)',
                                     'props': [('background-color', 'yellow')]},
                                    {'selector': 'th:nth-child(7)', 'props': [('background-color', 'blue')]},
                                    {'selector': 'th:nth-child(5), th:nth-child(6)',
                                     'props': [('background-color', '#87CEFA')]}])



html = filtered_df.to_html()
html = html.replace('<table', '<table style="border-collapse: collapse;"')
with open('output.html', 'w') as f:
    f.write(html)





fig = px.line(ready_df, x='position', y='B Factor', color='Uniprot ID')
fig.update_layout(legend=dict(orientation="h",yanchor="bottom", y=1.02, xanchor="right", x=1), width=950, height=600)
fig.show()





fig = px.line(
    ready_df, 
    x='position', 
    y='B Factor', 
    color='Uniprot ID',
    color_discrete_sequence=px.colors.qualitative.Set2  
)
fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=950, 
    height=600
)
fig.show()





fig, axs = plt.subplots(16, 4, figsize=(20, 80))
for i, uniprot_id in enumerate(ready_df['Uniprot ID'].unique()):
    row = i // 4
    col = i % 4
    axs[row][col].plot(ready_df[ready_df['Uniprot ID'] == uniprot_id]['position'], ready_df[ready_df['Uniprot ID'] == uniprot_id]['B Factor'])
    axs[row][col].set_title(uniprot_id)
    axs[row][col].set_xlim([360, 800])
plt.show()







