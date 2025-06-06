
################################################################################
# Libraries

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np

cm_to_inch = 1 / 2.54
plt.rcParams.update({'font.size': 8}) # set all font sizes to 8


################################################################################
# Make figure 1

#####
# Load data
# OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'
# EXCEL_FILE  = 'intensity_results_manual.xlsx'
# df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
# df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'

# #####
# # Make figure
# fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))

# df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
# df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]

# # Plot 1, showing which outliers are removed
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'black', 'no': 'grey'}, alpha=.8)
# # sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# # sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# # axs[0].legend(['kept', 'discarded'], loc='best')

# # Plot 2, outliers removed
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, color='black', ax=axs[1], size=1)
# sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])

# plt.tight_layout()

# # save fig
# # plt.show()
# plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

# plt.close()

# Figure 1, remade in style of 2-6 
# (this was done later)


OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn_Exp1/NFAT1/1.NFAT1_WT_vs_RQ/'
EXCEL_FILE  = '1.NFAT1_WT_vs_RQ_intensity_results_manual.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
os.makedirs(OUTPUT_DIR+'/plots/', exist_ok=True)

# Annotate conditions
# Add rep
# I added "Rep" manually to the excel file for exp 1, since the files were processed from a different file structure without "rpX_" prefixes.
# df_alldata.loc[:,'Rep'] = df_alldata['Filename+CellID'].str.split('_').str[0]
df_alldata.loc[:,'Rep'] = df_alldata.loc[:,'Rep'].astype(int).astype(str) # convert to string for consistency with other plots
# Search for WT/RQ
df_alldata.loc[:,'Phenotype'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[4]
# df_alldata.loc[:,'Treatment'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[3]
# Combine WT+treatment
# df_alldata.loc[:,'Pheno_Treat'] = df_alldata.loc[:,'Phenotype']+'_'+df_alldata.loc[:,'Treatment']
# Cells w/ roundness below threshold
df_alldata.loc[:,'Low_Roundness'] = df_alldata['Roundness nucleus']<0.4
# Cells with weird sizes
df_alldata.loc[:,'Weird_size'] = (df_alldata['Nucleus area (px)']<5000) | (df_alldata['Nucleus area (px)']>20_000)

    # np.unique(df_alldata.loc[:,'Phenotype'])
    
# Custom order of 
df_alldata.loc[:,'Phenotype_fct'] = pd.Categorical(df_alldata.loc[:,'Phenotype'], categories=[  'WT' , 'RQ'], ordered=True)    

# Save excel file
df_alldata.to_excel(OUTPUT_DIR + '1.NFAT1_WT_vs_RQ_intensity_results_manual_annotated.xlsx', index=False)

#####
# First a plot showing roundness parameter
fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 6*cm_to_inch))
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", hue='Weird_size', data=df_alldata, jitter=True, ax=axs[0], size=3, alpha=.8)
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio medians", hue='Roundness nucleus', data=df_alldata, jitter=True, ax=axs[1], size=3, alpha=.8)
plt.show(); plt.close()


#####
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]

# Version 1

fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 8*cm_to_inch))
# Plot 1, showing which outliers are removed
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'red', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')
axs[0].tick_params(axis='x', rotation=90)
axs[0].legend_.remove()

# Plot 2, outliers removed
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, hue='Rep', ax=axs[1], size=1) # color='black', 
sns.violinplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])
axs[1].tick_params(axis='x', rotation=90)
axs[1].legend_.remove()
# Ax lims
axs[1].set_ylim(0, 1.1*df_alldata_kept['Nucleus/Cytoplasm Ratio'].max())

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

plt.close()

# Version 2 of plots

fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 8*cm_to_inch))
# Plot 1, showing which outliers are removed
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'red', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')
axs[0].tick_params(axis='x', rotation=90)
axs[0].legend_.remove()

# Plot 2, outliers removed
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, hue='Rep', ax=axs[1], size=1, zorder=1) # color='black', 
sns.violinplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="lightgrey", ax=axs[1], inner=None, zorder=0, edgecolor='black', linewidth=.5)
sns.boxplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, ax=axs[1], showfliers=False, zorder=3, linewidth=.5, width=.2,
    boxprops=dict(facecolor='none', edgecolor='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='black'))
axs[1].tick_params(axis='x', rotation=90)
axs[1].legend_.remove()
# Ax lims
axs[1].set_ylim(0, 1.1*df_alldata_kept['Nucleus/Cytoplasm Ratio'].max())

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual_v2.pdf', dpi=600, bbox_inches='tight')

plt.close()

################################################################################
# Make figure 2

#####
# Load data
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/NFAT1/2.NFAT1_shDSCR1/'
EXCEL_FILE  = '2.NFAT1_shDSCR1_intensity_results_manual.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
os.makedirs(OUTPUT_DIR+'/plots/', exist_ok=True)

# Annotate conditions
# Add rep
df_alldata.loc[:,'Rep'] = df_alldata['Filename+CellID'].str.split('_').str[0]
# Search for WT/RQ
df_alldata.loc[:,'Phenotype'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[2]
df_alldata.loc[:,'Treatment'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[3]
# Combine WT+treatment
df_alldata.loc[:,'Pheno_Treat'] = df_alldata.loc[:,'Phenotype']+'_'+df_alldata.loc[:,'Treatment']
# Cells w/ roundness below threshold
df_alldata.loc[:,'Low_Roundness'] = df_alldata['Roundness nucleus']<0.4
# Cells with weird sizes
df_alldata.loc[:,'Weird_size'] = (df_alldata['Nucleus area (px)']<5000) | (df_alldata['Nucleus area (px)']>20_000)

    # np.unique(df_alldata.loc[:,'Pheno_Treat'])
    
# Custom order of 
df_alldata.loc[:,'Pheno_Treat_fct'] = pd.Categorical(df_alldata.loc[:,'Pheno_Treat'], categories=[  'WT_shControl', 'WT_shDSCR1' , 'RQ_shControl', 'RQ_shDSCR1'], ordered=True)    

# Save excel file
df_alldata.to_excel(OUTPUT_DIR + 'intensity_results_manual_annotated.xlsx', index=False)


#####
# First a plot showing roundness parameter
fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 6*cm_to_inch))
sns.stripplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", hue='Weird_size', data=df_alldata, jitter=True, ax=axs[0], size=3, alpha=.8)
sns.stripplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio medians", hue='Roundness nucleus', data=df_alldata, jitter=True, ax=axs[1], size=3, alpha=.8)
plt.show(); plt.close()


#####
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]


fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 8*cm_to_inch))
# Plot 1, showing which outliers are removed
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'red', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')
axs[0].tick_params(axis='x', rotation=90)
axs[0].legend_.remove()

# Plot 2, outliers removed
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, hue='Rep', ax=axs[1], size=1) # color='black', 
sns.violinplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])
axs[1].tick_params(axis='x', rotation=90)
axs[1].legend_.remove()
# Ax lims
axs[1].set_ylim(0, 1.1*df_alldata_kept['Nucleus/Cytoplasm Ratio'].max())

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

plt.close()




################################################################################
# Make figure 3

#####
# Load data
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/NFAT1/3.NFAT1_FK506/'
EXCEL_FILE  = '3.NFAT1_FK506_intensity_results_MANUAL.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
os.makedirs(OUTPUT_DIR+'/plots/', exist_ok=True)

# Annotate conditions
# Replace control by Con in 'Filename+CellID'
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('control', 'Con')
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('(_|-)NFAT1', '', regex=True)
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('(_|-)HDMECs|(_|-)hHDMECs|(_|-)hdmecs', '', regex=True)
# Add rep
df_alldata.loc[:,'Rep'] = df_alldata['Filename+CellID'].str.split('_|-', regex=True).str[0]
# Search for WT/RQ
df_alldata.loc[:,'Phenotype'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[1]
df_alldata.loc[:,'Treatment'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[2]
# Combine WT+treatment
df_alldata.loc[:,'Pheno_Treat'] = df_alldata.loc[:,'Phenotype']+'_'+df_alldata.loc[:,'Treatment']
# Cells w/ roundness below threshold
df_alldata.loc[:,'Low_Roundness'] = df_alldata['Roundness nucleus']<0.4
# Cells with weird sizes
df_alldata.loc[:,'Weird_size'] = (df_alldata['Nucleus area (px)']<5000) | (df_alldata['Nucleus area (px)']>20_000)

# Custom order of conditions
df_alldata.loc[:,'Pheno_Treat_fct'] = pd.Categorical(df_alldata.loc[:,'Pheno_Treat'], categories=[ 'WT_Con', 'WT_FK506', 'RQ_Con', 'RQ_FK506'], ordered=True)

    # np.unique(df_alldata.loc[:,'Pheno_Treat'])

# Save excel file
df_alldata.to_excel(OUTPUT_DIR + '3.NFAT1_FK506_intensity_results_MANUAL_annotated.xlsx', index=False)


#####
# First a plot showing roundness parameter
fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 8*cm_to_inch))
sns.stripplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", hue='Weird_size', data=df_alldata, jitter=True, ax=axs[0], size=3, alpha=.8)
sns.stripplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", hue='Low_Roundness', data=df_alldata, jitter=True, ax=axs[1], size=3, alpha=.8)
plt.show(); plt.close()


#####
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]


fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed
sns.stripplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'red', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')
axs[0].tick_params(axis='x', rotation=90)
axs[0].legend_.remove()

# Plot 2, outliers removed
sns.stripplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, hue='Rep', ax=axs[1], size=1) # color='black', 
sns.violinplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])
axs[1].tick_params(axis='x', rotation=90)
# remove legend
axs[1].legend_.remove()
# limits
axs[1].set_ylim(0, 1.1*df_alldata_kept['Nucleus/Cytoplasm Ratio'].max())

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

# adjust limits to custom
axs[1].set_ylim(0, 3)
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual_yzoom.pdf', dpi=600, bbox_inches='tight')

plt.close()




################################################################################
# Make figure 4

#####
# Load data
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/NFAT2/4.NFAT2_WT_vs_RQ/'
EXCEL_FILE  = '4.NFAT2_WT_vs_RQ_intensity_results_MANUAL.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
os.makedirs(OUTPUT_DIR+'/plots/', exist_ok=True)

# Annotate conditions
# Replace control by Con in 'Filename+CellID'
df_alldata.loc[:,'Filename+CellID_original'] = df_alldata.loc[:,'Filename+CellID'] # backup original filename
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('control', 'Con')
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('(_|-)NFAT2', '', regex=True)
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('(_|-)HDMECs|(_|-)hHDMECs|(_|-)hdmecs', '', regex=True)
# Add rep
df_alldata.loc[:,'Rep'] = df_alldata['Filename+CellID'].str.split('_').str[0]
# Search for WT/RQ
df_alldata.loc[:,'Phenotype'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[4]
# df_alldata.loc[:,'Treatment'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[2]
# Combine WT+treatment
# df_alldata.loc[:,'Pheno_Treat'] = df_alldata.loc[:,'Phenotype']+'_'+df_alldata.loc[:,'Treatment']
# Cells w/ roundness below threshold
df_alldata.loc[:,'Low_Roundness'] = df_alldata['Roundness nucleus']<0.4
# Cells with weird sizes
df_alldata.loc[:,'Weird_size'] = (df_alldata['Nucleus area (px)']<5000) | (df_alldata['Nucleus area (px)']>20_000)

# Custom order of conditions
# df_alldata.loc[:,'Pheno_Treat_fct'] = pd.Categorical(df_alldata.loc[:,'Pheno_Treat'], categories=[ 'WT_Con', 'WT_FK506', 'RQ_Con', 'RQ_FK506'], ordered=True)
df_alldata.loc[:,'Phenotype_fct'] = pd.Categorical(df_alldata.loc[:,'Phenotype'], categories=['WT', 'RQ'], ordered=True)

    # np.unique(df_alldata.loc[:,'Pheno_Treat'])

# Save excel file
df_alldata.to_excel(OUTPUT_DIR + '4.NFAT2_WT_vs_RQ_intensity_results_MANUAL_annotated.xlsx', index=False)


#####
# First a plot showing roundness parameter
fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 6*cm_to_inch))
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", hue='Weird_size', data=df_alldata, jitter=True, ax=axs[0], size=3, alpha=.8)
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", hue='Low_Roundness', data=df_alldata, jitter=True, ax=axs[1], size=3, alpha=.8)
plt.show(); plt.close()


#####
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]


fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'red', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')
axs[0].tick_params(axis='x', rotation=90)
axs[0].legend_.remove()

# Plot 2, outliers removed
sns.stripplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, hue='Rep', ax=axs[1], size=1) # color='black', 
sns.violinplot(x="Phenotype_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, ax=axs[1], color="grey")
axs[1].tick_params(axis='x', rotation=90)
# limits
axs[1].set_ylim(0, 1.1*df_alldata_kept['Nucleus/Cytoplasm Ratio'].max())

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

plt.close()



################################################################################
# Make figure 5

#####
# Load data
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/NFAT2/5.NFAT2_shDSCR1/'
EXCEL_FILE  = '5.NFAT2_shDSCR1_intensity_results_MANUAL.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
os.makedirs(OUTPUT_DIR+'/plots/', exist_ok=True)

# Annotate conditions
# Replace control by Con in 'Filename+CellID'
df_alldata.loc[:,'Filename+CellID_original'] = df_alldata.loc[:,'Filename+CellID'] # backup original filename
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('control', 'Con')
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('(_|-)NFAT2', '', regex=True)
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('(_|-)HDMECs|(_|-)hHDMECs|(_|-)hdmecs', '', regex=True)
# Add rep
df_alldata.loc[:,'Rep'] = df_alldata['Filename+CellID'].str.split('_|-', regex=True).str[0]
# Search for WT/RQ
df_alldata.loc[:,'Phenotype'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[1]
df_alldata.loc[:,'Treatment'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[2]
# Combine WT+treatment
df_alldata.loc[:,'Pheno_Treat'] = df_alldata.loc[:,'Phenotype']+'_'+df_alldata.loc[:,'Treatment']
# Cells w/ roundness below threshold
df_alldata.loc[:,'Low_Roundness'] = df_alldata['Roundness nucleus']<0.4
# Cells with weird sizes
df_alldata.loc[:,'Weird_size'] = (df_alldata['Nucleus area (px)']<5000) | (df_alldata['Nucleus area (px)']>20_000)

# Custom order of conditions
df_alldata.loc[:,'Pheno_Treat_fct'] = pd.Categorical(df_alldata.loc[:,'Pheno_Treat'], categories=[ 'WT_shControl', 'WT_shDSCR1', 'RQ_shControl', 'RQ_shDSCR1'], ordered=True)
# df_alldata.loc[:,'Phenotype_fct'] = pd.Categorical(df_alldata.loc[:,'Phenotype'], categories=['WT', 'RQ'], ordered=True)

    # np.unique(df_alldata.loc[:,'Pheno_Treat'])

# Save excel file
df_alldata.to_excel(OUTPUT_DIR + '5.NFAT2_shDSCR1_intensity_results_MANUAL_annotated.xlsx', index=False)


#####
# First a plot showing roundness parameter
fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 6*cm_to_inch))
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", hue='Weird_size', data=df_alldata, jitter=True, ax=axs[0], size=3, alpha=.8)
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", hue='Low_Roundness', data=df_alldata, jitter=True, ax=axs[1], size=3, alpha=.8)
plt.show(); plt.close()


#####
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]


fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 8*cm_to_inch))
# Plot 1, showing which outliers are removed
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'red', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')
axs[0].tick_params(axis='x', rotation=90)
axs[0].legend_.remove()

# Plot 2, outliers removed
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, hue='Rep', ax=axs[1], size=1) # color='black', 
sns.violinplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, ax=axs[1], color="grey")
axs[1].tick_params(axis='x', rotation=90)
# limits
axs[1].set_ylim(0, 1.1*df_alldata_kept['Nucleus/Cytoplasm Ratio'].max())
# remove legend
axs[1].legend_.remove()

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

plt.close()




################################################################################
# Make figure 6

#####
# Load data
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/NFAT2/6.NFAT2_FK506/'
EXCEL_FILE  = '6.NFAT2_FK506_intensity_results_MANUAL.xlsx'

df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
os.makedirs(OUTPUT_DIR+'/plots/', exist_ok=True)

# Annotate conditions
# Replace control by Con in 'Filename+CellID'
df_alldata.loc[:,'Filename+CellID_original'] = df_alldata.loc[:,'Filename+CellID'] # backup original filename
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('control|Control', 'Con', regex=True)
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('(_|-)NFAT2', '', regex=True)
df_alldata.loc[:,'Filename+CellID'] = df_alldata['Filename+CellID'].str.replace('(_|-)HDMECs|(_|-)hHDMECs|(_|-)hdmecs', '', regex=True)
# Add rep
df_alldata.loc[:,'Rep'] = df_alldata['Filename+CellID'].str.split('_|-', regex=True).str[0]
# Search for WT/RQ
df_alldata.loc[:,'Phenotype'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[1]
df_alldata.loc[:,'Treatment'] = df_alldata['Filename+CellID'].str.replace('_Stack','').str.split('_|-| ').str[2]
# Combine WT+treatment
df_alldata.loc[:,'Pheno_Treat'] = df_alldata.loc[:,'Phenotype']+'_'+df_alldata.loc[:,'Treatment']
# Cells w/ roundness below threshold
df_alldata.loc[:,'Low_Roundness'] = df_alldata['Roundness nucleus']<0.4
# Cells with weird sizes
df_alldata.loc[:,'Weird_size'] = (df_alldata['Nucleus area (px)']<5000) | (df_alldata['Nucleus area (px)']>20_000)

# Custom order of conditions
df_alldata.loc[:,'Pheno_Treat_fct'] = pd.Categorical(df_alldata.loc[:,'Pheno_Treat'], categories=[ 'WT_Con', 'WT_FK506', 'RQ_Con', 'RQ_FK506'], ordered=True)
# df_alldata.loc[:,'Phenotype_fct'] = pd.Categorical(df_alldata.loc[:,'Phenotype'], categories=['WT', 'RQ'], ordered=True)

    # np.unique(df_alldata.loc[:,'Pheno_Treat'])

# Save excel file
df_alldata.to_excel(OUTPUT_DIR + '6.NFAT2_FK506_intensity_results_MANUAL_annotated.xlsx', index=False)


#####

# First a plot showing roundness parameter
fig, axs = plt.subplots(2, 1, figsize=(10*cm_to_inch, 6*cm_to_inch))
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", hue='Weird_size', data=df_alldata, jitter=True, ax=axs[0], size=3, alpha=.8)
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", hue='Low_Roundness', data=df_alldata, jitter=True, ax=axs[1], size=3, alpha=.8)
plt.show(); plt.close()


#####
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]


fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 8*cm_to_inch))
# Plot 1, showing which outliers are removed
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'red', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')
axs[0].tick_params(axis='x', rotation=90)
axs[0].legend_.remove()

# Plot 2, outliers removed
sns.stripplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, hue='Rep', ax=axs[1], size=1) # color='black', 
sns.violinplot(x="Pheno_Treat_fct", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, ax=axs[1], color="grey")
axs[1].tick_params(axis='x', rotation=90)
# limits
axs[1].set_ylim(0, 1.1*df_alldata_kept['Nucleus/Cytoplasm Ratio'].max())
# remove legend
axs[1].legend_.remove()

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

plt.close()

