
################################################################################
# Libraries

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os

cm_to_inch = 1 / 2.54
plt.rcParams.update({'font.size': 8}) # set all font sizes to 8


################################################################################
# Make figure 1

#####
# Load data
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'
EXCEL_FILE  = 'intensity_results_manual.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'

#####
# Make figure
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))

df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]

# Plot 1, showing which outliers are removed
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'black', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')

# Plot 2, outliers removed
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

plt.close()



################################################################################
# Make figure 2

#####
# Load data
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/NFAT1/2.NFAT1_shDSCR1/'
EXCEL_FILE  = 'intensity_results_manual.xlsx'
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


fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed
sns.stripplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'red', 'no': 'grey'}, alpha=.8)
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
# axs[0].legend(['kept', 'discarded'], loc='best')
axs[0].tick_params(axis='x', rotation=90)
axs[0].legend_.remove()

# Plot 2, outliers removed
sns.stripplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Pheno_Treat", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])
axs[1].tick_params(axis='x', rotation=90)

plt.tight_layout()

# save fig
# plt.show()
plt.savefig(OUTPUT_DIR + 'plots/intensity_results_manual.pdf', dpi=600, bbox_inches='tight')

plt.close()

