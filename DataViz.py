
################################################################################
# Libraries

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cm_to_inch = 1 / 2.54
plt.rcParams.update({'font.size': 8}) # set all font sizes to 8

################################################################################
# Settings

OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'
EXCEL_FILE  = 'intensity_results_manual.xlsx'

################################################################################
# Load data

df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'

################################################################################
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

