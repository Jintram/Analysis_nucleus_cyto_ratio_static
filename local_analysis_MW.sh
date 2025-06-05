#!/bin/bash


# Set paths with overall input and output directories
BASE_PATH_IN='/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/DATA_ALL/NFAT_analysis_Martijn/'
BASE_PATH_OUT='/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/'

# Go to right directory and activate conda env
conda activate 2025_FLIM
cd /Users/m.wehrens/Documents/git_repos/_UVA/2024_scripts-others/Analysis_nucleus_cyto_ratio_static

# Process the data for figure 1 (done earlier manually, hence commented out)
# FIG1_PATH='NFAT1/1.NFAT1_WT_vs_RQ/'
# python Segmentation_script.py ${BASE_PATH_IN}${FIG1_PATH} ${BASE_PATH_OUT}${FIG1_PATH}

# Process data for figure 2-6
FIGURE_LIST=(
    'NFAT1/2.NFAT1_shDSCR1/'
    'NFAT1/3.NFAT1_FK506/'
    'NFAT2/4.NFAT2_WT_vs_RQ/'
    'NFAT2/5.NFAT2_shDSCR1/'
    'NFAT2/6.NFAT2_FK506/'
)

FIGURE_LIST=('NFAT2/5.NFAT2_shDSCR1/' 'NFAT2/6.NFAT2_FK506/')

for CURRENT_FIG_PATH in "${FIGURE_LIST[@]}"; do
    python Segmentation_script.py "${BASE_PATH_IN}${CURRENT_FIG_PATH}" "${BASE_PATH_OUT}${CURRENT_FIG_PATH}"
done
