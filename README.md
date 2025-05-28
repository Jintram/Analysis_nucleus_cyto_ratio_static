


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jintram/Analysis_nucleus_cyto_ratio_static/blob/main/User_friendly_Colab_interface.ipynb)


# Static_image_segmentation



### Overview

This repository contains a python script, R script and a user friendly notebook. The python script performs image pre processing and nucleus/cytoplasm segmentation. The script takes an image stack as input, where the first image in the stack is a nuclear marked image (where only the nucleus is visible) while the second image is the actual data channel. After the pre processing and segmentation, the average intensity of both the nucleus and cytoplasm is calculated, per individual cell. The output contains a csv file with the nucleus and cytoplasm intensities per cell, as well as visualisations of the generated masks and labeled cells, which can be used for interpretation of the data. The generated csv file is used in a R script that visualises the data (also included in notebook). Click the 'Open in Colab button in the top left to automatically take you to a Google Colab environment. You can save a personal copy of this notebook to your Google Drive by clicking "File" > "Save a copy in Drive". Once saved, you can modify it however you like.


### Features

Supports tiff file format

OTSU thresholding 

Outputs segmented images with visualization

Non coder friendly interface (Google Colab)

Google drive compatible

Distinguishes WT and RQ by detecting -WT- & -RQ_ in your input data filename.


**Running the scripts**

The user friendly interface can be downloaded and uploaded to google Colab for easy usage. However, the raw python script can also be downloaded for offline use. 


**Contact**

For questions or feedback, please contact Julian de Swart at julian.de.swart@student.uva.nl, or open an issue in the repository.


