######################################################################

# Cytoplasmic and nuclear signal quantification
# 2025-03
#
# Script by Julian de Swart [Julian.de.swart@student.uva.nl]
# Reviewed by Martijn Wehrens, m.wehrens@uva.nl.
# Joachim Goedhart group, Molecular Cytology group, UvA.
#
# As input, this script expectes a folder with image stacks, which contain
# an image with fluorescently labeled nuclei and an image with fluorescent 
# sensor data.
# Based on the nuclear image, the script applies nuclear segmentation 
# based on Otsu's thresholding method and subsequently determines an
# area around the nucleus to define the cytoplasmic region. 
# Then, using the fluorescent data, the script then calculates the mean 
# intensity for each region and determines the ratio between the nucleus
# and cytoplasmic intensity.
# The script involves some more steps and image corrections, see the script 
# below for further details.
# 
# The data is exported to .csv for further analysis, e.g. visualization
# by the 'DataViz.R' script.
#
# Technical note:
# The script can be run from command line like:
# python Segmentation_script.py <input_folder> <output_folder>
# E.g. 
# python Segmentation_script.py /Users/m.wehrens/Data_UVA/2024_10_Sebastian-KTR/static-example/tiff_input/ /Users/m.wehrens/Data_UVA/2024_10_Sebastian-KTR/static-example/output/
# python Segmentation_script.py /Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/DATA/ /Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/

print("Starting script..")

######################################################################
# Load libraries

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import re  # Import regular expressions module

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk, dilation, binary_dilation
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters import median

import sys
from scipy import stats

try:
    from google.colab import drive
    print('Running in Google Colab')
    running_in_colab = True
except:
    print('Not running in Google Colab')
    running_in_colab = False

######################################################################

# Mount Google Drive if not already mounted
if running_in_colab:
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
# Enable interactive mode for plotting
plt.ion

# Function to visualize nucleus segmentation
def visualize_nucleus_segmentation(image, mask, filename):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Segmented Nucleus")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, f"{filename}_Nucleus_segmentation.png"))
    plt.close()  # Close the figure after saving

# Function to visualize cytoplasmic ring creation
def visualize_cytoplasmic_ring(image, cytoplasmic_ring, filename):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Cytoplasmic Ring")
    plt.imshow(cytoplasmic_ring, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, f"{filename}_Cytoplasmic_ring.png"))
    plt.close()  # Close the figure after saving

# Function to visualize nucleus segmentation with masks and cell IDs
def visualize_nucleus_with_ids(image, labeled_mask, filename):
    # image = nucleus_channel_filtered; labeled_mask= nucleus_mask; filename= image_name
    # np.unique(labeled_mask, return_counts=True)

    overlay = label2rgb(labeled_mask, image=image, bg_label=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(overlay)
    plt.title("Nucleus Segmentation with Cell IDs")
    
    # Add cell IDs to the plot
    properties = regionprops(labeled_mask)
    for prop in properties:
        y, x = prop.centroid
        plt.text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

    plt.axis("off")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"{filename}_Cell_IDs.png"))
    plt.close()  # Close the figure after saving

# Function to visualize cytoplasmic ring overlayed on the data channel image
def visualize_cytoplasmic_ring_overlay(data_channel, cytoplasmic_ring_labeled, filename):
    
    cytoplasmic_ring = cytoplasmic_ring_labeled>0
    
    # Calculate dynamic range from the data channel
    vmin, vmax = np.percentile(data_channel[data_channel > 0], (1, 99))
    
    plt.figure(figsize=(12, 6))
    plt.title("Cytoplasmic Ring Overlay on data Channel")
    # Display data channel with dynamic range
    plt.imshow(data_channel, cmap="gray", vmin=vmin, vmax=vmax)
    # Overlay cytoplasmic ring
    plt.imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
              cmap="gray", alpha=0.7)
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, f"{filename}_Cytoplasmic_overlay.png"))
    plt.close()  # Close the figure after saving

# Function to visualize which parts of the image are considered background (<= mode)
def visualize_background_mode(image, mode_value, filename):
    # image=data_channel; filename=image_name
    '''
    Visualize the background mode value in the image.
    Pixels with values <= mode_value are considered background.
    '''
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))    
    fig.suptitle("Background areas highlighted in yellow\n"+"Background should not overlap with cells")
    
    # Create a mask for background pixels
    background_mask = image <= mode_value
    
    # Display the original image
    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(image, cmap="gray")
    
    # Overlay the background mask
    ax[1].imshow(np.ma.masked_where(background_mask == 0, background_mask), 
               cmap="viridis", alpha=0.9, vmin=0, vmax=1)
    
    ax[0].axis("off"); ax[1].axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"{filename}_Background.png"))
    plt.close()  # Close the figure after saving

######################################################################

# Function to subtract mode background from an image
def subtract_mode_background(image):
    # image = data_channel
    '''
    The mode is the most frequently occurring pixel value in the image,
    and it is assumed that there are mostly background pixels in this image,
    such that the mode corresponds to the background intensity, which can be subtracted.
    Negative values are then converted to zeroes. 
    '''
    #Flatten the image to 1D for mode calculation
    flat_image = image.flatten()
        
    #Convert to integer if needed
    if not np.issubdtype(flat_image.dtype, np.integer):
        flat_image = flat_image.astype(np.uint16)  # Convert to 16-bit if not integer
    
    # Remove values that equal the maximum int value (as saturated signal migh incorrectly be perceived as mode)
    max_int_value = np.iinfo(flat_image.dtype).max
    flat_image = flat_image[flat_image < max_int_value]
    
    #Compute mode using bincount
    mode_value = stats.mode(flat_image, keepdims=True)[0][0]  # More robust mode calculation
        # plt.hist(flat_image, bins=100); plt.axvline(mode_value, color='red', linestyle='dashed', linewidth=1)
        # plt.show(); plt.close()
    
    #Substract mode value
    background_subtracted_image = image.astype(np.int32) - mode_value

    #Ensure valid pixel range without clipping to white
    background_subtracted_image[background_subtracted_image < 0] = 0 #Set negative values to 0
        
    return background_subtracted_image.astype(np.uint16), mode_value

# Function to apply median filter to an image
def apply_median_filter(image, radius=2):
    '''
    Apply circular local median filter with radius 'radius'.
    '''
    # Apply median filter with a specified radius
    selem = disk(radius)
    filtered_image = median(image, selem)
    return filtered_image

# Function to segment nucleus
def segment_nucleus(image, min_size=200):
    '''
    Segmenting the nucleus is done by determining a threshold value based
    on Otsu's method.
    The mask, which might contain some artifacts, is then "cleaned up" by
    performing some morphological operations, ie removing small objects and holes,
    and performing binary opening (also smoothens edges). 
    ''' 
    # image = nucleus_channel_filtered; min_size=200
    
    # Determine mask
    thresh = threshold_otsu(image)
    binary_mask = image > thresh
        # plt.imshow(binary_mask); plt.show(); plt.close()
    
    # Clean up mask
    clean_mask = remove_small_objects(binary_mask, min_size=min_size)
    clean_mask = remove_small_holes(clean_mask, area_threshold=10)
    opened_mask = binary_opening(clean_mask, footprint=disk(2))
    # In some cases, additional removal necessary (though these cases are related to artifacts)
    final_mask = remove_small_objects(opened_mask, min_size=min_size)
    
    # Create labeled mask
    labeled_mask = label(final_mask)
    
    # plt.imshow(labeled_mask); plt.show(); plt.close()
            
    return labeled_mask

def create_test_mask():
    '''
    For debugging purposes.
    '''
    
    nucleus_mask = np.array(
        [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,1,1,2,2,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,1,1,2,2,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    
# Function to create cytoplasm ROI
def create_cytoplasm_roi(nucleus_mask, dilation_radius=5, distance_from_nucleus=2):
    '''
    Starting from the nuclei, the cytoplasmic ring is created by expanind that mask 
    twice, once to create a large area around the nucleus, and another time to create 
    and exclusion zone that includes the nucleus and an area around it. By ~subtracting
    the latter from the large area, the cytoplasmic ring is obtained.
    '''
    # dilation_radius=5; distance_from_nucleus=2
    
    #Create the two dilated masks
    expanded_nucleus_mask = dilation(nucleus_mask, footprint=disk(distance_from_nucleus))
    cytoplasm_ring = dilation(expanded_nucleus_mask, footprint=disk(dilation_radius))
        # _,ax=plt.subplots(1, 2); ax[0].imshow(expanded_nucleus_mask); ax[1].imshow(dilated_mask); plt.show(); plt.close()
    
    #Determine the ring
    cytoplasm_ring[expanded_nucleus_mask>0] = 0
        # plt.imshow(cytoplasm_ring); plt.show(); plt.close()
    
    return cytoplasm_ring

# Function to measure intensity for each labeled cell
def measure_intensity_per_cell(image, labeled_mask):
    '''
    Determine labeled mask and properties relating to the mask and 
    the intensity image.
    '''
    #Determine properties (labeled areas should be the cells)
    properties = regionprops(labeled_mask, intensity_image=image)
    
    # Extract intensities for each cell
    intensities = [prop.mean_intensity for prop in properties]
     
    return intensities

# Function to save all intensities and ratios to a single CSV file
def save_all_intensities_to_csv(data, output_folder):
    '''
    'data' is a list of lists, where the outer lists corresponds to the cells,
    and the inner lists contains multiple entries with properties and information 
    about the cell. This data is written to a csv file at the location 'output_csv'.
    all_data (usually) contains data from multiple image stacks.
    '''
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create if it doesn't exist
    
    # Define a filename inside the output folder
    csv_filename = os.path.join(output_folder, "intensity_results.csv")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename+CellID", "Condition", "Nucleus/Cytoplasm Ratio", "Nucleus Intensity", "Cytoplasmic Intensity"])
        for row in data:
            writer.writerow(row)

# Function to extract the condition from the filename
def extract_condition(filename):
    '''
    Apply regexp search to filename to classify the condition as either WT or RQ.
    '''
    
    match = re.search(r'-(WT)-|-(RQ)_', filename)
    return match.group(1) or match.group(2) if match else "Unknown"


######################################################################

# Some important tuning parameters
MIN_SIZE = 200 # threshold below which identified nuclei regions are considered small and discarded
DILATION_RADIUS = 5
DISTANCE_FROM_NUCLEUS = 2
PLOT_BACKGROUND_IMG = True

# Directory containing image stacks
if len(sys.argv) > 2:
    input_folder = sys.argv[1]  # First argument is input folder
    output_folder = sys.argv[2]  # Second argument is output folder
else:
    raise ValueError("Usage: python your_script.py <input_folder> <output_folder>")

# For debugging purposes
# input_folder = '/Users/m.wehrens/Data_UVA/2024_10_Sebastian-KTR/static-example/tiff_input/'
# output_folder = '/Users/m.wehrens/Data_UVA/2024_10_Sebastian-KTR/static-example/output/'
# input_folder = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/DATA/'
# output_folder = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'

all_data = []

# Loop through each image stack in the folder
for file_path in os.listdir(input_folder):
    # file_path = os.listdir(input_folder)[0]
    # file_path = "Stack_hDMECs_mtq-gaq-RQ_NFAT1_007.tif"
    
    if file_path.endswith(".tif"):  # Adjust file extension if needed
        
        #Read image stack
        image_stack = tiff.imread(os.path.join(input_folder, file_path))
        print(f"Processing file: {file_path}")
        
        ### Image processing
        
        image_name = os.path.splitext(file_path)[0]
        # Extract the condition from the filename
        condition = extract_condition(file_path)
        
        # Put the nucleus and data channels in two seperate variables
        nucleus_channel = (image_stack[0])  # Assuming the first image is the nucleus channel
        data_channel = (image_stack[1])  # Assuming the second image is the data channel
        
        # Apply median filter to the nucleus channel, increase radius to increase smoothing
        nucleus_channel_filtered = apply_median_filter(nucleus_channel, radius=2)
            # plt.imshow(nucleus_channel_filtered); plt.show(); plt.close()
        
        # Subtract mode background from the data channel
        data_channel_bg_subtracted, mode_value = subtract_mode_background(data_channel)
            # plt.imshow(data_channel_bg_subtracted); plt.show(); plt.close()
        
        if PLOT_BACKGROUND_IMG:
            # Visualize the background mode for the data channel
            visualize_background_mode(data_channel, mode_value, image_name)
        
        # Debugging: show resulting images
        # plt.imshow(nucleus_channel_filtered); plt.show(); plt.close()
        # plt.imshow(data_channel_bg_subtracted); plt.show(); plt.close()
        
        # Segment the nucleus and create cytoplasmic ROI
        nucleus_mask = segment_nucleus(nucleus_channel_filtered, min_size=MIN_SIZE)
        cytoplasm_roi = create_cytoplasm_roi(nucleus_mask, dilation_radius=DILATION_RADIUS, distance_from_nucleus= DISTANCE_FROM_NUCLEUS)
            # plt.imshow(nucleus_mask); plt.show(); plt.close()
            # plt.imshow(cytoplasm_roi); plt.show(); plt.close()
        
        #Optional visualization
        # Visualize nucleus segmentation and cytoplasmic ring
        # visualize_nucleus_segmentation(nucleus_channel_filtered, nucleus_mask,image_name)
        # visualize_cytoplasmic_ring(nucleus_channel_filtered, cytoplasm_roi,image_name)
        
        ### Plotting
        
        # Visualize with cell IDs for choosing interesting cells
        visualize_nucleus_with_ids(nucleus_channel_filtered, nucleus_mask, image_name)
        
        # Visualize cytoplasmic ring overlay on data channel image
        visualize_cytoplasmic_ring_overlay(data_channel_bg_subtracted, cytoplasm_roi, image_name)
        
        ### Determine mean intensities for each cytoplasmic and nuclear region
        
        # Measure intensities for each cell in the nucleus and cytoplasm
        nucleus_intensities = measure_intensity_per_cell(data_channel_bg_subtracted, nucleus_mask)
        cytoplasm_intensities = measure_intensity_per_cell(data_channel_bg_subtracted, cytoplasm_roi)
        
        ### Create output data structure
        # all_data is a list of lists, where the outer lists corresponds to the cells,
        # and the inner lists contains multiple entries with properties and information.
        # all_data contains data from multiple image stacks.
        
        # Combine data for each cell
        for cell_id, (nucleus_intensity, cytoplasm_intensity) in enumerate(zip(nucleus_intensities, cytoplasm_intensities), start=1):
            ratio = nucleus_intensity / cytoplasm_intensity if cytoplasm_intensity != 0 else np.nan
            all_data.append([file_path + f"_{cell_id}", condition, ratio, nucleus_intensity, cytoplasm_intensity])

# Save all combined data to a single CSV file
save_all_intensities_to_csv(all_data, output_folder)
print(f"All results saved to {output_folder}")


################################################################################

if False:
    
    '''
    # Now make a plot of the two conditions using seaborn
    import seaborn as sns
    import pandas as pd
    df_all_data = pd.DataFrame(all_data, columns=["Filename+CellID", "Condition", "Nucleus/Cytoplasm Ratio", "Nucleus Intensity", "Cytoplasmic Intensity"])
    sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_all_data, jitter=True)
    plt.show()
    '''
    
    