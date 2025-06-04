# coding: utf-8
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
PLOT_BACKGROUND_IMG
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
input_folder = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/DATA/'
output_folder = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'
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
all_data
import seaborn as sns
import pandas as pd
df_all_data = pd.DataFrame(all_data, columns=["Filename+CellID", "Condition", "Nucleus/Cytoplasm Ratio", "Nucleus Intensity", "Cytoplasmic Intensity"])
df_all_data
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=all_data, jitter=True)
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_all_data, jitter=True)
plt.show()
if False:
    '''
    # Now make a plot of the two conditions using seaborn
    import seaborn as sns
    import pandas as pd
    df_all_data = pd.DataFrame(all_data, columns=["Filename+CellID", "Condition", "Nucleus/Cytoplasm Ratio", "Nucleus Intensity", "Cytoplasmic Intensity"])
    sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_all_data, jitter=True)
    plt.show()
    '''
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

nucleus_intensities, nucleus_areas = measure_intensity_per_cell(data_channel_bg_subtracted, nucleus_mask)
cytoplasm_intensities, cytoplasm_areas = measure_intensity_per_cell(data_channel_bg_subtracted, cytoplasm_roi)
### Create output data structure
# all_data is a list of lists, where the outer lists corresponds to the cells,
# and the inner lists contains multiple entries with properties and information.
# all_data contains data from multiple image stacks.
def measure_intensity_per_cell(image, labeled_mask):
    '''
    Determine labeled mask and properties relating to the mask and 
    the intensity image.
    '''
    #Determine properties (labeled areas should be the cells)
    properties = regionprops(labeled_mask, intensity_image=image)
    # Extract intensities for each cell
    intensities = [prop.mean_intensity for prop in properties]
    areas       = [prop.area for prop in properties]
    return intensities, areas
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

nucleus_intensities, nucleus_areas = measure_intensity_per_cell(data_channel_bg_subtracted, nucleus_mask)
cytoplasm_intensities, cytoplasm_areas = measure_intensity_per_cell(data_channel_bg_subtracted, cytoplasm_roi)
### Create output data structure
# all_data is a list of lists, where the outer lists corresponds to the cells,
# and the inner lists contains multiple entries with properties and information.
# all_data contains data from multiple image stacks.
all_data = []
all_data2 = []
np.array([1,2,3])/np.array([1,2,0])
bla = np.array([ 1.,  1., inf])
bla = np.array([ 1.,  1., np.inf])
bla
bla.isinf()
bla == np.inf
np.isinf(bla)
ratios = [N / C if C != 0 else np.nan for N, C in zip(nucleus_intensities, cytoplasm_intensities)]
ratios
cell_ids = list(range(len(nucleus_intensities)))
cell_ids
for cell_id, (nucleus_intensity, cytoplasm_intensity, nucleus_area, cytoplasm_area) in enumerate(zip(nucleus_intensities, cytoplasm_intensities, nucleus_areas, cytoplasm_areas), start=1):
    ratio = nucleus_intensity / cytoplasm_intensity if cytoplasm_intensity != 0 else np.nan
    all_data.append([file_path + f"_{cell_id}", condition, ratio, nucleus_intensity, cytoplasm_intensity])
all_data
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

all_data  = []
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
        nucleus_intensities, nucleus_areas = measure_intensity_per_cell(data_channel_bg_subtracted, nucleus_mask)
        cytoplasm_intensities, cytoplasm_areas = measure_intensity_per_cell(data_channel_bg_subtracted, cytoplasm_roi)
        ### Create output data structure
        # all_data is a list of lists, where the outer lists corresponds to the cells,
        # and the inner lists contains multiple entries with properties and information.
        # all_data contains data from multiple image stacks.
        # Combine data for each cell
        for cell_id, (nucleus_intensity, cytoplasm_intensity, nucleus_area, cytoplasm_area) in enumerate(zip(nucleus_intensities, cytoplasm_intensities, nucleus_areas, cytoplasm_areas), start=1):
            ratio = nucleus_intensity / cytoplasm_intensity if cytoplasm_intensity != 0 else np.nan
            all_data.append([file_path + f"_{cell_id}", condition, ratio, nucleus_intensity, cytoplasm_intensity, nucleus_area, cytoplasm_area])
# Save all combined data to a single CSV file

save_all_intensities_to_csv(all_data, output_folder)
print(f"All results saved to {output_folder}")
MIN_SIZE = 200 # threshold below which identified nuclei regions are considered small and discarded
DILATION_RADIUS = 5
DISTANCE_FROM_NUCLEUS = 2
PLOT_BACKGROUND_IMG = True
all_data  = []
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
        nucleus_intensities, nucleus_areas = measure_intensity_per_cell(data_channel_bg_subtracted, nucleus_mask)
        cytoplasm_intensities, cytoplasm_areas = measure_intensity_per_cell(data_channel_bg_subtracted, cytoplasm_roi)
        ### Create output data structure
        # all_data is a list of lists, where the outer lists corresponds to the cells,
        # and the inner lists contains multiple entries with properties and information.
        # all_data contains data from multiple image stacks.
        # Combine data for each cell
        for cell_id, (nucleus_intensity, cytoplasm_intensity, nucleus_area, cytoplasm_area) in enumerate(zip(nucleus_intensities, cytoplasm_intensities, nucleus_areas, cytoplasm_areas), start=1):
            ratio = nucleus_intensity / cytoplasm_intensity if cytoplasm_intensity != 0 else np.nan
            all_data.append([file_path + f"_{cell_id}", condition, ratio, nucleus_intensity, cytoplasm_intensity, nucleus_area, cytoplasm_area])
# Save all combined data to a single CSV file

save_all_intensities_to_csv(all_data, output_folder)
print(f"All results saved to {output_folder}")
image = nucleus_channel_filtered; labeled_mask= nucleus_mask; filename= image_name
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask, levels=0.5, colors='red', linewidths=0.5)
labeled_mask
ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask, levels=0.5, colors='red', linewidths=0.5)
ax[0].contour(labeled_mask>0, levels=0.5, colors='red', linewidths=0.5)
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
plt.show()
image_name
file_path = os.listdir(input_folder)[0]
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

nucleus_intensities, nucleus_areas = measure_intensity_per_cell(data_channel_bg_subtracted, nucleus_mask)
cytoplasm_intensities, cytoplasm_areas = measure_intensity_per_cell(data_channel_bg_subtracted, cytoplasm_roi)
### Create output data structure
# all_data is a list of lists, where the outer lists corresponds to the cells,
# and the inner lists contains multiple entries with properties and information.
# all_data contains data from multiple image stacks.
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
plt.show(); plt.close()
image = nucleus_channel_filtered; labeled_mask= nucleus_mask; filename= image_name
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
plt.show(); plt.close()
import math
50**2 * math.piu
50**2 * math.pi
25**2 * math.pi
12.5**2 * math.pi
MIN_SIZE = 500 
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

nucleus_intensities, nucleus_areas = measure_intensity_per_cell(data_channel_bg_subtracted, nucleus_mask)
cytoplasm_intensities, cytoplasm_areas = measure_intensity_per_cell(data_channel_bg_subtracted, cytoplasm_roi)
### Create output data structure
# all_data is a list of lists, where the outer lists corresponds to the cells,
# and the inner lists contains multiple entries with properties and information.
# all_data contains data from multiple image stacks.
image = nucleus_channel_filtered; labeled_mask= nucleus_mask; filename= image_name
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(overlay)
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Add cell IDs to the plot

properties = regionprops(labeled_mask)
for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# plt.show()
overlay = label2rgb(labeled_mask, image=image, bg_label=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(overlay)
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Add cell IDs to the plot

properties = regionprops(labeled_mask)
for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
plt.show(); plt.close()
np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)
np.unique(np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi))
overlay = label2rgb(labeled_mask, image=image, bg_label=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(overlay)
# Add cell IDs to the plot

properties = regionprops(labeled_mask)
for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)
    ax[1].imshow(cytoplasm_overlay)
# plt.show(); plt.close()
plt.show(); plt.close()
overlay = label2rgb(labeled_mask, image=image, bg_label=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(overlay)
# Add cell IDs to the plot

properties = regionprops(labeled_mask)
for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = label2rgb(np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi))
    ax[1].imshow(cytoplasm_overlay)
# plt.show(); plt.close()
plt.show(); plt.close()
label2rgb(np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi))
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask)
# Add cell IDs to the plot

properties = regionprops(labeled_mask)
for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)
    ax[1].imshow(cytoplasm_overlay)
# plt.show(); plt.close()
plt.show(); plt.close()
jet = cm.get_cmap('jet', 256)
colors = jet(np.arange(256))
np.random.shuffle(colors)
colors[0] = [0, 0, 0, 1]  # Set the first color to black
mycolormap = ListedColormap(colors)
from matplotlib import cm
from matplotlib.colors import ListedColormap
jet = cm.get_cmap('jet', 256)
colors = jet(np.arange(256))
np.random.shuffle(colors)
colors[0] = [0, 0, 0, 1]  # Set the first color to black
mycolormap = ListedColormap(colors)
mycolormap
jet = cm.get_cmap('jet', 256)
colors = jet(np.arange(256))
np.random.shuffle(colors)
colors[0] = [0, 0, 0, 1]  # Set the first color to black
mycolormap = ListedColormap(colors)
jet = cm.get_cmap('jet', 256)
jet = plt.get_cmap('jet')
colors = jet(np.arange(256))
np.random.shuffle(colors)
colors[0] = [0, 0, 0, 1]  # Set the first color to black
mycolormap = ListedColormap(colors)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask, cmap=mycolormap)
# Add cell IDs to the plot

properties = regionprops(labeled_mask)
for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)
    ax[1].imshow(cytoplasm_overlay, cmap=mycolormap)
# plt.show(); plt.close()
plt.show(); plt.close()
prop.bbox[:2]
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask, cmap=mycolormap)
# Add cell IDs to the plot

properties = regionprops(labeled_mask)
for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)        
    ax[1].imshow(cytoplasm_overlay, cmap=mycolormap)
    properties = regionprops(labeled_mask)
    for prop in properties:            
        y, x = prop.bbox[:2] # lefttop
        ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
# plt.show(); plt.close()
plt.show(); plt.close()
np.max(cytoplasm_overlay)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask, cmap=mycolormap)
# Add cell IDs to the plot

properties = regionprops(labeled_mask)
for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)        
    ax[1].imshow(cytoplasm_overlay, cmap=mycolormap, vmin=0, vmax=np.max(cytoplasm_overlay))
    properties = regionprops(labeled_mask)
    for prop in properties:            
        y, x = prop.bbox[:2] # lefttop
        ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
# plt.show(); plt.close()
plt.show(); plt.close()
roundness = [np.pi*4*prop.area / prop.perimeter**2 for prop in properties] 
roundness
DATAFILE_EXCEL = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/intensity_results_manual.xlsx'
pd.read_excel(DATAFILE_EXCEL)
df_alldata = pd.read_excel(DATAFILE_EXCEL)
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata, jitter=True)
plt.show()
df_alldata.keys()
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True)
plt.show()
df_alldata['Discard']
df_alldata['Discard'].isna()
df_alldata['Discard'][df_alldata['Discard'].isna()] = 'No'
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True)
plt.show()
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'
EXCEL_FILE  = 'intensity_results_manual.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata['Discard'][df_alldata['Discard'].isna()] = 'No'
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'No'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'No'
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True)
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata)
plt.show()
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True)
plt.show()
df_alldata['Discard']!='Yes'
df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='Yes',]
df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='Yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata, jitter=True)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel)
plt.show()
plt.close()
df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='Yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel)
plt.show()
plt.close()
np.unique(df_alldata['Discard'])
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel)
plt.show()
plt.close()
df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey")
plt.show()
plt.close()
df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black')
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey")
plt.show()
plt.close()
df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black')
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey")
plt.show()
plt.close()
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'
EXCEL_FILE  = 'intensity_results_manual.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1])
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.show()
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1])
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.show()
plt.close()
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'
EXCEL_FILE  = 'intensity_results_manual.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1])
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
# plt.show()

plt.tight_layout()
# save fig

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
cm_to_inch = 1 / 2.54
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/Analysis_20250602/'
EXCEL_FILE  = 'intensity_results_manual.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1])
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
# plt.show()

plt.tight_layout()
# save fig

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
plt.rcParams.update({'font.size': 8}) # set all font sizes to 8
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
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1])
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1], size=5)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1)
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
properties = regionprops(labeled_mask)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    y_rightbound = prop.bbox[2]
    ax[0].text(x, y_rightbound, str(prop.label), color='red', fontsize=12, ha='center', va='center')
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask, cmap=mycolormap)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)        
    ax[1].imshow(cytoplasm_overlay, cmap=mycolormap, vmin=0, vmax=np.max(cytoplasm_overlay))
    # Add extra label labeling the cytoplasm regions
    # properties = regionprops(labeled_mask)
    # for prop in properties:            
    #     y, x = prop.bbox[:2] # lefttop
    #     ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
plt.show(); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    x_rightbound = prop.bbox[3]
    ax[0].text(x_rightbound, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask, cmap=mycolormap)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)        
    ax[1].imshow(cytoplasm_overlay, cmap=mycolormap, vmin=0, vmax=np.max(cytoplasm_overlay))
    # Add extra label labeling the cytoplasm regions
    # properties = regionprops(labeled_mask)
    # for prop in properties:            
    #     y, x = prop.bbox[:2] # lefttop
    #     ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
# plt.show(); plt.close()
plt.show(); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    x_rightbound = prop.bbox[3]
    ax[0].text(x_rightbound, y, str(prop.label), color='red', fontsize=12, ha='left', va='center')
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask, cmap=mycolormap)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)        
    ax[1].imshow(cytoplasm_overlay, cmap=mycolormap, vmin=0, vmax=np.max(cytoplasm_overlay))
    # Add extra label labeling the cytoplasm regions
    # properties = regionprops(labeled_mask)
    # for prop in properties:            
    #     y, x = prop.bbox[:2] # lefttop
    #     ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
# plt.show(); plt.close()
plt.show(); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    x_rightbound = prop.bbox[3]
    ax[0].text(x_rightbound, y, str(prop.label), color='red', fontsize=12, ha='left', va='center')
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask, cmap=mycolormap)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    x_rightbound = prop.bbox[3]
    ax[1].text(x_rightbound, y, str(prop.label), color='red', fontsize=12, ha='left', va='center')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)        
    ax[1].imshow(cytoplasm_overlay, cmap=mycolormap, vmin=0, vmax=np.max(cytoplasm_overlay))
    # Add extra label labeling the cytoplasm regions
    # properties = regionprops(labeled_mask)
    # for prop in properties:            
    #     y, x = prop.bbox[:2] # lefttop
    #     ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
# plt.show(); plt.close()
plt.show(); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Nucleus Segmentation with Cell IDs")
# Segmentation on top of nuclei

ax[0].imshow(image, cmap="gray")
ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    y_top = prop.bbox[0]
    # x_rightbound = prop.bbox[3]
    ax[0].text(x, y_top, str(prop.label), color='red', fontsize=12, ha='center', va='bottom')
# plt.show(); plt.close()
# Segmentation with IDs

ax[1].imshow(labeled_mask, cmap=mycolormap)
# Add cell IDs to the plot

for prop in properties:
    y, x = prop.centroid
    y_top = prop.bbox[0]
    # x_rightbound = prop.bbox[3]
    ax[1].text(x, y_top, str(prop.label), color='red', fontsize=12, ha='center', va='bottom')

ax[1].axis("off")
# Plot cytoplasm_roi if given

if cytoplasm_roi is not None:
    cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)        
    ax[1].imshow(cytoplasm_overlay, cmap=mycolormap, vmin=0, vmax=np.max(cytoplasm_overlay))
    # Add extra label labeling the cytoplasm regions
    # properties = regionprops(labeled_mask)
    # for prop in properties:            
    #     y, x = prop.bbox[:2] # lefttop
    #     ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
plt.show(); plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1, colors=['grey','black'])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1, color=['grey','black'])
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'grey', 'no': 'black'})
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
# Plot 1, showing which outliers are removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'black', 'no': 'grey'})
# Plot 2, outliers removed

df_alldata_sel = df_alldata.loc[df_alldata['Discard']!='yes',]
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_sel, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]
# Plot 1, showing which outliers are removed
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'black', 'no': 'grey'})

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, label='kept', color='grey')
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, label='discarded', color='black')
# Plot 2, outliers removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]
# Plot 1, showing which outliers are removed
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'black', 'no': 'grey'})

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, label='kept', color='grey')
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, label='discarded', color='black')
# Plot 2, outliers removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]
# Plot 1, showing which outliers are removed
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'black', 'no': 'grey'})

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, label='kept', color='grey')
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, label='discarded', color='black')
# Plot 2, outliers removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 6*cm_to_inch))
df_alldata_kept = df_alldata.loc[df_alldata['Discard']!='yes',]
df_alldata_discarded = df_alldata.loc[df_alldata['Discard']=='yes',]
# Plot 1, showing which outliers are removed
# sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", hue='Discard', data=df_alldata, jitter=True, ax=axs[0], size=1.5, palette={'yes': 'black', 'no': 'grey'})

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, ax=axs[0], size=1.5, color='grey')
sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_discarded, jitter=True, ax=axs[0], size=1.5, color='black')
axs[0].legend(['kept', 'discarded'], loc='best')
# Plot 2, outliers removed

sns.stripplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, jitter=True, color='black', ax=axs[1], size=1)
sns.violinplot(x="Condition", y="Nucleus/Cytoplasm Ratio", data=df_alldata_kept, color="grey", ax=axs[1])
plt.tight_layout()
# save fig
# plt.show()

plt.savefig(OUTPUT_DIR + 'intensity_results_manual.pdf', dpi=600, bbox_inches='tight')
plt.close()
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
import glob
output_folder = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/NFAT1/2.NFAT1_shDSCR1'
input_folder = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/DATA_ALL/NFAT_analysis_Martijn/NFAT1/2.NFAT1_shDSCR1'
os.makedirs(output_folder, exist_ok=True)
all_data  = []
tif_files = glob.glob(os.path.join(input_folder, "**/*.tif"), recursive=True)
tif_files
tif_files
image_stack = tiff.imread(os.path.join(input_folder, file_path))
tif_files     = [os.path.basename(file_path) for file_path in tif_filepaths] 
tif_filepaths = glob.glob(os.path.join(input_folder, "**/*.tif"), recursive=True)
# file basename

tif_files     = [os.path.basename(file_path) for file_path in tif_filepaths] 
tif_files
tif_filepaths = glob.glob(os.path.join(input_folder, "**/*.tif"), recursive=True)
# file basename

tif_filenames = [os.path.basename(file_path) for file_path in tif_filepaths] 
file_path = tif_filepaths[0]; filename = tif_filenames[0]
image_stack = tiff.imread(file_path)
print(f"Processing file: {filename}")
image_name = os.path.splitext(filename)[0]
condition = extract_condition(filename)
nucleus_channel = (image_stack[0])  # Assuming the first image is the nucleus channel
data_channel = (image_stack[1])  # Assuming the second image is the data channel
nucleus_channel_filtered = apply_median_filter(nucleus_channel, radius=2)
data_channel_bg_subtracted, mode_value = subtract_mode_background(data_channel)
PLOT_BACKGROUND_IMG
visualize_background_mode(data_channel, mode_value, image_name)
nucleus_mask = segment_nucleus(nucleus_channel_filtered, min_size=MIN_SIZE)
cytoplasm_roi = create_cytoplasm_roi(nucleus_mask, dilation_radius=DILATION_RADIUS, distance_from_nucleus= DISTANCE_FROM_NUCLEUS)
visualize_nucleus_with_ids(nucleus_channel_filtered, nucleus_mask, image_name, cytoplasm_roi=cytoplasm_roi)
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
# Create a shuffled color map of jet, starting with color black  

jet = plt.get_cmap('jet')
colors = jet(np.arange(256))
np.random.shuffle(colors)
colors[0] = [0, 0, 0, 1]  # Set the first color to black
mycolormap = ListedColormap(colors)
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

def visualize_nucleus_with_ids(image, labeled_mask, filename, cytoplasm_roi=None):
    # image = nucleus_channel_filtered; labeled_mask= nucleus_mask; filename= image_name
    # np.unique(labeled_mask, return_counts=True)
    # overlay = label2rgb(labeled_mask, image=image, bg_label=0)
    properties = regionprops(labeled_mask)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle("Nucleus Segmentation with Cell IDs")
    # Segmentation on top of nuclei
    ax[0].imshow(image, cmap="gray")
    ax[0].contour(labeled_mask>0, colors='red', linewidths=0.5)
    # Add cell IDs to the plot
    for prop in properties:
        y, x = prop.centroid
        y_top = prop.bbox[0]
        # x_rightbound = prop.bbox[3]
        ax[0].text(x, y_top, str(prop.label), color='red', fontsize=12, ha='center', va='bottom')
    # plt.show(); plt.close()
    # Segmentation with IDs
    ax[1].imshow(labeled_mask, cmap=mycolormap)
    # Add cell IDs to the plot
    for prop in properties:
        y, x = prop.centroid
        # y_top = prop.bbox[0]
        # x_rightbound = prop.bbox[3]
        ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
    ax[1].axis("off")
    # Plot cytoplasm_roi if given
    if cytoplasm_roi is not None:
        cytoplasm_overlay = np.ma.masked_where(cytoplasm_roi == 0, cytoplasm_roi)        
        ax[1].imshow(cytoplasm_overlay, cmap=mycolormap, vmin=0, vmax=np.max(cytoplasm_overlay))
        # Add extra label labeling the cytoplasm regions
        # properties = regionprops(labeled_mask)
        # for prop in properties:            
        #     y, x = prop.bbox[:2] # lefttop
        #     ax[1].text(x, y, str(prop.label), color='red', fontsize=12, ha='center', va='center')
    # plt.show(); plt.close()
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
    #plt.imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
    #          cmap="gray", alpha=0.7)
    plt.imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
            cmap="OrRd", alpha=0.9, vmin=0, vmax=1)
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
    areas       = [prop.area for prop in properties]
    # Also extract roundness
    roundness = [np.pi*4*prop.area / prop.perimeter**2 for prop in properties] 
    return intensities, areas, roundness
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
        writer.writerow(["Filename+CellID", "Condition", "Nucleus/Cytoplasm Ratio", "Nucleus Intensity", "Cytoplasmic Intensity","Nucleus area (px)","Cytoplasm ring area (px)","Roundness nucleus"])
        for row in data:
            writer.writerow(row)
# Function to extract the condition from the filename

def extract_condition(filename):
    '''
    Apply regexp search to filename to classify the condition as either WT or RQ.
    '''
    match = re.search(r'-(WT)-|-(RQ)_', filename)
    return match.group(1) or match.group(2) if match else "Unknown"
visualize_nucleus_with_ids(nucleus_channel_filtered, nucleus_mask, image_name, cytoplasm_roi=cytoplasm_roi)
visualize_cytoplasmic_ring_overlay(data_channel_bg_subtracted, cytoplasm_roi, image_name)
nucleus_intensities, nucleus_areas, nucleus_roundness = measure_intensity_per_cell(data_channel_bg_subtracted, nucleus_mask)
cytoplasm_intensities, cytoplasm_areas, _ = measure_intensity_per_cell(data_channel_bg_subtracted, cytoplasm_roi)
for cell_id, (nucleus_intensity, cytoplasm_intensity, nucleus_area, cytoplasm_area, nucleus_roundness_value) in enumerate(zip(nucleus_intensities, cytoplasm_intensities, nucleus_areas, cytoplasm_areas, nucleus_roundness), start=1):
    ratio = nucleus_intensity / cytoplasm_intensity if cytoplasm_intensity != 0 else np.nan
    all_data.append([filename + f"_{cell_id}", condition, ratio, nucleus_intensity, cytoplasm_intensity, nucleus_area, cytoplasm_area, nucleus_roundness_value])
image
labeled_mask
properties = regionprops(labeled_mask, intensity_image=image)
[np.median(prop.intensity_image[prop.coords[:, 0], prop.coords[:, 1]]) for prop in properties]
prop=properties[0]
prop.intensity_image
prop.coords
[np.median(image[prop.coords[:, 0], prop.coords[:, 1]]) for prop in properties]
[prop.mean_intensity for prop in properties]
cytoplasmic_ring = cytoplasmic_ring_labeled>0
cytoplasmic_ring_labeled
cytoplasmic_ring = cytoplasmic_ring_labeled>0
# Calculate dynamic range from the data channel

vmin, vmax = np.percentile(data_channel[data_channel > 0], (1, 99))
data_channel=data_channel_bg_subtracted; cytoplasmic_ring_labeled=cytoplasm_roi; filename=image_name
cytoplasmic_ring = cytoplasmic_ring_labeled>0
vmin, vmax = np.percentile(data_channel[data_channel > 0], (1, 99))
fig, axs = plt.subplots(1, 2,  figsize=(12, 6))
plt.suptitle("Cytoplasmic Ring Overlay on data Channel")
# Display data channel with dynamic range

axs[0].imshow(data_channel, cmap="gray", vmin=vmin, vmax=vmax)
# Overlay cytoplasmic ring
#plt.imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
#          cmap="gray", alpha=0.7)

axs[0].imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
        cmap="OrRd", alpha=0.9, vmin=0, vmax=1)

plt.axis("off")
# Same, but now with labels added

axs[1].imshow(data_channel, cmap="gray", vmin=vmin, vmax=vmax)
# Overlay cytoplasmic ring
#plt.imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
#          cmap="gray", alpha=0.7)

axs[1].imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
        cmap="OrRd", alpha=0.9, vmin=0, vmax=1)
# Add the labels

properties = regionprops(cytoplasmic_ring_labeled)
for prop in properties:
    y, x = prop.centroid
    y_top = prop.bbox[0]
    # x_rightbound = prop.bbox[3]
    axs[1].text(x, y_top, str(prop.label), color='red', fontsize=12, ha='center', va='bottom')

plt.axis("off")
plt.tight_layout()
# plt.show()
plt.show()
image=data_channel; filename=image_name
vmin, vmax = np.percentile(image[image > 0], (1, 99))
fig, ax = plt.subplots(1, 2, figsize=(12, 6))    
fig.suptitle("Background areas highlighted in yellow\n"+"Background should not overlap with cells")
# Create a mask for background pixels

background_mask = image <= mode_value
# Display the original image

ax[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
ax[1].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
# Overlay the background mask

ax[1].imshow(np.ma.masked_where(background_mask == 0, background_mask), 
           cmap="viridis", alpha=0.9, vmin=0, vmax=1)


ax[0].axis("off"); ax[1].axis("off")
plt.tight_layout()
# plt.show()
plt.show()
vmin, vmax = np.percentile(image[image > 0], (1, 99))
fig, ax = plt.subplots(1, 2, figsize=(12, 6))    
fig.suptitle("Background areas highlighted in yellow\n"+"Background should not overlap with cells")
# Create a mask for background pixels

background_mask = image <= mode_value
# Display the original image

ax[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
ax[1].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
# Overlay the background mask

ax[1].imshow(np.ma.masked_where(background_mask == 0, background_mask), 
           cmap="viridis", alpha=0.9, vmin=0, vmax=1)
# Now add the cytoplasmic rings if given

if cytoplasmic_ring_labeled is not None:
    cytoplasmic_ring = cytoplasmic_ring_labeled > 0
    ax[1].imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
                 cmap="OrRd", alpha=0.9, vmin=0, vmax=1)
    # Add the labels
    properties = regionprops(cytoplasmic_ring_labeled)
    for prop in properties:
        y, x = prop.centroid
        y_top = prop.bbox[0]
        ax[1].text(x, y_top, str(prop.label), color='red', fontsize=12, ha='center', va='bottom')
ax[0].axis("off"); ax[1].axis("off")
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))    
fig.suptitle("Background areas highlighted in yellow\n"+"Background should not overlap with cells")
# Create a mask for background pixels

background_mask = image <= mode_value
# Display the original image

ax[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
ax[1].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
# Overlay the background mask

ax[1].imshow(np.ma.masked_where(background_mask == 0, background_mask), 
           cmap="viridis", alpha=0.9, vmin=0, vmax=1)
# Now add the cytoplasmic rings if given

if cytoplasmic_ring_labeled is not None:
    cytoplasmic_ring = cytoplasmic_ring_labeled > 0
    ax[1].imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
                 cmap="OrRd", alpha=0.9, vmin=0, vmax=1)
    # Add the labels
    properties = regionprops(cytoplasmic_ring_labeled)
    for prop in properties:
        y, x = prop.centroid
        y_top = prop.bbox[0]
        ax[1].text(x, y_top, str(prop.label), color='#6e2179', fontsize=12, ha='center', va='bottom')


ax[0].axis("off"); ax[1].axis("off")
plt.tight_layout()
# plt.show()
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))    
fig.suptitle("Background areas highlighted in yellow\n"+"Background should not overlap with cells")
# Create a mask for background pixels

background_mask = image <= mode_value
# Display the original image

ax[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
ax[1].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
# Overlay the background mask

ax[1].imshow(np.ma.masked_where(background_mask == 0, background_mask), 
           cmap="viridis", alpha=0.9, vmin=0, vmax=1)
# Now add the cytoplasmic rings if given

if cytoplasmic_ring_labeled is not None:
    cytoplasmic_ring = cytoplasmic_ring_labeled > 0
    ax[1].imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
                 cmap="OrRd", alpha=0.9, vmin=0, vmax=1)
    # Add the labels
    properties = regionprops(cytoplasmic_ring_labeled)
    for prop in properties:
        y, x = prop.centroid
        y_top = prop.bbox[0]
        ax[1].text(x, y_top, f"**{prop.label}**", color='white', fontsize=12, ha='center', va='bottom')
        ax[1].text(x, y_top, str(prop.label), color='red', fontsize=12, ha='center', va='bottom')
ax[0].axis("off"); ax[1].axis("off")
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))    
fig.suptitle("Background areas highlighted in yellow\n"+"Background should not overlap with cells")
# Create a mask for background pixels

background_mask = image <= mode_value
# Display the original image

ax[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
ax[1].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
# Overlay the background mask

ax[1].imshow(np.ma.masked_where(background_mask == 0, background_mask), 
           cmap="viridis", alpha=0.9, vmin=0, vmax=1)
# Now add the cytoplasmic rings if given

if cytoplasmic_ring_labeled is not None:
    cytoplasmic_ring = cytoplasmic_ring_labeled > 0
    ax[1].imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
                 cmap="OrRd", alpha=0.9, vmin=0, vmax=1)
    # Add the labels
    properties = regionprops(cytoplasmic_ring_labeled)
    for prop in properties:
        y, x = prop.centroid
        y_top = prop.bbox[0]
        ax[1].text(x, y_top, r"**{prop.label}**", color='white', fontsize=12, ha='center', va='bottom')
        ax[1].text(x, y_top, str(prop.label), color='red', fontsize=12, ha='center', va='bottom')


ax[0].axis("off"); ax[1].axis("off")
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))    
fig.suptitle("Background areas highlighted in yellow\n"+"Background should not overlap with cells")
# Create a mask for background pixels

background_mask = image <= mode_value
# Display the original image

ax[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
ax[1].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
# Overlay the background mask

ax[1].imshow(np.ma.masked_where(background_mask == 0, background_mask), 
           cmap="viridis", alpha=0.9, vmin=0, vmax=1)
# Now add the cytoplasmic rings if given

if cytoplasmic_ring_labeled is not None:
    cytoplasmic_ring = cytoplasmic_ring_labeled > 0
    ax[1].imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), 
                 cmap="OrRd", alpha=0.9, vmin=0, vmax=1)
    # Add the labels
    properties = regionprops(cytoplasmic_ring_labeled)
    for prop in properties:
        y, x = prop.centroid
        y_top = prop.bbox[0]
        ax[1].text(x, y_top, str(prop.label), color='white', fontsize=13, ha='center', va='bottom')
        ax[1].text(x, y_top, str(prop.label), color='red', fontsize=12, ha='center', va='bottom')


ax[0].axis("off"); ax[1].axis("off")
plt.tight_layout()
# plt.show()
plt.show()
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/202506_Huveneers_Gaq_translocation/ANALYSIS_ALL_20250604/NFAT_analysis_Martijn/NFAT1/2.NFAT1_shDSCR1/'
EXCEL_FILE  = 'intensity_results_manual.xlsx'
df_alldata = pd.read_excel(OUTPUT_DIR+EXCEL_FILE)
df_alldata.loc[df_alldata['Discard'].isna(), 'Discard'] = 'no'
os.create(OUTPUT_DIR+'/plots/', exist_ok=True)
import os
os.makedirs(OUTPUT_DIR+'/plots/', exist_ok=True)
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
