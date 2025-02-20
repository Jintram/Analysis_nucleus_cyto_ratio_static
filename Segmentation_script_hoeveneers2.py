import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk, binary_dilation
from skimage.measure import label, regionprops
import os
import csv
from skimage.color import label2rgb
from skimage import exposure, img_as_float
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
import re  # Import regular expressions module
from scipy import stats
from skimage.filters import median
from skimage.morphology import disk

# Enable interactive mode
plt.ion


# Function to visualize nucleus segmentation
def visualize_nucleus_segmentation(image, mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Segmented Nucleus")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.show(block=False)
    plt.waitforbuttonpress()

# Function to visualize cytoplasmic ring creation
def visualize_cytoplasmic_ring(image, cytoplasmic_ring):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Cytoplasmic Ring")
    plt.imshow(cytoplasmic_ring, cmap="gray")
    plt.axis("off")
    plt.show(block=False)
    plt.waitforbuttonpress()

# Function to visualize nucleus segmentation with masks and cell IDs
def visualize_nucleus_with_ids(image, mask):
    labeled_mask = label(mask)
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
    plt.show(block=False)
    plt.waitforbuttonpress()

# Function to visualize cytoplasmic ring overlayed on the cyan channel image
def visualize_cytoplasmic_ring_overlay(cyan_channel, cytoplasmic_ring,vmin=0, vmax=255):
    plt.figure(figsize=(12, 6))
    plt.title("Cytoplasmic Ring Overlay on Cyan Channel")
    plt.imshow(cyan_channel, cmap="gray",vmin=vmin, vmax=vmax)  
    plt.imshow(np.ma.masked_where(cytoplasmic_ring == 0, cytoplasmic_ring), cmap="gray", alpha=0.7)  # Overlay with white rings
    plt.axis("off")
    plt.show(block=False)
    plt.waitforbuttonpress()

# Function to subtract mode background from an image
def subtract_mode_background(image):
    # Flatten image to 1D for mode calculation
    flat_image = image.flatten()

    # Convert to integer if needed
    if not np.issubdtype(flat_image.dtype, np.integer):
        flat_image = flat_image.astype(int)

    # Compute mode using bincount
    mode_value = np.bincount(flat_image).argmax()

    # Subtract mode value
    background_subtracted_image = image.astype(np.int16) - mode_value  # Use int16 to prevent underflow

    # Ensure valid pixel range without clipping to white
    background_subtracted_image[background_subtracted_image < 0] = 0  # Set negative values to 0

    return background_subtracted_image.astype(np.uint8)

# Function to apply median filter to an image
def apply_median_filter(image, radius=2):
    # Apply median filter with a specified radius
    selem = disk(radius)
    filtered_image = median(image, selem)
    return filtered_image

# Function to segment nucleus
def segment_nucleus(image):
    thresh = threshold_otsu(image)
    binary_mask = image > thresh
    clean_mask = remove_small_objects(binary_mask, min_size=200)
    clean_mask = remove_small_holes(clean_mask, area_threshold=10)
    opened_mask = binary_opening(clean_mask, footprint=disk(2))
    return opened_mask

# Function to create cytoplasm ROI
def create_cytoplasm_roi(nucleus_mask, dilation_radius=10, distance_from_nucleus=5):
    expanded_nucleus_mask = binary_dilation(nucleus_mask, footprint=disk(distance_from_nucleus))
    dilated_mask = binary_dilation(expanded_nucleus_mask, footprint=disk(dilation_radius))
    cytoplasm_ring = dilated_mask ^ expanded_nucleus_mask
    return cytoplasm_ring

# Function to measure intensity for each labeled cell
def measure_intensity_per_cell(image, mask):
    labeled_mask = label(mask)
    properties = regionprops(labeled_mask, intensity_image=image)
    intensities = [prop.mean_intensity for prop in properties]
    return intensities

# Function to save all intensities and ratios to a single CSV file
def save_all_intensities_to_csv(data, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename+CellID", "Condition", "Nucleus/Cytoplasm Ratio", "Nucleus Intensity", "Cytoplasmic Intensity"])
        for row in data:
            writer.writerow(row)

# Directory containing image stacks
if len(sys.argv) > 2:
    input_folder = sys.argv[1]  # First argument is input folder
    output_folder = sys.argv[2]  # Second argument is output folder
else:
    raise ValueError("Usage: python your_script.py <input_folder> <output_folder>")

all_data = []

# Function to extract the condition from the filename
def extract_condition(filename):
    match = re.search(r'-(WT)-|-(RQ)_', filename)
    return match.group(1) or match.group(2) if match else "Unknown"


# Loop through each image stack in the folder
for file_path in os.listdir(input_folder):
    if file_path.endswith(".tif"):  # Adjust file extension if needed
        image_stack = tiff.imread(os.path.join(input_folder, file_path))
        print(f"Processing file: {file_path}")

        # Extract the condition from the filename
        condition = extract_condition(file_path)

        # Preprocess the nucleus and cyan channels
        nucleus_channel = (image_stack[0])  # Assuming the first image is the nucleus channel
        cyan_channel = (image_stack[1])  # Assuming the second image is the cyan channel

        # Apply median filter to the nucleus channel, increase radius to increase smoothing
        nucleus_channel_filtered = apply_median_filter(nucleus_channel, radius=2)

        # Subtract mode background from the cyan channel
        cyan_channel_bg_subtracted = subtract_mode_background(cyan_channel)

        # Segment the nucleus and create cytoplasmic ROI
        nucleus_mask = segment_nucleus(nucleus_channel_filtered)
        cytoplasm_roi = create_cytoplasm_roi(nucleus_mask, dilation_radius=10, distance_from_nucleus=5)

        # Visualize nucleus segmentation and cytoplasmic ring
        # visualize_nucleus_segmentation(nucleus_channel_filtered, nucleus_mask)
        # visualize_cytoplasmic_ring(nucleus_channel_filtered, cytoplasm_roi)


   
        # Visualize with cell IDs for choosing interesting cells
        visualize_nucleus_with_ids(nucleus_channel_filtered, nucleus_mask)
        
        # Visualize cytoplasmic ring overlay on cyan channel image
        visualize_cytoplasmic_ring_overlay(cyan_channel_bg_subtracted, cytoplasm_roi)

        # Measure intensities for each cell in the nucleus and cytoplasm
        nucleus_intensities = measure_intensity_per_cell(cyan_channel_bg_subtracted, nucleus_mask)
        cytoplasm_intensities = measure_intensity_per_cell(cyan_channel_bg_subtracted, cytoplasm_roi)

        # Combine data for each cell
        for cell_id, (nucleus_intensity, cytoplasm_intensity) in enumerate(zip(nucleus_intensities, cytoplasm_intensities)):
            ratio = nucleus_intensity / cytoplasm_intensity if cytoplasm_intensity != 0 else np.nan
            all_data.append([file_path + f"_{cell_id}", condition, ratio, nucleus_intensity, cytoplasm_intensity])

# Save all combined data to a single CSV file
save_all_intensities_to_csv(all_data, output_csv)
print(f"All results saved to {output_csv}")
