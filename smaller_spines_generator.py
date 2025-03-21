import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile
import time
from PIL import Image
import json


def create_output_folder():
    """Create folder on desktop for output images"""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_folder = os.path.join(desktop, "spine_color_removal")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def analyze_tiff_file(input_path):
    """Analyze the TIFF file and return useful information"""
    try:
        with tifffile.TiffFile(input_path) as tif:
            num_pages = len(tif.pages)
            page_shapes = [page.shape for page in tif.pages]
            page_dtypes = [page.dtype for page in tif.pages]

            print(f"TIFF file has {num_pages} pages")
            print(f"Page shapes: {page_shapes[0]} (showing first page)")
            print(f"Page dtypes: {page_dtypes[0]} (showing first page)")

            # Check if we're dealing with a multi-channel file
            if len(page_shapes[0]) > 2:
                print(f"Multi-channel image detected with {page_shapes[0][-1]} channels")

            return num_pages, page_shapes, page_dtypes
    except Exception as e:
        print(f"Error analyzing TIFF file: {e}")
        return 0, [], []


def count_colors_across_pages(pages):
    """Count all unique non-zero values across all pages and their frequency"""
    color_counts = {}

    for page in pages:
        flat_page = page.flatten()
        unique_values, counts = np.unique(flat_page, return_counts=True)

        for value, count in zip(unique_values, counts):
            if value > 0:  # Skip background (0)
                if value in color_counts:
                    color_counts[value] += count
                else:
                    color_counts[value] = count

    # Sort colors by frequency (least frequent first)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
    return sorted_colors


def visualize_tiff_3d(pages, title="3D Visualization of TIFF Stack"):
    """
    Create a 3D visualization of the TIFF stack
    Uses a subset of slices for performance if there are many pages
    """
    # Determine how many slices to use (max 50 for performance)
    n_pages = len(pages)
    if n_pages > 50:
        step = n_pages // 50
        pages_subset = pages[::step]
    else:
        pages_subset = pages

    # Create a binary representation (non-zero values become 1)
    binary_pages = []
    for page in pages_subset:
        binary_page = np.zeros_like(page, dtype=bool)
        binary_page[page > 0] = True
        binary_pages.append(binary_page)

    # Create coordinates for non-zero points
    points = []
    for z, page in enumerate(binary_pages):
        y, x = np.where(page)
        points.extend([(x[i], y[i], z) for i in range(len(x))])

    if not points:
        print("No non-zero points found for 3D visualization")
        return

    # Convert to numpy array for easier slicing
    points = np.array(points)

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample points if there are too many
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points = points[indices]

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Slice)')
    ax.set_title(title)

    # Improve the view angle
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()


def extract_palette_from_tiff(input_path):
    """Extract palette information from input TIFF file"""
    try:
        with Image.open(input_path) as img:
            if img.mode == 'P':
                # Get the color palette
                palette = img.getpalette()
                # Extract RGB triplets from the palette
                num_colors = len(palette) // 3
                rgb_colors = [(palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]) for i in range(num_colors)]

                # Extract metadata if available
                metadata_dict = {}
                try:
                    raw_value = img.tag_v2.get(270)
                    if isinstance(raw_value, bytes):
                        raw_value = raw_value.decode('utf-8')
                    metadata_dict = json.loads(raw_value)
                except Exception as e:
                    print(f"Unable to read metadata: {e}")

                return palette, rgb_colors, metadata_dict
            else:
                print(f"Warning: Input file is not in P mode. Found mode: {img.mode}")
                return None, None, {}
    except Exception as e:
        print(f"Error extracting palette: {e}")
        return None, None, {}


def process_multipage_tiff(input_path, output_folder):
    """Process a multi-page TIFF file, progressively removing colors while maintaining P-mode structure"""
    try:
        # Extract palette information from the input file
        palette, rgb_colors, metadata_dict = extract_palette_from_tiff(input_path)

        if palette is None:
            print("Could not extract palette. The output may not be compatible with main.py.")

        # Read the full TIFF file with all metadata preserved
        with tifffile.TiffFile(input_path) as tif:
            # Get file name without extension
            file_name = os.path.splitext(os.path.basename(input_path))[0]

            # Load all pages with their original format
            pages = []
            for i, page in enumerate(tif.pages):
                print(f"Loading page {i + 1}/{len(tif.pages)}...")
                pages.append(page.asarray())

            # Count colors across all pages
            print("Counting colors across all pages...")
            sorted_color_counts = count_colors_across_pages(pages)
            colors_to_remove = [color for color, _ in sorted_color_counts]
            n_colors = len(colors_to_remove)

            print(f"Found {n_colors} unique colors across all pages")

            # Save original file as reference with .tiff extension
            original_path = os.path.join(output_folder, f"{file_name}_original.tiff")
            print(f"Saving original file to: {original_path}")

            # Open the original file and save it with PIL to preserve P-mode and palette
            if palette is not None:
                with Image.open(input_path) as img:
                    img.save(original_path, format='TIFF')
            else:
                # Fallback if we couldn't get the palette
                tifffile.imwrite(original_path, data=pages, bigtiff=True)

            # Display a middle slice for visualization
            middle_index = len(pages) // 2
            plt.figure(figsize=(10, 8))
            plt.imshow(pages[middle_index])
            plt.title(f"Original Image (Page {middle_index + 1} of {len(pages)})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            # Visualize original in 3D
            print("Creating 3D visualization of original image...")
            visualize_tiff_3d(pages, "Original 3D Visualization")

            # Progressively remove colors
            current_pages = [page.copy() for page in pages]

            # Keep track of the 2-color version pages
            two_color_pages = None

            for i in range(n_colors, 0, -1):
                if i == n_colors:
                    # Skip the first iteration (all colors)
                    continue

                print(f"Creating version with {i} colors (removing {n_colors - i} colors)...")
                start_time = time.time()

                # Get the next color to remove
                if len(colors_to_remove) > 0:
                    color_to_remove = colors_to_remove.pop(0)
                    print(f"Removing color with value: {color_to_remove}")

                    # Update all pages
                    for j in range(len(current_pages)):
                        # Create a mask for this color
                        mask = current_pages[j] == color_to_remove
                        # Set to zero (background)
                        current_pages[j] = current_pages[j].copy()  # Ensure copy for safety
                        current_pages[j][mask] = 0

                # Save as multi-page TIFF with .tiff extension using PIL to preserve P-mode
                output_path = os.path.join(output_folder, f"{file_name}_{i}_colors.tiff")
                print(f"Saving to: {output_path}")

                # First create a stack of PIL images
                pil_images = []
                for page in current_pages:
                    # Create a new PIL image in P mode with the original palette
                    img = Image.fromarray(page.astype(np.uint8), mode='P')
                    if palette is not None:
                        img.putpalette(palette)
                    pil_images.append(img)

                # Save the first image with append_images for the rest
                if pil_images:
                    # Add metadata to the image
                    if metadata_dict:
                        # Update the metadata to reflect the current state
                        metadata_dict["Number_of_Colors"] = i
                        metadata_json = json.dumps(metadata_dict)
                    else:
                        # Create basic metadata if none existed
                        metadata_json = json.dumps({"Number_of_Colors": i})

                    # Save the multi-page TIFF with metadata
                    pil_images[0].save(
                        output_path,
                        format='TIFF',
                        save_all=True,
                        append_images=pil_images[1:],
                        description=metadata_json
                    )

                # Display the middle slice for visualization
                plt.figure(figsize=(10, 8))
                plt.imshow(current_pages[middle_index])
                plt.title(f"Image with {i} colors (Page {middle_index + 1} of {len(pages)})")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                # Save the 2-color version for later display
                if i == 2:
                    two_color_pages = [page.copy() for page in current_pages]

                print(f"Processing completed in {time.time() - start_time:.2f} seconds")

            # Display the 2-color version at the end
            if two_color_pages is not None:
                print("\n\n--- Displaying the 2-color version ---")

                # Display a middle slice
                plt.figure(figsize=(12, 10))
                plt.imshow(two_color_pages[middle_index])
                plt.title("Final 2-Color Version (Middle Slice)")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                # Display the 3D visualization of the 2-color version
                print("Creating 3D visualization of 2-color version...")
                visualize_tiff_3d(two_color_pages, "2-Color Version 3D Visualization")

    except Exception as e:
        print(f"Error processing multi-page TIFF: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Set the input path directly as a string
    # input_path = "C:/Users/matic/Documents/Yale_BioInformatics_Lab/Neuron_Project_Tk/Tiffany_Module/Test_cases/cropped_label_Vermelho_2_output.tiff"  # Your actual file path
    input_path = "C:/Users/matic/Downloads/label (1)/label.tif"  # Your actual file path
    # Validate input file
    if not os.path.isfile(input_path):
        print(f"Error: File {input_path} does not exist")
        return

    # Create output folder
    output_folder = create_output_folder()
    print(f"Output will be saved to: {output_folder}")

    # Analyze file structure
    num_pages, page_shapes, page_dtypes = analyze_tiff_file(input_path)

    if num_pages > 0:
        # Process the multi-page TIFF
        process_multipage_tiff(input_path, output_folder)
    else:
        print("Could not properly analyze the TIFF file structure.")

    print("Processing complete!")


if __name__ == "__main__":
    main()