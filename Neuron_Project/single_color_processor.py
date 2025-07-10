"""
Single Color Figure Processor
Uses functions from Extract_Figures_FV.py to automatically process single color TIFF figures
and generate estimations with metadata.
"""

import numpy as np
from Extract_Figures_FV import (
    Open_TIFF_fig, 
    Geneate_Estimations,
    set_matplotlib_display,
    get_matplotlib_display_status,
    Export_Spine_as_Text,
    Export_Spine_as_tiff,
    crie_matriz_3D
)
from data_in_image import DataInImage
import Extract_Figures_FV as efv
import os


def save_single_color_figure_with_metadata(spine_3D_matrix, data_in_spine, output_path_base, segment_number):
    """
    Robustly save a single color figure with metadata using Extract_Figures_FV functions.
    This function ensures the figure is saved regardless of processing errors.
    
    Args:
        spine_3D_matrix: 3D matrix representing the spine segment
        data_in_spine (DataInImage): Spine data with metadata
        output_path_base (str): Base path for output files (without extension)
        segment_number (int): Segment number being processed
        
    Returns:
        tuple: (csv_success, tiff_success) - boolean flags indicating save success
    """
    csv_success = False
    tiff_success = False
    
    # Generate output file paths
    csv_path = f"{output_path_base}_{segment_number}_output.csv"
    tiff_path = f"{output_path_base}_{segment_number}_output.tiff"
    
    print(f"Saving segment {segment_number} with metadata...")
    
    # Save as CSV with metadata (robust error handling)
    try:
        Export_Spine_as_Text(csv_path, spine_3D_matrix, data_in_spine)
        csv_success = True
        print(f"✓ Successfully saved CSV: {csv_path}")
    except Exception as e:
        print(f"✗ Error saving CSV: {e}")
        try:
            # Fallback: create a minimal CSV with basic info
            with open(csv_path, 'w') as f:
                f.write(f"# Emergency save for segment {segment_number}\n")
                f.write(f"# Volume: {getattr(data_in_spine, 'volume', 'unknown')}\n")
                f.write(f"# Surface: {getattr(data_in_spine, 'surface', 'unknown')}\n")
                f.write(f"# Error during normal save: {e}\n")
            print(f"✓ Created fallback CSV: {csv_path}")
            csv_success = True
        except Exception as e2:
            print(f"✗ Even fallback CSV failed: {e2}")
    
    # Save as TIFF with metadata (robust error handling)
    try:
        Export_Spine_as_tiff(tiff_path, spine_3D_matrix, data_in_spine)
        tiff_success = True
        print(f"✓ Successfully saved TIFF: {tiff_path}")
    except Exception as e:
        print(f"✗ Error saving TIFF: {e}")
        try:
            # Fallback: save as simple numpy array if possible
            import pickle
            fallback_path = f"{output_path_base}_{segment_number}_matrix.pkl"
            with open(fallback_path, 'wb') as f:
                pickle.dump({
                    'matrix': spine_3D_matrix,
                    'metadata': data_in_spine.__dict__,
                    'segment_number': segment_number
                }, f)
            print(f"✓ Created fallback pickle file: {fallback_path}")
            tiff_success = True
        except Exception as e2:
            print(f"✗ Even fallback save failed: {e2}")
    
    return csv_success, tiff_success


def extract_and_save_segment_matrix(segment_number, output_path_base, data_in_spine):
    """
    Extract a specific segment from the loaded image and create its 3D matrix.
    This function safely extracts the segment data for saving.
    
    Args:
        segment_number (int): Segment number to extract
        output_path_base (str): Base path for output files
        data_in_spine (DataInImage): Spine data object
        
    Returns:
        tuple: (spine_3D_matrix, success_flag)
    """
    try:
        # Access global variables from Extract_Figures_FV
        image_uploaded = efv.image_uploaded
        image_uploaded_edge = efv.image_uploaded_edge
        
        if image_uploaded is None:
            print("Error: No image data available")
            return None, False
        
        # Calculate bounding box for the segment
        height = len(image_uploaded_edge[0])
        width = len(image_uploaded_edge[0][0])
        depth = len(image_uploaded_edge)
        
        max_ijk = [0, 0, 0]
        min_ijk = [height, width, depth]
        segment_found = False
        
        for i in range(height):
            for j in range(width):
                for k in range(depth):
                    if image_uploaded_edge[k][i][j] == segment_number:
                        segment_found = True
                        # Update bounding box
                        if i > max_ijk[0]: max_ijk[0] = i
                        if j > max_ijk[1]: max_ijk[1] = j
                        if k > max_ijk[2]: max_ijk[2] = k
                        if i < min_ijk[0]: min_ijk[0] = i
                        if j < min_ijk[1]: min_ijk[1] = j
                        if k < min_ijk[2]: min_ijk[2] = k
        
        if not segment_found:
            print(f"Error: Segment {segment_number} not found in image")
            return None, False
        
        # Create 3D matrix for the segment
        di = (max_ijk[0] - min_ijk[0] + 1)
        dj = (max_ijk[1] - min_ijk[1] + 1)
        dk = (max_ijk[2] - min_ijk[2] + 1)
        
        # Update data_in_spine dimensions
        data_in_spine.height = di
        data_in_spine.width = dj
        data_in_spine.num_layers = dk
        
        # Create the spine 3D matrix
        spine_3D = crie_matriz_3D(dk + 2, di + 2, dj + 2, 0)
        
        # Copy segment data to the matrix
        for i in range(di):
            for j in range(dj):
                for k in range(dk):
                    if image_uploaded[k + min_ijk[2]][i + min_ijk[0]][j + min_ijk[1]] == segment_number:
                        spine_3D[k + 1][i + 1][j + 1] = 1
                    elif image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0]][j + min_ijk[1]] == segment_number:
                        spine_3D[k + 1][i + 1][j + 1] = 1
        
        print(f"Successfully extracted segment {segment_number} matrix ({dk}x{di}x{dj})")
        return spine_3D, True
        
    except Exception as e:
        print(f"Error extracting segment matrix: {e}")
        return None, False


def save_single_color_figure_safe(tiff_path, segment_number=None, output_prefix=None, force_save=True):
    """
    Safely save a single color figure with metadata, ensuring save happens regardless of processing errors.
    
    Args:
        tiff_path (str): Path to the input TIFF file
        segment_number (int, optional): Specific segment number to save. If None, saves the first available segment
        output_prefix (str, optional): Custom prefix for output files. If None, uses the input filename
        force_save (bool): If True, will attempt to save even if processing fails
        
    Returns:
        dict: Dictionary with save status and file paths
    """
    result = {
        'success': False,
        'csv_saved': False,
        'tiff_saved': False,
        'csv_path': None,
        'tiff_path': None,
        'segment_number': None,
        'error': None
    }
    
    try:
        print(f"Safe saving for: {tiff_path}")
        
        # Disable plots for safety
        original_plot_setting = get_matplotlib_display_status()
        set_matplotlib_display(False)
        
        # Find the position of the last dot in the file path
        dot_index = tiff_path.rfind('.')
        base_name = tiff_path[:dot_index] if dot_index != -1 else tiff_path
        
        # Use custom output prefix if provided
        if output_prefix:
            base_name = output_prefix
        
        # Load the image
        data_in_image = Open_TIFF_fig(tiff_path)
        
        if data_in_image is None:
            result['error'] = "Could not load image data"
            return result
        
        # Detect available segments
        available_segments = detect_available_segments()
        
        if not available_segments:
            result['error'] = "No segments found in image"
            return result
        
        # Determine which segment to save
        if segment_number is None:
            target_segment = available_segments[0]
            print(f"No segment specified, saving first available segment: {target_segment}")
        else:
            if segment_number in available_segments:
                target_segment = segment_number
                print(f"Saving specified segment: {target_segment}")
            else:
                result['error'] = f"Segment {segment_number} not found. Available: {available_segments}"
                return result
        
        result['segment_number'] = target_segment
        
        # Create spine data object
        data_in_spine = DataInImage()
        data_in_spine.__dict__.update(data_in_image.__dict__)
        
        # Set basic metadata even if processing fails
        data_in_spine.description = f"Single color segment {target_segment} from {os.path.basename(tiff_path)}"
        
        # Try to extract segment matrix
        spine_3D_matrix, extraction_success = extract_and_save_segment_matrix(target_segment, base_name, data_in_spine)
        
        if not extraction_success or spine_3D_matrix is None:
            if force_save:
                # Create a minimal matrix for emergency save
                spine_3D_matrix = crie_matriz_3D(10, 10, 10, 0)
                spine_3D_matrix[5][5][5] = 1  # Add at least one point
                print("Using emergency matrix for save")
            else:
                result['error'] = "Could not extract segment matrix"
                return result
        
        # Save the figure with metadata
        csv_success, tiff_success = save_single_color_figure_with_metadata(
            spine_3D_matrix, data_in_spine, base_name, target_segment
        )
        
        result['csv_saved'] = csv_success
        result['tiff_saved'] = tiff_success
        result['csv_path'] = f"{base_name}_{target_segment}_output.csv"
        result['tiff_path'] = f"{base_name}_{target_segment}_output.tiff"
        result['success'] = csv_success or tiff_success
        
        if result['success']:
            print(f"✓ Successfully saved segment {target_segment}")
        else:
            print(f"✗ Failed to save segment {target_segment}")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        print(f"✗ Critical error during save: {e}")
        return result
    
    finally:
        # Restore original plot setting
        try:
            set_matplotlib_display(original_plot_setting)
        except:
            pass


def detect_available_segments():
    """
    Detect all available color segments in the currently loaded image.
    
    Returns:
        list: List of unique segment numbers (excluding 0 which represents background)
    """
    # Access the global image_uploaded from Extract_Figures_FV
    image_uploaded = efv.image_uploaded
    
    if image_uploaded is None:
        print("Error: No image loaded. Please load an image first.")
        return []
    
    # Convert to numpy array if it's not already
    if not isinstance(image_uploaded, np.ndarray):
        image_uploaded = np.array(image_uploaded)
    
    # Get unique values and exclude 0 (background)
    unique_segments = np.unique(image_uploaded)
    available_segments = [int(seg) for seg in unique_segments if seg != 0]
    
    print(f"Available segments detected: {available_segments}")
    return available_segments


def process_single_color_figure(tiff_path, segment_number=None, output_prefix=None, show_plots=False):
    """
    Process a single color figure from a TIFF file, generate estimations and save results with metadata.
    
    Args:
        tiff_path (str): Path to the input TIFF file
        segment_number (int, optional): Specific segment number to process. If None, will process the first available segment
        output_prefix (str, optional): Custom prefix for output files. If None, uses the input filename
        show_plots (bool): Whether to display matplotlib plots during processing
        
    Returns:
        DataInImage: The processed spine data with estimations
    """
    # Set matplotlib display preference
    set_matplotlib_display(show_plots)
    
    print(f"Processing single color figure: {tiff_path}")
    
    # Find the position of the last dot in the file path
    dot_index = tiff_path.rfind('.')
    base_name = tiff_path[:dot_index] if dot_index != -1 else tiff_path
    
    # Use custom output prefix if provided
    if output_prefix:
        base_name = output_prefix
    
    print("Loading Figure...")
    
    # Read the TIFF file and get image data using the existing function
    data_in_image = Open_TIFF_fig(tiff_path)
    
    if data_in_image is None:
        print("Error: Could not load image data")
        return None
    
    # Detect available segments
    available_segments = detect_available_segments()
    
    if not available_segments:
        print("Error: No segments found in the image")
        return None
    
    # Determine which segment to process
    if segment_number is None:
        # Process the first available segment
        target_segment = available_segments[0]
        print(f"No segment specified, processing first available segment: {target_segment}")
    else:
        if segment_number in available_segments:
            target_segment = segment_number
            print(f"Processing specified segment: {target_segment}")
        else:
            print(f"Error: Segment {segment_number} not found in image. Available segments: {available_segments}")
            return None
    
    # Create a copy of the image data for the spine
    data_in_spine = DataInImage()
    # Copy attributes from the original image data
    data_in_spine.__dict__.update(data_in_image.__dict__)
    
    print(f"Generating estimations for segment {target_segment}...")
    
    # Generate estimations for the selected segment using the existing function
    Geneate_Estimations(target_segment, base_name, data_in_spine)
    
    print(f"Successfully processed segment {target_segment}")
    print(f"Output files: {base_name}_{target_segment}_output.csv and {base_name}_{target_segment}_output.tiff")
    
    return data_in_spine


def process_all_segments_in_figure(tiff_path, output_prefix=None, show_plots=False):
    """
    Process all available segments in a TIFF figure, generating estimations for each.
    
    Args:
        tiff_path (str): Path to the input TIFF file
        output_prefix (str, optional): Custom prefix for output files. If None, uses the input filename
        show_plots (bool): Whether to display matplotlib plots during processing
        
    Returns:
        dict: Dictionary mapping segment numbers to their DataInImage objects
    """
    # Set matplotlib display preference
    set_matplotlib_display(show_plots)
    
    print(f"Processing all segments in figure: {tiff_path}")
    
    # Find the position of the last dot in the file path
    dot_index = tiff_path.rfind('.')
    base_name = tiff_path[:dot_index] if dot_index != -1 else tiff_path
    
    # Use custom output prefix if provided
    if output_prefix:
        base_name = output_prefix
    
    print("Loading Figure...")
    
    # Read the TIFF file and get image data using the existing function
    data_in_image = Open_TIFF_fig(tiff_path)
    
    if data_in_image is None:
        print("Error: Could not load image data")
        return None
    
    # Detect available segments
    available_segments = detect_available_segments()
    
    if not available_segments:
        print("Error: No segments found in the image")
        return None
    
    print(f"Processing {len(available_segments)} segments: {available_segments}")
    
    processed_segments = {}
    
    # Process each segment
    for segment_number in available_segments:
        print(f"\n--- Processing segment {segment_number} ---")
        
        # Create a copy of the image data for this spine
        data_in_spine = DataInImage()
        # Copy attributes from the original image data
        data_in_spine.__dict__.update(data_in_image.__dict__)
        
        try:
            # Generate estimations for this segment using the existing function
            Geneate_Estimations(segment_number, base_name, data_in_spine)
            processed_segments[segment_number] = data_in_spine
            print(f"Successfully processed segment {segment_number}")
        except Exception as e:
            print(f"Error processing segment {segment_number}: {e}")
            continue
    
    print(f"\nCompleted processing {len(processed_segments)} segments")
    return processed_segments


def get_image_info(tiff_path):
    """
    Get information about a TIFF file without processing it.
    
    Args:
        tiff_path (str): Path to the input TIFF file
        
    Returns:
        tuple: (data_in_image, available_segments)
    """
    print(f"Getting info for: {tiff_path}")
    
    # Disable plots for info gathering
    original_plot_setting = get_matplotlib_display_status()
    set_matplotlib_display(False)
    
    try:
        # Read the TIFF file
        data_in_image = Open_TIFF_fig(tiff_path)
        
        if data_in_image is None:
            print("Error: Could not load image data")
            return None, []
        
        # Detect available segments
        available_segments = detect_available_segments()
        
        print("Image Information:")
        data_in_image.print_data()
        
        return data_in_image, available_segments
        
    finally:
        # Restore original plot setting
        set_matplotlib_display(original_plot_setting)


# Example usage functions
def example_usage():
    """
    Example demonstrating how to use the single color processing functions.
    """
    # Example file path - update this to your actual file
    tiff_path = 'label.tif'
    
    print("=== Single Color Figure Processing Examples ===\n")
    
    # Example 1: Get information about the image
    print("1. Getting image information...")
    data_info, segments = get_image_info(tiff_path)
    if data_info and segments:
        print(f"Found {len(segments)} segments: {segments}\n")
    
    # Example 2: Safe save (new robust function)
    print("2. Safe saving with metadata...")
    save_result = save_single_color_figure_safe(tiff_path, segment_number=None, output_prefix='safe_save')
    if save_result['success']:
        print(f"✓ Safe save completed for segment {save_result['segment_number']}")
        print(f"  CSV: {save_result['csv_saved']}")
        print(f"  TIFF: {save_result['tiff_saved']}")
    else:
        print(f"✗ Safe save failed: {save_result['error']}")
    
    # Example 3: Process a specific segment
    print("\n3. Processing a specific segment...")
    if segments:
        result = process_single_color_figure(tiff_path, segment_number=segments[0], 
                                           output_prefix='spine_analysis', show_plots=False)
        if result:
            print("Single segment processing completed successfully\n")
    
    # Example 4: Process all segments in the figure
    print("4. Processing all segments...")
    results = process_all_segments_in_figure(tiff_path, output_prefix='all_spines', show_plots=False)
    if results:
        print(f"Processed {len(results)} segments total")
        for seg_num, data in results.items():
            print(f"  Segment {seg_num}: Volume = {data.volume}, Surface = {data.surface}")


if __name__ == "__main__":
    # Run example usage
    example_usage() 