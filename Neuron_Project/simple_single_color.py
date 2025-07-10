"""
Simple Single Color Processor
Directly uses the existing functions from Extract_Figures_FV.py to automatically
process and save single color figures without user input.
"""

import numpy as np
from Extract_Figures_FV import Open_TIFF_fig, Geneate_Estimations, set_matplotlib_display
from data_in_image import DataInImage
import Extract_Figures_FV as efv


def get_available_segments():
    """Get all available segments from the currently loaded image."""
    if efv.image_uploaded is None:
        return []
    
    image_array = np.array(efv.image_uploaded)
    unique_segments = np.unique(image_array)
    segments = [int(seg) for seg in unique_segments if seg != 0]
    return segments


def process_single_color_auto(tiff_path, segment_number=None, output_prefix=None, show_plots=False):
    """
    Automatically process and save a single color figure using Extract_Figures_FV functions.
    
    Args:
        tiff_path (str): Path to TIFF file
        segment_number (int, optional): Specific segment to process. If None, uses first available
        output_prefix (str, optional): Custom output prefix
        show_plots (bool): Whether to show matplotlib plots
    
    Returns:
        DataInImage: Processed spine data, or None if failed
    """
    # Set plot display
    set_matplotlib_display(show_plots)
    
    print(f"Processing: {tiff_path}")
    
    # Determine output base name
    dot_index = tiff_path.rfind('.')
    base_name = tiff_path[:dot_index] if dot_index != -1 else tiff_path
    if output_prefix:
        base_name = output_prefix
    
    try:
        # Load the image using the original function
        data_in_image = Open_TIFF_fig(tiff_path)
        if data_in_image is None:
            print("Failed to load image")
            return None
        
        # Get available segments
        segments = get_available_segments()
        if not segments:
            print("No segments found")
            return None
        
        # Choose segment
        if segment_number is None:
            target_segment = segments[0]
            print(f"Auto-selected segment: {target_segment}")
        elif segment_number in segments:
            target_segment = segment_number
            print(f"Processing segment: {target_segment}")
        else:
            print(f"Segment {segment_number} not found. Available: {segments}")
            return None
        
        # Create spine data object
        data_in_spine = DataInImage()
        data_in_spine.__dict__.update(data_in_image.__dict__)
        
        # Use the original function to process and save everything
        print("Generating estimations and saving...")
        Geneate_Estimations(target_segment, base_name, data_in_spine)
        
        print(f"✓ Complete! Files saved:")
        print(f"  - {base_name}_{target_segment}_output.csv")
        print(f"  - {base_name}_{target_segment}_output.tiff")
        
        return data_in_spine
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_all_segments_auto(tiff_path, output_prefix=None, show_plots=False):
    """
    Process all segments in a figure automatically.
    
    Args:
        tiff_path (str): Path to TIFF file
        output_prefix (str, optional): Custom output prefix
        show_plots (bool): Whether to show matplotlib plots
    
    Returns:
        dict: Mapping of segment numbers to DataInImage objects
    """
    # Set plot display
    set_matplotlib_display(show_plots)
    
    print(f"Processing all segments in: {tiff_path}")
    
    # Determine output base name
    dot_index = tiff_path.rfind('.')
    base_name = tiff_path[:dot_index] if dot_index != -1 else tiff_path
    if output_prefix:
        base_name = output_prefix
    
    try:
        # Load the image
        data_in_image = Open_TIFF_fig(tiff_path)
        if data_in_image is None:
            print("Failed to load image")
            return {}
        
        # Get all segments
        segments = get_available_segments()
        if not segments:
            print("No segments found")
            return {}
        
        print(f"Found {len(segments)} segments: {segments}")
        
        results = {}
        
        # Process each segment
        for segment_num in segments:
            print(f"\n--- Processing segment {segment_num} ---")
            
            try:
                # Create fresh spine data for this segment
                data_in_spine = DataInImage()
                data_in_spine.__dict__.update(data_in_image.__dict__)
                
                # Process and save using original function
                Geneate_Estimations(segment_num, base_name, data_in_spine)
                results[segment_num] = data_in_spine
                
                print(f"✓ Segment {segment_num} complete")
                
            except Exception as e:
                print(f"✗ Error with segment {segment_num}: {e}")
                continue
        
        print(f"\nProcessed {len(results)}/{len(segments)} segments successfully")
        return results
        
    except Exception as e:
        print(f"Critical error: {e}")
        return {}


def quick_save(tiff_path, segment_number=None):
    """
    Quick function to save a single color figure with minimal setup.
    
    Args:
        tiff_path (str): Path to TIFF file
        segment_number (int, optional): Segment to save. If None, saves first available
    
    Returns:
        bool: True if successful, False otherwise
    """
    result = process_single_color_auto(tiff_path, segment_number, show_plots=False)
    return result is not None


def save_with_original_workflow(tiff_path, segment_number=None, output_prefix=None, show_plots=False):
    """
    Save a single color figure using the *original* workflow inside Extract_Figures_FV.Generate_3D_Segment_Library
    but without any interactive input. We monkey-patch efv.get_integer_list so the original
    code thinks the user entered the desired segment list.
    
    This guarantees the CSV and TIFF are created the same way the original interactive script does,
    but programmatically.
    
    Args:
        tiff_path (str): Path to the TIFF image.
        segment_number (int|None): If None, automatically picks the first non-zero palette index.
        output_prefix (str|None): Optional override for base output name.
        show_plots (bool): Whether to display matplotlib GUI windows.
    Returns:
        list[int]: List of segments that were actually processed and (attempted to) save.
    """
    # Control plot display
    set_matplotlib_display(show_plots)

    # ------------------------------------------------------------
    # 1.  Inspect the image to determine available segments
    # ------------------------------------------------------------
    data_in_image = efv.Open_TIFF_fig(tiff_path)
    if data_in_image is None:
        raise RuntimeError("Could not open TIFF or unsupported color mode – only indexed (P-mode) is supported.")

    segments = get_available_segments()
    if not segments:
        raise RuntimeError("No non-zero palette indices (segments) detected in the image – nothing to save.")

    # Decide which segments to save
    if segment_number is not None:
        if segment_number not in segments:
            raise ValueError(f"Requested segment {segment_number} not found. Available segments: {segments}")
        chosen_segments = [segment_number]
    else:
        chosen_segments = segments  # save them all by default

    print(f"Chosen segments to save: {chosen_segments}")

    # ------------------------------------------------------------
    # 2.  Monkey-patch efv.get_integer_list so the original code
    #     thinks the user typed our chosen list at the prompt.
    # ------------------------------------------------------------
    def _fake_input():
        return chosen_segments

    original_get_integer_list = efv.get_integer_list  # keep reference
    efv.get_integer_list = _fake_input

    # ------------------------------------------------------------
    # 3.  Temporarily adjust the base name if the caller supplied one
    #     (The Generate_3D_Segment_Library builds base_name from the path.)
    # ------------------------------------------------------------
    if output_prefix is not None:
        # quick hack: copy the file temporarily with a new name that has the desired prefix
        import shutil, os, tempfile
        tmp_dir = tempfile.mkdtemp(prefix="single_color_")
        ext = os.path.splitext(tiff_path)[1]
        tmp_path = os.path.join(tmp_dir, output_prefix + ext)
        shutil.copy2(tiff_path, tmp_path)
        path_for_processing = tmp_path
    else:
        path_for_processing = tiff_path

    try:
        # --------------------------------------------------------
        # 4.  Call the original high-level function – this will:
        #         - call Open_TIFF_fig again (fast, already cached)
        #         - build boundary matrices
        #         - call Geneate_Estimations which finally writes the CSV & TIFF
        # --------------------------------------------------------
        efv.Generate_3D_Segment_Library(path_for_processing)
    finally:
        # Always restore the original function so we don't surprise other code
        efv.get_integer_list = original_get_integer_list

    print("Saving workflow complete (check files with *_output.csv / *_output.tiff suffixes).")
    return chosen_segments


# ====================================================================
# If run as a script – demo the new guaranteed-save workflow
# ====================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python simple_single_color.py <tiff_path> [segment_number]")
        sys.exit(1)

    path = sys.argv[1]
    seg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        save_with_original_workflow(path, segment_number=seg)
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(2)
    print("✓ Done") 