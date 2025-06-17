import tkinter as tk
import numpy as np
import tifffile as tiff
from PIL import Image
import json
import os
import io
import gc
from threading import Thread
from queue import Queue
import mmap
from concurrent.futures import ThreadPoolExecutor
from data_in_image import DataInImage
from tiff_viewer_ui import TIFFViewerUI
# Legacy image helpers
from Extract_Figures_FV_Classes import ImageProcessing


class TIFFViewer3D:
    """
    Main class that handles the loading and processing of the TIFF images.
    """

    def __init__(self):
        # Initialize variables
        self.rgb_colors = None
        self.image_uploaded = None
        self.selected_p_values = []
        self.current_color_index = 0
        self.current_spine_number = None
        self._processing_queue = Queue()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize metadata handling
        self.data_in_image = DataInImage()
        self.single_color_data = None
        self.current_mask = None

        # Create the Tkinter window
        self.root = tk.Tk()
        self.root.title("3D TIFF Viewer with Color Selection")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Initialize UI
        self.ui = TIFFViewerUI(self.root, self)

        # Start background processing thread
        self._start_background_processing()

        # Start the Tkinter event loop
        self.root.mainloop()

    def _start_background_processing(self):
        """Start background processing thread"""
        def process_queue():
            while True:
                try:
                    func, args, kwargs = self._processing_queue.get()
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Background processing error: {e}")
                finally:
                    self._processing_queue.task_done()
                    gc.collect()

        thread = Thread(target=process_queue, daemon=True)
        thread.start()

    def initialize_viewer(self, image_path):
        """Initialize the viewer with optimized TIFF loading"""
        self.image_path = image_path
        self.ui.update_status("Loading TIFF file...")
        
        # Reset metadata for new file
        self.data_in_image = DataInImage()
        self.single_color_data = None
        self.current_mask = None
        
        # Use legacy checker to validate mode early
        try:
            if ImageProcessing().check_image(image_path) != 'P':
                raise ValueError('The image is not in indexed color mode (P mode).')
        except Exception as _e:
            # Fall back to original check later
            pass
        
        try:
            # Memory-mapped file reading
            with open(image_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                img = Image.open(io.BytesIO(mm.read()))

                if img.mode != 'P':
                    raise ValueError('The image is not in indexed color mode (P mode).')

                if not hasattr(img, "n_frames") or img.n_frames < 1:
                    raise ValueError('The TIFF file does not contain multiple frames.')

                # Initialize basic metadata
                self.data_in_image.num_layers = img.n_frames
                self.data_in_image.height = img.size[1]
                self.data_in_image.width = img.size[0]
                
                # Get color palette and ensure it exists
                palette = img.getpalette()
                if not palette:
                    raise ValueError('No color palette found in the image.')
                
                # Process color palette
                num_colors = len(palette) // 3
                self.rgb_colors = []
                for i in range(num_colors):
                    r = palette[i * 3]
                    g = palette[i * 3 + 1]
                    b = palette[i * 3 + 2]
                    if r is not None and g is not None and b is not None:
                        self.rgb_colors.append((r, g, b))
                    else:
                        self.rgb_colors.append((0, 0, 0))  # Default to black for invalid colors
                
                # Ensure we have at least one color
                if not self.rgb_colors:
                    raise ValueError('No valid colors found in the palette.')

                # Convert to numpy array for faster processing
                self.rgb_colors_normalized = np.array(self.rgb_colors, dtype=np.float32) / 255.0

                # Process frames in chunks
                chunk_size = self.ui.chunk_size.get()
                total_frames = img.n_frames
                layers = []
                
                for start_frame in range(0, total_frames, chunk_size):
                    end_frame = min(start_frame + chunk_size, total_frames)
                    chunk_layers = []
                    
                    for frame in range(start_frame, end_frame):
                        img.seek(frame)
                        frame_data = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
                        chunk_layers.append(frame_data)
                        
                        # Update progress
                        progress = (frame + 1) / total_frames * 100
                        self.ui.update_status(f"Loading frames... {progress:.1f}%")
                    
                    # Stack chunk and append to layers
                    chunk_array = np.stack(chunk_layers, axis=0)
                    layers.append(chunk_array)
                    
                    # Force garbage collection after each chunk
                    gc.collect()

                # Stack all chunks
                self.image_uploaded = np.concatenate(layers, axis=0)
                
                # Clean up
                mm.close()
                del layers
                gc.collect()

                # Get unique indices and create color index list
                unique_indices = np.unique(self.image_uploaded)
                self.color_index_list = []
                for idx in unique_indices:
                    if idx == 0:  # Skip background
                        continue
                    if idx < len(self.rgb_colors):
                        self.color_index_list.append((idx, self.rgb_colors[idx]))
                    else:
                        print(f"Warning: Color index {idx} out of range")

                if not self.color_index_list:
                    raise ValueError('No valid color indices found in the image.')

                self.ui.update_status("TIFF file loaded successfully")

        except Exception as e:
            self.ui.update_status("Error loading TIFF file")
            tk.messagebox.showerror(
                "File Error",
                f"The selected file could not be read as a TIFF image.\nError: {str(e)}"
            )
            return

        self.selected_p_values = []

    def _create_colors_mapped(self):
        """Create color mapping with memory optimization"""
        try:
            # Calculate shape and preallocate memory
            shape = self.image_uploaded.shape + (4,)
            self.colors_mapped = np.zeros(shape, dtype=np.float32)
            
            # Process in chunks
            chunk_size = self.ui.chunk_size.get()
            for i in range(0, len(self.rgb_colors), chunk_size):
                chunk = slice(i, min(i + chunk_size, len(self.rgb_colors)))
                
                for idx in range(chunk.start, chunk.stop):
                    if idx == 0:
                        continue
                    
                    # Process each color index efficiently
                    mask = (self.image_uploaded == idx)
                    if np.any(mask):
                        rgb_normalized = self.rgb_colors_normalized[idx]
                        self.colors_mapped[mask] = np.append(rgb_normalized, 1.0)
                
                # Update progress
                progress = (i + chunk_size) / len(self.rgb_colors) * 100
                self.ui.update_status(f"Creating color mapping... {progress:.1f}%")
                
                # Clean up after each chunk
                gc.collect()
            
            self.ui.update_status("Color mapping complete")
            
        except Exception as e:
            self.ui.update_status(f"Error creating color mapping: {str(e)}")

    def save_data_as_tiff(self, save_path, data_in_spine):
        """Save TIFF with optimized memory usage"""
        try:
            # Prepare the image data efficiently
            self.ui.update_status("Preparing data for saving...")
            mask = np.zeros_like(self.current_mask, dtype=np.uint8)
            mask[self.current_mask] = 255

            # Create metadata
            metadata_dict = {}
            for tag, attr in data_in_spine.tag_dict.items():
                value = getattr(data_in_spine, attr)
                if value is not None:
                    metadata_dict[tag] = self._convert_numpy_types(value)

            # Save in chunks
            self.ui.update_status("Saving TIFF file...")
            chunk_size = min(100, mask.shape[0])  # Adjust chunk size based on available memory
            with tiff.TiffWriter(save_path) as tif:
                for i in range(0, mask.shape[0], chunk_size):
                    end = min(i + chunk_size, mask.shape[0])
                    chunk = mask[i:end]
                    if i == 0:  # First chunk includes metadata
                        tif.write(chunk, photometric='minisblack',
                                description=json.dumps(metadata_dict))
                    else:
                        tif.write(chunk, photometric='minisblack')
                    
                    # Update progress
                    progress = (end / mask.shape[0]) * 100
                    self.ui.update_status(f"Saving... {progress:.1f}%")

            self.ui.update_status(f"Saved TIFF file to {save_path}")
            
        except Exception as e:
            self.ui.update_status("Error saving TIFF file")
            print(f"Failed to save TIFF with metadata: {e}")

    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types"""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(o) for o in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        else:
            return obj 