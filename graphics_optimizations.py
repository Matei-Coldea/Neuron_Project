import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Thread
import queue
import time
import gc
import tempfile
import os
import tkinter as tk
from tkinter import ttk
import webbrowser
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# Set Plotly to not auto-open browser
pio.renderers.default = "browser"

class SmartDataReducer:
    """Intelligent data reduction system that preserves structure while reducing points"""
    
    def __init__(self):
        self.reduction_strategies = {
            'spatial_sampling': self._spatial_sampling,
            'random_sampling': self._random_sampling,
            'grid_sampling': self._grid_sampling
        }
        self.cache = {}
    
    def reduce_for_visualization(self, coords, colors=None, target_points=25000, strategy='spatial_sampling'):
        """Intelligently reduce voxel data while preserving structure"""
        if len(coords) <= target_points:
            if colors is not None:
                return coords, colors
            return coords
        
        # Create cache key
        cache_key = (len(coords), target_points, strategy)
        if cache_key in self.cache:
            indices = self.cache[cache_key]
        else:
            # Generate reduction indices
            if strategy in self.reduction_strategies:
                indices = self.reduction_strategies[strategy](coords, target_points)
            else:
                indices = self._random_sampling(coords, target_points)
            
            # Cache the indices
            self.cache[cache_key] = indices
        
        reduced_coords = coords[indices]
        if colors is not None:
            reduced_colors = colors[indices] if len(colors) == len(coords) else colors
            return reduced_coords, reduced_colors
        
        return reduced_coords
    
    def _spatial_sampling(self, coords, target_points):
        """Sample points while preserving spatial distribution using clustering"""
        try:
            # Use mini-batch k-means for large datasets
            n_clusters = min(target_points, len(coords))
            
            if len(coords) > 100000:
                # Use mini-batch for very large datasets
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            clusters = kmeans.fit_predict(coords)
            
            # Take representative points from each cluster
            sampled_indices = []
            for i in range(n_clusters):
                cluster_mask = clusters == i
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) > 0:
                    # Take point closest to cluster center
                    cluster_coords = coords[cluster_indices]
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_coords - center, axis=1)
                    best_local_idx = np.argmin(distances)
                    best_global_idx = cluster_indices[best_local_idx]
                    sampled_indices.append(best_global_idx)
            
            return np.array(sampled_indices)
            
        except Exception as e:
            print(f"Spatial sampling failed: {e}, falling back to random sampling")
            return self._random_sampling(coords, target_points)
    
    def _random_sampling(self, coords, target_points):
        """Fallback random sampling"""
        indices = np.random.choice(len(coords), target_points, replace=False)
        return indices
    
    def _grid_sampling(self, coords, target_points):
        """Grid-based sampling for uniform distribution"""
        # Create 3D grid and sample from each occupied cell
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        
        # Estimate grid size
        total_volume = np.prod(max_coords - min_coords + 1)
        grid_size = int(np.cbrt(total_volume / target_points))
        grid_size = max(1, grid_size)
        
        # Assign points to grid cells
        grid_coords = ((coords - min_coords) // grid_size).astype(int)
        
        # Sample one point from each occupied cell
        unique_cells, cell_indices = np.unique(grid_coords, axis=0, return_inverse=True)
        sampled_indices = []
        
        for i in range(len(unique_cells)):
            cell_points = np.where(cell_indices == i)[0]
            if len(cell_points) > 0:
                # Take random point from cell or center-most point
                sampled_indices.append(np.random.choice(cell_points))
        
        return np.array(sampled_indices[:target_points])


class BackgroundProcessor:
    """Background processing system to prevent UI freezing"""
    
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_futures = {}
        self.result_queue = queue.Queue()
    
    def process_spine_data_async(self, image_data, spine_number, rgb_colors, callback=None):
        """Process spine data in background thread"""
        future = self.executor.submit(self._process_spine_data, image_data, spine_number, rgb_colors)
        
        if callback:
            def on_complete(fut):
                try:
                    result = fut.result()
                    callback(result)
                except Exception as e:
                    print(f"Background processing error: {e}")
                    if callback:
                        callback(None)
            
            future.add_done_callback(on_complete)
        
        return future
    
    def _process_spine_data(self, image_data, spine_number, rgb_colors):
        """Process spine data without blocking UI"""
        try:
            # Extract mask efficiently using boolean indexing
            mask = image_data == spine_number
            coords = np.argwhere(mask)
            
            if len(coords) == 0:
                return None
            
            # Get colors for the coordinates
            colors = []
            rgb_color = rgb_colors.get(spine_number, (255, 0, 0))  # Default to red
            normalized_color = np.array(rgb_color, dtype=np.float32) / 255.0
            colors = np.tile(normalized_color, (len(coords), 1))
            
            # Apply data reduction if needed
            reducer = SmartDataReducer()
            if len(coords) > 25000:
                coords, colors = reducer.reduce_for_visualization(coords, colors, 25000)
            
            return {
                'coords': coords,
                'colors': colors,
                'original_mask': mask,
                'spine_number': spine_number,
                'total_points': len(np.argwhere(mask)),
                'displayed_points': len(coords)
            }
            
        except Exception as e:
            print(f"Error in background processing: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the thread pool"""
        self.executor.shutdown(wait=True)


class OptimizedPlotlyRenderer:
    """Drop-in replacement for matplotlib with Plotly WebGL rendering"""
    
    def __init__(self):
        self.temp_files = []
        self.current_html_file = None
    
    def create_figure_replacement(self, figsize=(8, 6), dpi=100):
        """Create a figure object that mimics matplotlib behavior"""
        return OptimizedFigure(self, figsize, dpi)
    
    def create_optimized_plot(self, coords, colors, title="3D Visualization", point_size=3):
        """Create optimized Plotly 3D scatter plot"""
        
        # Ensure coords and colors are numpy arrays
        coords = np.asarray(coords)
        colors = np.asarray(colors)
        
        # Handle different color formats
        if colors.ndim == 2 and colors.shape[1] >= 3:
            # RGB format - convert to hex strings
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(int)
            color_strings = [f'rgb({r},{g},{b})' for r, g, b in colors[:, :3]]
        else:
            # Single color or other format
            color_strings = 'red'
        
        # Create the 3D scatter plot
        trace = go.Scatter3d(
            x=coords[:, 2],  # Swap to match matplotlib convention
            y=coords[:, 1],
            z=coords[:, 0],
            mode='markers',
            marker=dict(
                color=color_strings,
                size=point_size,
                opacity=0.8,
                line=dict(width=0)  # Remove outlines for performance
            ),
            hoverinfo='skip',  # Disable hover for performance
            showlegend=False
        )
        
        layout = go.Layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis=dict(title='X axis', showgrid=True, gridcolor='lightgray'),
                yaxis=dict(title='Y axis', showgrid=True, gridcolor='lightgray'),
                zaxis=dict(title='Z axis', showgrid=True, gridcolor='lightgray'),
                bgcolor='white',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig = go.Figure(data=[trace], layout=layout)
        return fig
    
    def save_html_temp(self, fig):
        """Save Plotly figure as temporary HTML file"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w')
        html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
        
        # Create a complete HTML page
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>3D Visualization</title>
            <style>
                body {{ margin: 0; padding: 0; }}
                .plotly-graph-div {{ height: 100vh; width: 100vw; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        temp_file.write(full_html)
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        self.current_html_file = temp_file.name
        return temp_file.name
    
    def cleanup_temp_files(self):
        """Clean up temporary HTML files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self.temp_files = []
    
    def __del__(self):
        self.cleanup_temp_files()


class OptimizedFigure:
    """Figure replacement that uses Plotly backend"""
    
    def __init__(self, renderer, figsize, dpi):
        self.renderer = renderer
        self.figsize = figsize
        self.dpi = dpi
        self.axes = []
        self.plotly_fig = None
        
    def add_subplot(self, *args, **kwargs):
        """Add subplot - return optimized axis"""
        ax = OptimizedAxis(self)
        self.axes.append(ax)
        return ax
    
    def clear(self):
        """Clear the figure"""
        for ax in self.axes:
            ax.clear()
        self.plotly_fig = None


class OptimizedAxis:
    """Axis replacement that builds Plotly plots"""
    
    def __init__(self, figure):
        self.figure = figure
        self.plots_data = []
        self.title = ""
        self.xlabel = "X axis"
        self.ylabel = "Y axis" 
        self.zlabel = "Z axis"
        
    def scatter(self, x, y, z, c=None, s=20, alpha=1.0, **kwargs):
        """Scatter plot replacement"""
        coords = np.column_stack([z, y, x])  # Note: z,y,x order for Plotly
        
        # Handle colors
        if c is not None:
            if hasattr(c, '__len__') and len(c) > 1:
                colors = np.asarray(c)
            else:
                colors = np.tile([1.0, 0.0, 0.0], (len(coords), 1))  # Default red
        else:
            colors = np.tile([1.0, 0.0, 0.0], (len(coords), 1))
        
        # Apply data reduction
        reducer = SmartDataReducer()
        if len(coords) > 25000:
            coords, colors = reducer.reduce_for_visualization(coords, colors, 25000)
        
        # Store plot data
        self.plots_data.append({
            'type': 'scatter',
            'coords': coords,
            'colors': colors,
            'size': s if hasattr(s, '__len__') else 3,
            'alpha': alpha
        })
    
    def voxels(self, filled, facecolors=None, **kwargs):
        """Voxels plot replacement - convert to scatter"""
        coords = np.argwhere(filled)
        
        if facecolors is not None:
            # Extract colors from facecolors array
            colors = []
            for coord in coords:
                color = facecolors[tuple(coord)]
                if color.ndim == 1 and len(color) >= 3:
                    colors.append(color[:3])
                else:
                    colors.append([1.0, 0.0, 0.0])  # Default red
            colors = np.array(colors)
        else:
            colors = np.tile([1.0, 0.0, 0.0], (len(coords), 1))
        
        # Apply data reduction
        reducer = SmartDataReducer()
        if len(coords) > 25000:
            coords, colors = reducer.reduce_for_visualization(coords, colors, 25000)
        
        # Store as scatter plot data
        self.plots_data.append({
            'type': 'voxels',
            'coords': coords,
            'colors': colors,
            'size': 4,  # Slightly larger for voxels
            'alpha': 0.8
        })
    
    def set_title(self, title):
        self.title = title
    
    def set_xlabel(self, label):
        self.xlabel = label
        
    def set_ylabel(self, label):
        self.ylabel = label
        
    def set_zlabel(self, label):
        self.zlabel = label
    
    def set_xlim(self, limits):
        pass  # Plotly handles this automatically
        
    def set_ylim(self, limits):
        pass
        
    def set_zlim(self, limits):
        pass
    
    def set_box_aspect(self, aspect):
        pass  # Plotly handles this automatically
    
    def get_xlim(self):
        return [0, 100]  # Dummy values for compatibility
    
    def get_ylim(self):
        return [0, 100]
    
    def get_zlim(self):
        return [0, 100]
    
    def clear(self):
        """Clear the axis"""
        self.plots_data = []
        self.title = ""
    
    def _build_plotly_figure(self):
        """Build the Plotly figure from stored plot data"""
        if not self.plots_data:
            return None
        
        # Combine all plot data
        all_coords = []
        all_colors = []
        
        for plot_data in self.plots_data:
            all_coords.append(plot_data['coords'])
            all_colors.append(plot_data['colors'])
        
        if all_coords:
            coords = np.vstack(all_coords)
            colors = np.vstack(all_colors)
            
            # Create Plotly figure
            renderer = OptimizedPlotlyRenderer()
            fig = renderer.create_optimized_plot(coords, colors, self.title)
            return fig
        
        return None


class OptimizedCanvas:
    """Canvas replacement for Plotly integration"""
    
    def __init__(self, figure, master, renderer):
        self.figure = figure
        self.master = master
        self.renderer = renderer
        self.tk_widget = None
        self.html_file = None
        self._create_widget()
    
    def _create_widget(self):
        """Create the tkinter widget container"""
        # Create a frame to hold our "canvas"
        self.tk_widget = tk.Frame(self.master, bg='white')
        
        # Add a label indicating this is a web-based visualization
        info_frame = tk.Frame(self.tk_widget, bg='lightblue', height=30)
        info_frame.pack(fill='x', side='top')
        info_frame.pack_propagate(False)
        
        info_label = tk.Label(info_frame, 
                             text="🚀 Optimized WebGL Visualization", 
                             bg='lightblue', 
                             font=('Arial', 10, 'bold'))
        info_label.pack(pady=5)
        
        # Add button to open in browser
        button_frame = tk.Frame(self.tk_widget, bg='white')
        button_frame.pack(fill='both', expand=True)
        
        open_button = tk.Button(button_frame,
                               text="Open 3D Visualization in Browser",
                               command=self._open_in_browser,
                               font=('Arial', 12),
                               bg='#4a90e2',
                               fg='white',
                               pady=10)
        open_button.pack(expand=True)
        
        # Add performance info
        perf_label = tk.Label(button_frame,
                             text="Optimized rendering with automatic data reduction\nfor improved performance",
                             bg='white',
                             font=('Arial', 9),
                             fg='gray')
        perf_label.pack(pady=5)
    
    def _open_in_browser(self):
        """Open the visualization in the default browser"""
        if self.html_file and os.path.exists(self.html_file):
            webbrowser.open(f'file://{os.path.abspath(self.html_file)}')
        else:
            self.draw()  # Generate the plot first
            if self.html_file:
                webbrowser.open(f'file://{os.path.abspath(self.html_file)}')
    
    def draw(self):
        """Draw the plot - generate HTML file"""
        if self.figure.axes:
            ax = self.figure.axes[0]
            plotly_fig = ax._build_plotly_figure()
            if plotly_fig:
                self.html_file = self.renderer.save_html_temp(plotly_fig)
                
                # Update button text to indicate it's ready
                for widget in self.tk_widget.winfo_children():
                    if isinstance(widget, tk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, tk.Button):
                                child.config(text="🌐 View 3D Visualization in Browser (Ready!)",
                                           bg='#28a745')
    
    def get_tk_widget(self):
        return self.tk_widget
    
    def mpl_connect(self, event, callback):
        """Dummy method for compatibility"""
        pass


class OptimizedVisualizationManager:
    """Main manager for optimized visualizations"""
    
    def __init__(self):
        self.renderer = OptimizedPlotlyRenderer()
        self.background_processor = BackgroundProcessor()
        self.data_reducer = SmartDataReducer()
        
    def create_figure(self, figsize=(8, 6), dpi=100):
        """Create an optimized figure"""
        return self.renderer.create_figure_replacement(figsize, dpi)
    
    def create_canvas(self, figure, master):
        """Create an optimized canvas"""
        return OptimizedCanvas(figure, master, self.renderer)
    
    def process_spine_async(self, image_data, spine_number, rgb_colors, callback=None):
        """Process spine data asynchronously"""
        return self.background_processor.process_spine_data_async(
            image_data, spine_number, rgb_colors, callback
        )
    
    def cleanup(self):
        """Cleanup resources"""
        self.renderer.cleanup_temp_files()
        self.background_processor.shutdown()


# Global instance for easy access
_global_viz_manager = OptimizedVisualizationManager()

# Drop-in replacement functions
def figure(figsize=(8, 6), dpi=100):
    """Drop-in replacement for plt.figure()"""
    return _global_viz_manager.create_figure(figsize, dpi)

def FigureCanvasTkAgg_Optimized(figure, master):
    """Drop-in replacement for FigureCanvasTkAgg"""
    return _global_viz_manager.create_canvas(figure, master)

def cleanup_optimizations():
    """Call this when shutting down the application"""
    _global_viz_manager.cleanup() 