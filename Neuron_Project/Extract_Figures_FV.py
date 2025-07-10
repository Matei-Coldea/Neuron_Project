import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tifffile
from PIL import Image
import json

from data_in_image import DataInImage

# from Import3DSpine import Find_Mid_Point_by_arithmetic_mean, Find_far_Point


# Define global variables
# It would be better if it's removed from here (It would be better if it's not a global variable
rgb_colors = None
image_uploaded = None
image_uploaded_edge = None

# Global flag to control matplotlib display - set to False to suppress plots
SHOW_MATPLOTLIB_PLOTS = True


# Remove it when not in use.
def crie_matriz_3D(n_imagens, n_linhas, n_colunas, valor):
    # image_uploaded_edge=crie_matriz_3D(depth,height,width,0) # (n_fig,y,x,0)  y eh linha e x eh coluna
    # image_uploaded_edge = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(depth)]
    ''' (int, int, int, valor) -> matriz (lista de listas)

    Cria e retorna uma matriz com n_linhas linha e n_colunas
    colunas em que cada elemento é igual ao valor dado.
    '''

    matriz = []  # lista vazia
    for k in range(n_imagens):
        # cria a linha i
        imagens = []  # lista vazia
        for i in range(n_linhas):
            # cria a linha i
            linha = []
            for j in range(n_colunas):
                linha += [valor]
            imagens += [linha]
        # coloque linha na matriz
        matriz += [imagens]

    return matriz


# mk, mi, mj, max_distance = Find_far_Point(dk_connect, di_connect, dj_connect, spine_3D_edge, 1)
def Find_far_Point(dk_connect, di_connect, dj_connect, matrix, specific_number):
    # Lists to store coordinates
    coords = []

    # Traverse through the 3D matrix to find all coordinates with the specific number
    # Erro aqui por nao ter shape sexta
    # for i in range(matrix.shape[0]):  # Iterate over the first dimension
    #    for j in range(matrix.shape[1]):  # Iterate over the second dimension
    #        for k in range(matrix.shape[2]):  # Iterate over the third dimension

    # solucao (k, i, j)
    for i in range(len(matrix)):  # Iterate over the first dimension
        for j in range(len(matrix[0])):  # Iterate over the second dimension
            for k in range(len(matrix[0][0])):  # Iterate over the third dimension
                if matrix[i][j][k] == specific_number:
                    coords.append((i, j, k))

    # Calculate the maximum distance and the corresponding point
    max_distance = 0
    farthest_point = None
    # Given point as a numpy array for easy calculation
    given_point = np.array([dk_connect, di_connect, dj_connect])

    for point in coords:
        # Calculate distance
        distance = np.linalg.norm(given_point - np.array(point))

        # Check if this is the maximum distance
        if distance > max_distance:
            max_distance = distance
            farthest_point = point

    mk, mi, mj = (farthest_point)

    print(f"mk: {mk}")
    print(f"mi: {mi}")
    print(f"mj: {mj}")
    # Return
    return mk, mi, mj, max_distance


def Find_Mid_Point_by_arithmetic_mean(matrix, specific_number):
    # Lists to store coordinates
    coords = []

    # Traverse through the 3D matrix to find all coordinates with the specific number
    # solucao (k, i, j)
    for i in range(len(matrix)):  # Iterate over the first dimension
        for j in range(len(matrix[0])):  # Iterate over the second dimension
            for k in range(len(matrix[0][0])):  # Iterate over the third dimension
                if matrix[i][j][k] == specific_number:
                    coords.append((i, j, k))

    if coords:
        # Extract i, j, k values
        i_values, j_values, k_values = zip(*coords)

        i_mean = sum(i_values) / len(i_values)
        j_mean = sum(j_values) / len(j_values)
        k_mean = sum(k_values) / len(k_values)

        print(f"i_mean: {i_mean}")
        print(f"j_mean: {j_mean}")
        print(f"k_mean: {k_mean}")

        # Compute max and min for i, j, k
        # i_max, i_min = max(i_values), min(i_values)
        # j_max, j_min = max(j_values), min(j_values)
        # k_max, k_min = max(k_values), min(k_values)

        # Return all max and min values
        return i_mean, j_mean, k_mean
    else:
        print("The specific number is not present in the matrix.")
        # Return None for all values if the specific number is not present
        return None, None, None


def CkeckImage(tiff_path):
    try:
        # Open the image
        with Image.open(
                tiff_path) as img:  # with: ensures that the file is properly closed after the block of code is executed
            # Check if the image is in 'P' mode
            if img.mode == 'P':
                return img.mode

            else:
                raise ValueError('The image is not in indexed color mode suported (P).')

        ''' TODO
        TIFF or TIF Options
        1: 1-bit pixels, black and white, stored with one pixel per byte.
        L: 8-bit pixels, grayscale.
        P: 8-bit pixels, mapped to any other mode using a color palette.
        RGB: 3x8-bit pixels, true color.
        RGBA: 4x8-bit pixels, true color with transparency mask.
        CMYK: 4x8-bit pixels, color separation.
        YCbCr: 3x8-bit pixels, color video format.
        I: 32-bit signed integer pixels.
        F: 32-bit floating point pixels.
        '''
    except Exception as e:
        print(f"Error: {e}")


import ast


def get_integer_list():
    while True:
        user_input = input("Which segment would you like to import? (e.g., [2] or [1, 2, 4, 8]): ")

        # Try to parse the input as a list
        try:
            numbers = ast.literal_eval(user_input)
            if isinstance(numbers, list) and all(isinstance(x, int) for x in numbers):
                return numbers
            else:
                print("Invalid input. Please enter a list of integers.")
        except (SyntaxError, ValueError):
            # If parsing fails, process as space-separated integers
            try:
                numbers = list(map(int, user_input.split()))
                return numbers
            except ValueError:
                print("Invalid input. Please enter integers separated by spaces or a valid list of integers.")


def Plot_matrix_scatter(matrix, spine_color):
    # Create a 3D grid
    # Get the indices where the matrix value is 1 (boundary or inside segment) or 2 (connection with other segment)
    matrix = np.array(matrix)
    x, y, z = np.nonzero(matrix)

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point
    # Convert RGB tuple to a format Matplotlib understands
    spine_color_normalized = tuple(c / 255 for c in spine_color)

    # Update scatter plot call to use the normalized color
    ax.scatter(-x[matrix[x, y, z] == 1], z[matrix[x, y, z] == 1], -y[matrix[x, y, z] == 1],
               color=spine_color_normalized, marker='o', label='Spine')
    ax.scatter(-x[matrix[x, y, z] == 2], z[matrix[x, y, z] == 2], -y[matrix[x, y, z] == 2],
               c='black', marker='o', label='Connection')

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Plot of 3D Matrix')

    # Show the plot only if flag is enabled
    if SHOW_MATPLOTLIB_PLOTS:
        # plt.show()  # Disabled to avoid GUI pop-ups
        pass
    else:
        plt.close(fig)  # Close figure to free memory


''' #Not good layout
def Plot_matrix_Surface(matrix):

    matrix=np.array(matrix)

    # Create a mesh grid for the 3D matrix
    x, y, z = np.indices(matrix.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = matrix.flatten()

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot surfaces for value 1
    mask_1 = values == 1
    x_1 = x[mask_1]
    y_1 = y[mask_1]
    z_1 = z[mask_1]

    if len(x_1) > 0:
        ax.plot_trisurf(x_1, y_1, z_1, color='red', alpha=0.5, edgecolor='none', label='Value 1')

    # Plot surfaces for value 2
    mask_2 = values == 2
    x_2 = x[mask_2]
    y_2 = y[mask_2]
    z_2 = z[mask_2]

    if len(x_2) > 0:
        ax.plot_trisurf(x_2, y_2, z_2, color='blue', alpha=0.5, edgecolor='none', label='Value 2')

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Surface Plot of 3D Matrix (Values 1 and 2)')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
'''


# Just for text purposes (will be removed)
def Plot_3D_Line_Test(x1, y1, z1, x2, y2, z2):
    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter([x1, x2], [y1, y2], [z1, z2], color='red')  # Points

    # Plot the line connecting the points
    ax.plot([x1, x2], [y1, y2], [z1, z2], color='blue')  # Line

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    ax.set_title('3D Line Connecting Two Points')

    # Show the plot only if flag is enabled
    if SHOW_MATPLOTLIB_PLOTS:
        # plt.show()  # Disabled to avoid GUI pop-ups
        pass
    else:
        plt.close(fig)  # Close figure to free memory


# Just for text purposes (will be removed)
def Plot_3D_Matrix_Line_Test(matrix, spine_color, i1, j1, k1, i2, j2, k2):
    matrix = np.array(matrix)
    x, y, z = np.nonzero(matrix)  # (x,y,z)=(k, i, j)

    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point
    # Convert RGB tuple to a format Matplotlib understands
    spine_color_normalized = tuple(c / 255 for c in spine_color)

    # Update scatter plot call to use the normalized color
    ax.scatter(-x[matrix[x, y, z] == 1], z[matrix[x, y, z] == 1], -y[matrix[x, y, z] == 1],
               color=spine_color_normalized, marker='o', label='Spine')
    ax.scatter(-x[matrix[x, y, z] == 2], z[matrix[x, y, z] == 2], -y[matrix[x, y, z] == 2],
               c='black', marker='o', label='Connection')

    # Plot the points
    ax.scatter([-k1, -k2], [j1, j2], [-i1, -i2], color='blue')  # Points

    # Plot the line connecting the points
    ax.plot([-k1, -k2], [j1, j2], [-i1, -i2], color='blue')  # Line

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    ax.set_title('3D Plot of 3D Matrix and Line Connecting Two Points')

    # Show the plot only if flag is enabled
    if SHOW_MATPLOTLIB_PLOTS:
        # plt.show()  # Disabled to avoid GUI pop-ups
        pass
    else:
        plt.close(fig)  # Close figure to free memory


def Export_Spine_as_Text(tiff_path_output_txt, matrix, data_in_spine):
    # Save the spine information and spacial data

    # Create and write to the text file
    with open(tiff_path_output_txt, 'w') as file:

        # See Class_Data_In_Image.py
        # Write the information

        depth = len(matrix)
        height = len(matrix[0])
        # Write the information
        file.write(f"# Standardization file Generated by NEURON Simulator 2024 \n")
        file.write(f"# Reading instructions \n# 0: empty space. \n# 1: delimited region \n# 2: connection region \n")

        data = data_in_spine.tag_dict
        for tag, value in data.items():
            # data_in_image.pick(tag)
            # file.write(f"# {tag}: {data_in_spine.get_value_by_tag(tag) }\n")
            file.write(f"# {tag}: {getattr(data_in_spine, value)}\n")

        # data = data_in_spine.get_data()
        # Write data to file
        # for key, value in data.items():
        #    file.write(f"# {key}: {value}\n")

        # Skip saving raw image matrix to reduce file size
        file.write("# Image: <omitted to reduce file size>\n")
    print("Data has been written to ", tiff_path_output_txt)


# Uncomment it only if it becomes useful.
'''
def map_values_to_colors(matrix, color_map):
    """
    Map each value in the matrix to a corresponding RGB color using the provided color map.

    :param matrix: 2D numpy array with values to map.
    :param color_map: Dictionary mapping values to RGB colors.
    :return: 3D numpy array with RGB colors.
    """
    height, width = matrix.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for value, color in color_map.items():
        rgb_image[matrix == value] = color

    return rgb_image
'''


def Export_Spine_as_tiff(tiff_path_output_tiff, matrix, data_in_spine):
    """
    Save each layer of a 3D matrix as separate pages in a single TIFF file with custom metadata.

    :param matrix: A 3D numpy array of shape (i, j, k).
    :param spine_color: The color in RGB to plot the spine. If NONE use default (Red).
    :param tiff_path_output_tiff: Path to the output TIFF file.
    :param ... : metadata value to be included in the custom TIFF tag.
    """
    # Define color palette (RGB tuples)
    palette = [
        (255, 255, 255),  # White
        (255, 0, 0),  # Red
        (0, 0, 0)  # Black
    ]
    palette[1] = data_in_spine.spine_color

    # Flatten the palette list to the format required by Pillow
    flat_palette = [value for color in palette for value in color]

    # num_layers, height, width = matrix.shape
    num_layers = len(matrix)
    height = len(matrix[0])
    width = len(matrix[0][0])

    metadata = data_in_spine.tag_dict
    text_output = "{"
    for tag, attr in metadata.items():
        value = getattr(data_in_spine, attr)
        if isinstance(value, (int, float)):  # Numeric values
            attr_value = str(value)
        elif isinstance(value, (tuple)):  # Numeric values
            attr_value = '"' + str(value) + '"'
        elif value is None:  # Handle None
            attr_value = 'null'  # Ou None?
        else:
            attr_value = '"' + value + '"'

        text_output += '"' + tag + '": ' + attr_value + ', '  # Append the word and a space

    # Remove the trailing comma and space, and close the brace
    text_output = text_output.rstrip(', ') + "}"

    '''
    metadata = data_in_spine.get_data()
    # Convert metadata dictionary to string
    # Format the final text output
    # See Class_Data_In_Image.py
    #text_output = f"{{'Description':'{{\"Description\": \"{metadata['Description']}\", \"X_Resolution\": {metadata['X_Resolution']}, \"Y_Resolution\": {metadata['Y_Resolution']}, \"Z_Resolution\": {metadata['Z_Resolution']}, \"Resolution_Unit\": \"{metadata['Resolution_Unit']}\", \"Volume_Estimate\": {metadata['Volume_Estimate']}, \"Surface_Estimate\": {metadata['Surface_Estimate']}, \"Number of Layers\": {metadata['Number of Layers']}, \"Image Height\": {metadata['Image Height']}, \"Image Width\": {metadata['Image Width']}, \"shape\": [2, 3, 3]}}'}}"
    text_output = f"{{\"Description\": \"{metadata['Description']}\", \"X_Resolution\": {metadata['X_Resolution']}, \"Y_Resolution\": {metadata['Y_Resolution']}, \"Z_Resolution\": {metadata['Z_Resolution']}, \"Resolution_Unit\": \"{metadata['Resolution_Unit']}\", \"Volume\": {metadata['Volume']}, \"Surface\": {metadata['Surface']}, \"Number of Layers\": {metadata['Number_of_Layers']}, \"Image Height\": {metadata['Image_Height']}, \"Image Width\": {metadata['Image_Width']}}}"
    text_output = f"{{\"Description\": \"{metadata['Description']}\", \"X_Resolution\": {metadata['X_Resolution']}, \"Y_Resolution\": {metadata['Y_Resolution']}, \"Z_Resolution\": {metadata['Z_Resolution']}, \"Resolution_Unit\": \"{metadata['Resolution_Unit']}\", \"Volume\": {metadata['Volume']}, \"Surface\": {metadata['Surface']}, \"Number of Layers\": {metadata['Number_of_Layers']}, \"Image Height\": {metadata['Image_Height']}, \"Image Width\": {metadata['Image_Width']}}}"
    '''

    metadata = {'Description': text_output}  # Print the result
    print(text_output)

    # Create a list to hold each image layer
    images = []

    # Iterate through each layer (i)
    for i in range(num_layers):

        # Extract the 2D slice for the current layer
        layer_matrix = matrix[i]

        # Create an image in 'P' mode for each matrix
        height, width = len(layer_matrix), len(layer_matrix[0])
        img = Image.new('P', (width, height))

        # Set the palette
        img.putpalette(flat_palette)

        # Create an array of indices for the palette ??
        index_image = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                index_image[i, j] = layer_matrix[i][j]

        # Put the array data into the image
        img.putdata(index_image.flatten())

        # Append the image to the list of images
        images.append(img)

    # Save the first image with the description for the entire TIFF file
    images[0].save(
        tiff_path_output_tiff,
        save_all=True,
        append_images=images[1:],
        format='TIFF',
        description=metadata.get('Description', '')  # Set file-wide description here
    )

    ################################################################
    ##
    ##
    ##
    ##
    ################################################################


def Open_TIFF_fig(tiff_path):
    global rgb_colors, image_uploaded, image_uploaded_edge
    #
    # Here starts the module to check tiff standar for color.
    # This program just work for mode P (1 number per color).
    # The RGB return 3 or 4 numbers insted 1 per pix. We need to convert this 3 or 4 numbers in 1 and save the RGB in a separade table (a pallet) and then call this module
    mode = CkeckImage(tiff_path)
    # print(mode)

    # def Process_TIFF_P(tiff_path):
    if mode != 'P':
        print("\n\n\nColor standard not yet supported\n\n\n")

    if mode == 'P':
        # Cria classe para armazenar Informacoes carregadas
        data_in_image = DataInImage()

        with Image.open(
                tiff_path) as img:  # with: ensures that the file is properly closed after the block of code is executed
            if img.mode != 'P':
                raise ValueError('The image is not in indexed color mode (P mode).')

            # Ensure the image file contains multiple frames
            if not hasattr(img, "n_frames") or img.n_frames < 1:
                raise ValueError('The TIFF file does not contain multiple frames.')

            # Get the color palette
            palette = img.getpalette()
            # The palette contains RGB values in a flat list

            # Extract RGB triplets from the palette and Normalized it
            num_colors = len(palette) // 3
            rgb_colors = [(palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]) for i in range(num_colors)]

            # ------------------------------------------------------------------
            # Safety net: some label TIFFs store integer indices larger than the
            # palette itself.  If that happens, accessing rgb_colors[idx] later
            # raises an IndexError ("index X is out of bounds for axis 0").
            # We extend the palette with a default colour (white) so every index
            # found in the volume has a valid entry.  This keeps downstream code
            # unchanged.
            # ------------------------------------------------------------------
            # We'll extend the palette later, after we know the real max index
            rgb_colors_normalized = None  # placeholder

            # Print the RGB colors and their indices
            # for index, color in enumerate(rgb_colors):
            #    print(f'Index {index}: RGB {color}')

            try:
                img.tag.get(270)  # This step is mandatory. Do not remove.
                raw_value = img.tag.tags[270]
            except:
                raw_value = 0

            # raw_value = img.tag.get(270)
            # If the raw value is a string and you expect it to be a JSON object
            try:
                # Aqui, alterar aqui!!
                # Attempt to parse the JSON string
                tag_dict = json.loads(raw_value)

                # See Class_Data_In_Image.py
                data = data_in_image.tag_dict
                for key, value in data.items():
                    # Extract values from the dictionary
                    if key in tag_dict:
                        # data_in_image.key = tag_dict[key]
                        setattr(data_in_image, value, tag_dict[key])

                '''
                # See Class_Data_In_Image.py
                # Extract values from the dictionary
                if "Resolution_Unit" in tag_dict:
                    data_in_image.resolution_unit = tag_dict["Resolution_Unit"]

                data_in_image.description = tag_dict.get("Description", None)
                data_in_image.x_resolution = tag_dict.get('X_Resolution', None)
                data_in_image.y_resolution = tag_dict.get("Y_Resolution", None)
                data_in_image.z_resolution = tag_dict.get("Z_Resolution", None)
                data_in_image.resolution_unit = tag_dict.get("Resolution_Unit", None)
                data_in_image.volume = tag_dict.get("Volume", None)
                data_in_image.surface = tag_dict.get("Surface", None)
                data_in_image.num_layers = tag_dict.get("Number_of_Layers", None)
                data_in_image.height = tag_dict.get("Image_Height", None)
                data_in_image.width = tag_dict.get("Image_Width", None)
                '''

            except (json.JSONDecodeError, TypeError):
                # Handle cases where the raw_value isn't valid JSON or is not a string
                # Failed to parse JSON from tag 270
                print("Unable to Upload Data Embedded in the Image.")

            # Prepare an empty list to store each layer's data
            layers = []

            # Iterate through each frame (layer) in the TIFF file
            for frame in range(img.n_frames):
                img.seek(frame)  # Move to the next frame
                # if img.mode != 'P':
                #    raise ValueError(f'Layer {frame} is not in indexed color mode (P mode).')

                # Get image dimensions
                width, height = img.size

                # Get image data (index values)
                image_data = img.getdata()

                # Convert image data to a 2D array of indices
                indices = np.array(list(image_data)).reshape((height, width))

                # Append the 2D array to the list
                layers.append(indices)

            # Stack layers to form a 3D matrix
            image_uploaded = np.stack(layers, axis=0)

            # ------------------------------------------------------------------
            # Palette safety extension (must happen *after* we have image_uploaded)
            # ------------------------------------------------------------------
            max_idx_in_image = int(image_uploaded.max()) if image_uploaded.size else 0

            if max_idx_in_image >= len(rgb_colors):
                default_colour = (255, 255, 255)
                rgb_colors.extend([default_colour] * (max_idx_in_image - len(rgb_colors) + 1))

            rgb_colors_normalized = np.array(rgb_colors, dtype=np.float32) / 255.0

        # Module just for test

        # Get the dimensions
        depth, height, width = image_uploaded.shape

        image_uploaded_edge = np.copy(image_uploaded)

        for i in range(1, height - 1, 1):  # linhas 1024
            for j in range(1, width - 1, 1):
                for k in range(1, depth - 1, 1):
                    if image_uploaded_edge[k][i][j] != 0:
                        if (image_uploaded[k - 1][i][j] == image_uploaded[k][i][j]) and (
                                image_uploaded[k + 1][i][j] == image_uploaded[k][i][j]) and (
                                image_uploaded[k][i + 1][j] == image_uploaded[k][i][j]) and (
                                image_uploaded[k][i - 1][j] == image_uploaded[k][i][j]) and (
                                image_uploaded[k][i][j - 1] == image_uploaded[k][i][j]) and (
                                image_uploaded[k][i][j + 1] == image_uploaded[k][i][j]):
                            image_uploaded_edge[k][i][j] = 0
                            # print("aqui")
            # print(i, "\n")

        # Create a 3D grid
        z, x, y = np.indices(image_uploaded.shape)  # depth, height, width

        # Mask for non-transparent regions (non-zero pixels)
        mask = image_uploaded_edge != 0

        # Extract coordinates and colors for plotting
        x_flat = x[mask].flatten()
        y_flat = y[mask].flatten()
        z_flat = z[mask].flatten()
        colors_flat = image_uploaded[mask].flatten()

        # Get Normalized colors to [0, 1] range for RGB
        colors_rgb = [rgb_colors_normalized[idx] for idx in colors_flat]

        # Extract unique colors for the legend
        unique_colors = np.unique(image_uploaded_edge[image_uploaded_edge != 0])

        # Create a figure with a GridSpec layout
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])  # 3:1 ratio for the 3D plot and legend

        # 3D Plot
        ax1 = fig.add_subplot(gs[0], projection='3d')
        # voxels = ax1.scatter( z_flat, -x_flat, y_flat, c=colors_rgb, marker='o') # (z, y, x)
        # voxels = ax1.scatter( x_flat, y_flat, z_flat, c=colors_rgb, marker='o') # (z, y, x)
        # voxels = ax1.scatter( y_flat, z_flat, x_flat, c=colors_rgb, marker='o') # (z, y, x)
        # voxels = ax1.scatter( y_flat, -z_flat, -x_flat, c=colors_rgb, marker='o') # (z, y, x)
        # voxels = ax1.scatter( y_flat, z_flat, -x_flat, c=colors_rgb, marker='o') # (z, y, x)
        voxels = ax1.scatter(-z_flat, y_flat, -x_flat, c=colors_rgb, marker='o')  # (z, y, x)
        # voxels = ax1.scatter( z_flat, y_flat, -x_flat, c=colors_rgb, marker='o') # (z, y, x)
        # ax1.scatter(z_flat, y_flat, x_flat, c=colors_rgb, marker='o')
        # ax1.set_title('3D Scatter Plot')
        # Set titles and labels
        ax1.set_title('3D Scatter Plot')
        ax1.set_xlabel('X value')
        ax1.set_ylabel('Y value')
        ax1.set_zlabel('Z value')

        # Legend Plot
        ax2 = fig.add_subplot(gs[1])
        for col in unique_colors:
            ax2.plot([], [], color=rgb_colors_normalized[col], label=f'Value: {col}')
        ax2.legend()
        ax2.set_title('Value Legend')
        ax2.axis('off')  # Hide the axis for a cleaner look

        # Adjust layout
        plt.tight_layout()
        print("Figure Loaded \n")
        # plt.show()  # Disabled to avoid GUI pop-ups
        return data_in_image


# Aqui melhorar
def Import_3D_segment_from_tiff_figure(tiff_path):
    data_in_image = Open_TIFF_fig(tiff_path)
    return data_in_image


def Geneate_Estimations(spine_number, base_name, data_in_spine):
    global rgb_colors, image_uploaded, image_uploaded_edge

    # Create the new file path by appending '_output'
    tiff_path_output_txt = f"{base_name}_{spine_number}_output.csv"
    tiff_path_output_tiff = f"{base_name}_{spine_number}_output.tiff"

    spine_color = rgb_colors[spine_number]  # save the collor of the spine from original figure
    data_in_spine.spine_color = spine_color
    # Calcula
    height = len(image_uploaded_edge[0])
    width = len(image_uploaded_edge[0][0])
    depth = len(image_uploaded_edge)
    max_ijk = [0, 0, 0]
    min_ijk = [height, width, depth]
    for i in range(height):  # linhas 1024
        for j in range(width):
            for k in range(depth):
                if image_uploaded_edge[k][i][j] == spine_number:
                    # confere max
                    if i > max_ijk[0]: max_ijk[0] = i
                    if j > max_ijk[1]: max_ijk[1] = j
                    if k > max_ijk[2]: max_ijk[2] = k
                    if i < min_ijk[0]: min_ijk[0] = i
                    if j < min_ijk[1]: min_ijk[1] = j
                    if k < min_ijk[2]: min_ijk[2] = k
                    # image_uploaded_edge[k][i][j]=0
                    # print("aqui")
        # print(i, "\n")

    di = (max_ijk[0] - min_ijk[0] + 1)
    dj = (max_ijk[1] - min_ijk[1] + 1)
    dk = (max_ijk[2] - min_ijk[2] + 1)

    data_in_spine.height = di
    data_in_spine.width = dj
    data_in_spine.num_layers = dk

    # Estimacao simples de comprimento L
    dmax = max(di, dj, dk)
    resolution_estimate = 1000.0 / (dmax - 2)

    ##print('k = ' , (max_ijk[2]-min_ijk[2]+1), "\n", 'i = ' , (max_ijk[0]-min_ijk[0]+1), "\n", 'j = ' , (max_ijk[1]-min_ijk[1]+1), "\n")
    ##print('dmax = ' , dmax, "\n")

    # Verify the data
    # data = image_data.get_data()

    spine_3D = crie_matriz_3D(max_ijk[2] - min_ijk[2] + 3, max_ijk[0] - min_ijk[0] + 3, max_ijk[1] - min_ijk[1] + 3,
                              0)  # (depth,height,width) (k,i, j)
    spine_3D_edge = crie_matriz_3D(max_ijk[2] - min_ijk[2] + 3, max_ijk[0] - min_ijk[0] + 3,
                                   max_ijk[1] - min_ijk[1] + 3, 0)  # (depth,height,width) (k,i, j)
    spine_3D_volume_estimate = 0
    spine_3D_surface_estimate = 0
    # Copia soh spine min_ijk[1], max_ijk[1]
    for i in range(max_ijk[0] - min_ijk[0] + 1):  # linhas 1024
        for j in range(max_ijk[1] - min_ijk[1] + 1):
            for k in range(max_ijk[2] - min_ijk[2] + 1):
                if image_uploaded[k + min_ijk[2]][i + min_ijk[0]][j + min_ijk[1]] == spine_number:
                    spine_3D[k + 1][i + 1][j + 1] = 1
                    spine_3D_volume_estimate = spine_3D_volume_estimate + 1
                if image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0]][j + min_ijk[1]] == spine_number:
                    spine_3D_edge[k + 1][i + 1][j + 1] = 1
                    spine_3D_surface_estimate = spine_3D_surface_estimate + 1
                    # check connection boundary
                    if k + min_ijk[2] != 0:  # k min
                        if (image_uploaded_edge[k + min_ijk[2] - 1][i + min_ijk[0]][
                                j + min_ijk[1]] != spine_number) and (
                                image_uploaded_edge[k + min_ijk[2] - 1][i + min_ijk[0]][j + min_ijk[1]] != 0):
                            spine_3D_edge[k][i + 1][j + 1] = 2
                            spine_3D[k][i + 1][j + 1] = 2
                    if k + min_ijk[2] != depth - 1:  # k max
                        if (image_uploaded_edge[k + min_ijk[2] + 1][i + min_ijk[0]][
                                j + min_ijk[1]] != spine_number) and (
                                image_uploaded_edge[k + min_ijk[2] + 1][i + min_ijk[0]][j + min_ijk[1]] != 0):
                            spine_3D_edge[k + 2][i + 1][j + 1] = 2
                            spine_3D[k + 2][i + 1][j + 1] = 2

                    if k + min_ijk[0] != 0:  # i min
                        if (image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0] - 1][
                                j + min_ijk[1]] != spine_number) and (
                                image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0] - 1][j + min_ijk[1]] != 0):
                            spine_3D[k + 1][i][j + 1] = 2
                            spine_3D_edge[k + 1][i][j + 1] = 2
                    if k + min_ijk[0] != height - 1:  # i max
                        if (image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0] + 1][
                                j + min_ijk[1]] != spine_number) and (
                                image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0] + 1][j + min_ijk[1]] != 0):
                            spine_3D_edge[k + 1][i + 2][j + 1] = 2
                            spine_3D[k + 1][i + 2][j + 1] = 2

                    if k + min_ijk[1] != 0:  # j min
                        if (image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0]][
                                j + min_ijk[1] - 1] != spine_number) and (
                                image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0]][j + min_ijk[1] - 1] != 0):
                            spine_3D[k + 1][i + 1][j] = 2
                            spine_3D_edge[k + 1][i + 1][j] = 2
                    if k + min_ijk[1] != width - 1:  # k max
                        if (image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0]][
                                j + min_ijk[1] + 1] != spine_number) and (
                                image_uploaded_edge[k + min_ijk[2]][i + min_ijk[0]][j + min_ijk[1] + 1] != 0):
                            spine_3D[k + 1][i + 1][j + 2] = 2
                            spine_3D_edge[k + 1][i + 1][j + 2] = 2

    # Plot_matrix_scatter(spine_3D)
    Plot_matrix_scatter(spine_3D_edge, spine_color)
    # Plot_matrix_scatter(image_uploaded_edge) #x[spine_3D[x, y, z] == 1]

    # Print each attribute on a new line
    print('Data Uploaded:\n')
    data_in_spine.print_data()

    print('Calculating Missing Data...\n')

    # X Y Z Recalcular
    if data_in_spine.x_resolution == None:
        data_in_spine.x_resolution = resolution_estimate
    if data_in_spine.y_resolution == None:
        data_in_spine.y_resolution = resolution_estimate
    if data_in_spine.z_resolution == None:
        data_in_spine.z_resolution = resolution_estimate
    if data_in_spine.resolution_unit == None:
        data_in_spine.resolution_unit = "nm"

    resolution = [0, 0, 0]
    resolution[0] = data_in_spine.x_resolution
    resolution[1] = data_in_spine.y_resolution
    resolution[2] = data_in_spine.z_resolution
    # Plot_matrix_Surface(spine_3D)

    # Volume e Area
    # conferir unidades usadas na literatura e neuron nm ou um
    volume_vortex = resolution[0] * resolution[1] * resolution[2]
    volume_estimate = volume_vortex * (
                spine_3D_volume_estimate - 0.5 * spine_3D_surface_estimate) * 1e-9  # convert nm to um
    surface_vortex = (resolution[0] * (resolution[1] + resolution[2]) + resolution[1] * resolution[2]) / 3.0
    surface_estimate = surface_vortex * spine_3D_surface_estimate * 0.75 * 1e-6  # convert nm to um

    if data_in_spine.volume == None:
        data_in_spine.volume = volume_estimate
    if (data_in_spine.volume_unit == None) and (data_in_spine.resolution_unit == "nm"):
        data_in_spine.volume_unit = "um3"

    if data_in_spine.surface == None:
        data_in_spine.surface = surface_estimate
    if (data_in_spine.surface_unit == None) and (data_in_spine.resolution_unit == "nm"):
        data_in_spine.surface_unit = "um2"

    # L e d
    # Melhorar considerando resolucao
    # k_max, k_min, i_max, i_min, j_max, j_min  = Find_Mid_Point(image_uploaded_edge, 2)
    # k_max, k_min, i_max, i_min, j_max, j_min  = Find_Mid_Point(spine_3D_edge, 2)
    # di_connect = (i_max + i_min) // 2
    # dj_connect = (j_max + j_min) // 2
    # dk_connect = (k_max + k_min) // 2

    dk_connect, di_connect, dj_connect = Find_Mid_Point_by_arithmetic_mean(spine_3D_edge, 2)

    # mk, mi, mj, max_distance = Find_far_Point(dk_connect, di_connect, dj_connect, spine_3D_edge, 1)
    # Aqui estimativas
    # L = distance_3d(di_connect//1, dj_connect//1, dk_connect//1, di//2 , dj//2, dk//2)
    # L=2*L

    mk, mi, mj, L = Find_far_Point(dk_connect, di_connect, dj_connect, spine_3D_edge, 1)
    data_in_spine.L = L * resolution_estimate / 1000.0
    # Aqui resolution

    # d=distance_3d(di_connect, dj_connect, dk_connect, i_max, j_max,  k_max )
    _, _, _, d = Find_far_Point(dk_connect, di_connect, dj_connect, spine_3D_edge, 2)
    # d=d*2
    data_in_spine.d = 2 * d * resolution_estimate / 1000.0

    point_connect = (dk_connect, di_connect, dj_connect)
    # point_middle=(dk//2, di//2, dj//2)
    point_middle = ((mk + dk_connect) // 2, (mi + di_connect) // 2, (mj + dj_connect) // 2)
    point_far = (mk, mi, mj)
    point_connect_value = Value_Point_In_Matrix(spine_3D, round(dk_connect), round(di_connect),
                                                round(dj_connect))  # k, i, j

    data_in_spine.point_connect = (round(point_connect[2]), round(point_connect[1]), round(point_connect[0]))
    # data_in_spine.point_connect=(dj_connect, di_connect, dk_connect)
    # data_in_spine.point_middle=(dk//2, di//2, dj//2)
    # data_in_spine.point_middle=((dj_connect-mj)//2, (di_connect-mi)//2, (dk_connect-mk)//2)
    data_in_spine.point_middle = (round(point_middle[2]), round(point_middle[1]), round(point_middle[0]))
    data_in_spine.point_far = (point_far[2], point_far[1], point_far[0])
    data_in_spine.point_connect_value = Value_Point_In_Matrix(spine_3D, round(dk_connect), round(di_connect),
                                                              round(dj_connect))  # k, i, j

    Plot_3D_Line_Test(di_connect, dj_connect, dk_connect, di // 2, dj // 2, dk // 2)
    Plot_3D_Matrix_Line_Test(spine_3D_edge, spine_color, di_connect, dj_connect, dk_connect, di // 2, dj // 2, dk // 2)
    # print("Estimated Spine Lenght melhorar: ", {dmax}, "\n")
    # print("Estimated Surface: ", {surface_estimate}, "\n")
    # print("Estimated Voulume: ", {volume_estimate}, "\n")

    # Print each attribute on a new line
    print('Data Estimated:\n')
    data_in_spine.print_data()

    print('Would like to change data manualy? yes no\n')
   #

    print('Savind Output (csv and tiff)...\n')
    Export_Spine_as_Text(tiff_path_output_txt, spine_3D, data_in_spine)
    Export_Spine_as_tiff(tiff_path_output_tiff, spine_3D, data_in_spine)

    print("\n\nSpin {spine_number}: Data Saved \n\n")


def distance_3d(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    return distance


# To be removed
def Find_Mid_Point(matrix, specific_number):
    # Lists to store coordinates
    coords = []

    # Traverse through the 3D matrix to find all coordinates with the specific number
    # Erro aqui por nao ter shape sexta
    # for i in range(matrix.shape[0]):  # Iterate over the first dimension
    #    for j in range(matrix.shape[1]):  # Iterate over the second dimension
    #        for k in range(matrix.shape[2]):  # Iterate over the third dimension

    # solucao (k, i, j)
    for i in range(len(matrix)):  # Iterate over the first dimension
        for j in range(len(matrix[0])):  # Iterate over the second dimension
            for k in range(len(matrix[0][0])):  # Iterate over the third dimension
                if matrix[i][j][k] == specific_number:
                    coords.append((i, j, k))

    if coords:
        # Extract i, j, k values
        i_values, j_values, k_values = zip(*coords)

        # Compute max and min for i, j, k
        i_max, i_min = max(i_values), min(i_values)
        j_max, j_min = max(j_values), min(j_values)
        k_max, k_min = max(k_values), min(k_values)

        print(f"i_min: {i_min}, i_max: {i_max}")
        print(f"j_min: {j_min}, j_max: {j_max}")
        print(f"k_min: {k_min}, k_max: {k_max}")
        # Return all max and min values
        return i_max, i_min, j_max, j_min, k_max, k_min
    else:
        print("The specific number is not present in the matrix.")
        # Return None for all values if the specific number is not present
        return None, None, None, None, None, None


# Aqui Erro por k i j
def Value_Point_In_Matrix(matrix, x, y, z):
    # Check if the point is within the bounds of the matrix
    # if (0 <= x < matrix.shape[0] and
    #    0 <= y < matrix.shape[1] and
    #    0 <= z < matrix.shape[2]):
    # Check if the value at the point matches the specified value
    #    return matrix[x, y, z]
    if (0 <= x < len(matrix) and
            0 <= y < len(matrix[0]) and
            0 <= z < len(matrix[0][0])):
        # Check if the value at the point matches the specified value
        return matrix[x][y][z]
    else:
        # Point is outside the bounds of the matrix
        return False


def Generate_3D_Segment_Library(tiff_path):
    # Path to the TIFF file
    # Original file path
    # tiff_path = 'cropped_label_Azul.tif'
    # tiff_path = 'cropped_label_Vermelho.tif'
    # tiff_path = 'label.tif'
    # tiff_path = './Exemplo_01.tiff'
    # tiff_path = 'cropped_label_Azul_compact.tif'
    # tiff_path = 'cropped_label_Vermelho.tif'
    # tiff_path = 'cropped_label_Vermelho_output.tiff'
    # tiff_path = 'raw_nd2_image.nd2'
    # tiff_path = 'label.tif'
    # def plot_tiff_stack_3d(tiff_path):

    # Find the position of the last dot in the file path
    dot_index = tiff_path.rfind('.')

    # Extract the base name and extension
    base_name = tiff_path[:dot_index]
    ext = tiff_path[dot_index:]

    # Create the new file path by appending '_output'
    # tiff_path_output = f"{base_name}_output{ext}"
    # tiff_path_output_txt = f"{base_name}_output.csv"
    # tiff_path_output_tiff = f"{base_name}_output.tiff"

    print("Loading Figure... \n")

    # Read the TIFF file
    data_in_image = Open_TIFF_fig(tiff_path)

    # Get the integer list from the user
    number_list = get_integer_list()

    # Print the result
    print("The list of segments to be imported:", number_list)

    # Seleciona apenas pix do neuronio (uma borda de 1 pix extra)
    for spine_number in number_list:
        data_in_spine = DataInImage()
        # data_in_spine=copy(data_in_image)
        # Copy attributes using __dict__
        data_in_spine.__dict__.update(data_in_image.__dict__)
        Geneate_Estimations(spine_number, base_name, data_in_spine)

    ################################################################
    ##
    ##
    ##
    ##
    ################################################################


def example_single_color_processing():
    """
    Example function demonstrating how to use the single color processing functions.
    """
    # Example 1: Process a specific segment
    tiff_path = 'label.tif'
    
    # Process a specific segment (e.g., segment 1)
    result = Process_Single_Color_Figure(tiff_path, segment_number=1, output_prefix='spine_analysis')
    if result:
        print("Single segment processing completed successfully")
        result.print_data()
    
    # Example 2: Process the first available segment automatically
    result = Process_Single_Color_Figure(tiff_path)
    if result:
        print("Automatic segment processing completed successfully")
    
    # Example 3: Process all segments in the figure
    results = Process_All_Segments_In_Figure(tiff_path, output_prefix='all_spines')
    if results:
        print(f"Processed {len(results)} segments total")
        for seg_num, data in results.items():
            print(f"Segment {seg_num}: Volume = {data.volume}, Surface = {data.surface}")


def main():
    # tiff_path = 'label.tif'
    # tiff_path = 'label_3_output.tiff'
    # tiff_path = 'cropped_label_Vermelho.tif'
    tiff_path = 'cropped_label_Vermelho_2_output.tiff'
    # tiff_path = 'cropped_label_Vermelho_output.tiff'
    # tiff_path = 'cropped_label_Vermelho_output_1_output.tiff'
    # tiff_path = 'cropped_label_Vermelho_output.tiff'
    # tiff_path = 'cropped_label_Vermelho.tif'

    # Original method - interactive segment selection
    # Generate_3D_Segment_Library(tiff_path)
    
    # New method - automatic single color processing
    # Uncomment one of the following lines to test:
    
    # Process first available segment automatically
    # Process_Single_Color_Figure(tiff_path)
    
    # Process a specific segment
    # Process_Single_Color_Figure(tiff_path, segment_number=2)
    
    # Process all segments
    # Process_All_Segments_In_Figure(tiff_path)
    
    # Run example
    # example_single_color_processing()
    
    # For now, run the original method
    Generate_3D_Segment_Library(tiff_path)


def set_matplotlib_display(show_plots: bool):
    """
    Convenience function to control matplotlib plot display.
    
    Args:
        show_plots (bool): True to show matplotlib plots, False to suppress them
    """
    global SHOW_MATPLOTLIB_PLOTS
    SHOW_MATPLOTLIB_PLOTS = show_plots
    print(f"Matplotlib plots {'enabled' if show_plots else 'disabled'}")

def get_matplotlib_display_status():
    """
    Get the current matplotlib display status.
    
    Returns:
        bool: True if plots are shown, False if suppressed
    """
    return SHOW_MATPLOTLIB_PLOTS

def detect_available_segments(image_uploaded):
    """
    Detect all available color segments in the uploaded image.
    
    Args:
        image_uploaded: 3D numpy array representing the TIFF stack
        
    Returns:
        list: List of unique segment numbers (excluding 0 which represents background)
    """
    import numpy as np
    
    # Convert to numpy array if it's not already
    if not isinstance(image_uploaded, np.ndarray):
        image_uploaded = np.array(image_uploaded)
    
    # Get unique values and exclude 0 (background)
    unique_segments = np.unique(image_uploaded)
    available_segments = [int(seg) for seg in unique_segments if seg != 0]
    
    print(f"Available segments detected: {available_segments}")
    return available_segments


def Process_Single_Color_Figure(tiff_path, segment_number=None, output_prefix=None):
    """
    Process a single color figure from a TIFF file, generate estimations and save results with metadata.
    
    Args:
        tiff_path (str): Path to the input TIFF file
        segment_number (int, optional): Specific segment number to process. If None, will process the first available segment
        output_prefix (str, optional): Custom prefix for output files. If None, uses the input filename
        
    Returns:
        DataInImage: The processed spine data with estimations
    """
    global rgb_colors, image_uploaded, image_uploaded_edge
    
    print(f"Processing single color figure: {tiff_path}")
    
    # Find the position of the last dot in the file path
    dot_index = tiff_path.rfind('.')
    base_name = tiff_path[:dot_index] if dot_index != -1 else tiff_path
    
    # Use custom output prefix if provided
    if output_prefix:
        base_name = output_prefix
    
    print("Loading Figure...")
    
    # Read the TIFF file and get image data
    data_in_image = Open_TIFF_fig(tiff_path)
    
    if data_in_image is None:
        print("Error: Could not load image data")
        return None
    
    # Detect available segments
    available_segments = detect_available_segments(image_uploaded)
    
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
    
    # Generate estimations for the selected segment
    Geneate_Estimations(target_segment, base_name, data_in_spine)
    
    print(f"Successfully processed segment {target_segment}")
    print(f"Output files: {base_name}_{target_segment}_output.csv and {base_name}_{target_segment}_output.tiff")
    
    return data_in_spine


def Process_All_Segments_In_Figure(tiff_path, output_prefix=None):
    """
    Process all available segments in a TIFF figure, generating estimations for each.
    
    Args:
        tiff_path (str): Path to the input TIFF file
        output_prefix (str, optional): Custom prefix for output files. If None, uses the input filename
        
    Returns:
        dict: Dictionary mapping segment numbers to their DataInImage objects
    """
    global rgb_colors, image_uploaded, image_uploaded_edge
    
    print(f"Processing all segments in figure: {tiff_path}")
    
    # Find the position of the last dot in the file path
    dot_index = tiff_path.rfind('.')
    base_name = tiff_path[:dot_index] if dot_index != -1 else tiff_path
    
    # Use custom output prefix if provided
    if output_prefix:
        base_name = output_prefix
    
    print("Loading Figure...")
    
    # Read the TIFF file and get image data
    data_in_image = Open_TIFF_fig(tiff_path)
    
    if data_in_image is None:
        print("Error: Could not load image data")
        return None
    
    # Detect available segments
    available_segments = detect_available_segments(image_uploaded)
    
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
            # Generate estimations for this segment
            Geneate_Estimations(segment_number, base_name, data_in_spine)
            processed_segments[segment_number] = data_in_spine
            print(f"Successfully processed segment {segment_number}")
        except Exception as e:
            print(f"Error processing segment {segment_number}: {e}")
            continue
    
    print(f"\nCompleted processing {len(processed_segments)} segments")
    return processed_segments

if __name__ == "__main__":
    main()
