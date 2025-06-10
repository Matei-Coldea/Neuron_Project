"""
This module provides a class-based interface for the functions in the Extract_Figures_FV.py file.
It is intended to be used as a library to access the functionalities of the original file in a more organized way.
"""

import Extract_Figures_FV as eff

class ImageProcessing:
    def __init__(self):
        """Initializes the ImageProcessing class."""

    def crie_matriz_3D(self, n_imagens, n_linhas, n_colunas, valor):
        """Creates a 3D matrix with the given dimensions and initial value."""
        return eff.crie_matriz_3D(n_imagens, n_linhas, n_colunas, valor)

    def find_far_point(self, dk_connect, di_connect, dj_connect, matrix, specific_number):
        """Finds the farthest point in the matrix with a specific number from a given point."""
        return eff.Find_far_Point(dk_connect, di_connect, dj_connect, matrix, specific_number)

    def find_mid_point_by_arithmetic_mean(self, matrix, specific_number):
        """Finds the midpoint of points with a specific number in the matrix using the arithmetic mean."""
        return eff.Find_Mid_Point_by_arithmetic_mean(matrix, specific_number)

    def check_image(self, tiff_path):
        """Checks if the image is in a supported format."""
        return eff.CkeckImage(tiff_path)

    def open_tiff_fig(self, tiff_path):
        """Opens a TIFF figure and returns the data."""
        return eff.Open_TIFF_fig(tiff_path)

    def import_3d_segment_from_tiff_figure(self, tiff_path):
        """Imports a 3D segment from a TIFF figure."""
        return eff.Import_3D_segment_from_tiff_figure(tiff_path)

class Plotting:
    def __init__(self):
        """Initializes the Plotting class."""

    def plot_matrix_scatter(self, matrix, spine_color):
        """Creates a 3D scatter plot of the matrix."""
        eff.Plot_matrix_scatter(matrix, spine_color)

    def plot_3d_line_test(self, x1, y1, z1, x2, y2, z2):
        """Plots a 3D line for testing purposes."""
        eff.Plot_3D_Line_Test(x1, y1, z1, x2, y2, z2)

    def plot_3d_matrix_line_test(self, matrix, spine_color, i1, j1, k1, i2, j2, k2):
        """Plots a 3D matrix line for testing purposes."""
        eff.Plot_3D_Matrix_Line_Test(matrix, spine_color, i1, j1, k1, i2, j2, k2)

    def plot_matrix_surface(self, matrix):
        """Creates a 3D surface plot of the matrix."""
        eff.Plot_matrix_Surface(matrix)

class DataExport:
    def __init__(self):
        """Initializes the DataExport class."""

    def export_spine_as_text(self, tiff_path_output_txt, matrix, data_in_spine):
        """Exports the spine data as a text file."""
        eff.Export_Spine_as_Text(tiff_path_output_txt, matrix, data_in_spine)

    def export_spine_as_tiff(self, tiff_path_output_tiff, matrix, data_in_spine):
        """Exports the spine data as a TIFF file."""
        eff.Export_Spine_as_tiff(tiff_path_output_tiff, matrix, data_in_spine)

class Utilities:
    def __init__(self):
        """Initializes the Utilities class."""

    def get_integer_list(self):
        """Gets a list of integers from the user."""
        return eff.get_integer_list()

    def generate_estimations(self, spine_number, base_name, data_in_spine):
        """Generates estimations for the spine data."""
        return eff.Geneate_Estimations(spine_number, base_name, data_in_spine)

    def distance_3d(self, x1, y1, z1, x2, y2, z2):
        """Calculates the 3D distance between two points."""
        return eff.distance_3d(x1, y1, z1, x2, y2, z2)

    def find_mid_point(self, matrix, specific_number):
        """Finds the midpoint of points with a specific number in the matrix."""
        return eff.Find_Mid_Point(matrix, specific_number)

    def value_point_in_matrix(self, matrix, x, y, z):
        """Returns the value of a point in the matrix."""
        return eff.Value_Point_In_Matrix(matrix, x, y, z)

class MainExecution:
    def __init__(self):
        """Initializes the MainExecution class."""

    def generate_3d_segment_library(self, tiff_path):
        """Generates a 3D segment library from a TIFF file."""
        eff.Generate_3D_Segment_Library(tiff_path)

    def main(self):
        """Runs the main execution of the program."""
        eff.main()

if __name__ == '__main__':
    # Example of how to use the classes
    main_executor = MainExecution()
    main_executor.main() 