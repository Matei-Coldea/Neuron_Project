class DataInImage:
    """
    Class for handling metadata of 3D figure properties.
    """
    tag_dict = {
        'Number_of_Layers': 'num_layers',
        'Image_Height': 'height',
        'Image_Width': 'width',
        'X_Resolution': 'x_resolution',
        'Y_Resolution': 'y_resolution',
        'Z_Resolution': 'z_resolution',
        'Resolution_Unit': 'resolution_unit',
        'Volume': 'volume',
        'Volume_unit': 'volume_unit',
        'Surface': 'surface',
        'Surface_unit': 'surface_unit',
        'L': 'L',
        'd': 'd',
        'Spine_Color': 'spine_color',
        'Connection_Point': 'point_connect',
        'Spine_Middle_Point': 'point_middle',
        'Spine_Far_Point': 'point_far',
        'Connection_Is_Inner': 'point_connect_value',
        'Description': 'description'
    }

    def __init__(self):
        self.num_layers = None
        self.height = None
        self.width = None
        self.x_resolution = None
        self.y_resolution = None
        self.z_resolution = None
        self.resolution_unit = "nm"
        self.volume = None
        self.volume_unit = "um3"
        self.surface = None
        self.surface_unit = "um2"
        self.L = None
        self.d = None
        self.spine_color = (255, 0, 0)
        self.point_connect = (None, None, None)
        self.point_middle = (None, None, None)
        self.point_far = (None, None, None)
        self.point_connect_value = None
        self.description = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Attribute {key} not found in the class.")

    def update_from_strings(self, attr_value_dict):
        """
        Update attributes from a dictionary of strings, converting to the appropriate type.
        """
        for attr, value_str in attr_value_dict.items():
            if not hasattr(self, attr):
                continue  # Skip unknown attributes
            # Determine the type of the attribute
            current_value = getattr(self, attr)
            attr_type = type(current_value)
            if current_value is None:
                # Assume default types based on attribute name
                if attr in ['num_layers', 'height', 'width', 'point_connect_value']:
                    attr_type = int
                elif attr in ['x_resolution', 'y_resolution', 'z_resolution', 'volume', 'surface', 'L', 'd']:
                    attr_type = float
                elif attr in ['point_connect', 'point_middle', 'point_far', 'spine_color']:
                    attr_type = tuple
                else:
                    attr_type = str

            if attr_type is tuple:
                # Handle tuples, e.g., "(1, 2, 3)"
                try:
                    # Remove parentheses and split by comma
                    value = tuple(map(float, value_str.strip('()').split(',')))
                    # If original tuple contains ints, convert to ints
                    if all(isinstance(x, int) for x in current_value or []):
                        value = tuple(map(int, value))
                except ValueError:
                    value = current_value  # Keep original value if parsing fails
            elif attr_type is int:
                try:
                    value = int(float(value_str))  # Convert via float to handle inputs like "1.0"
                except ValueError:
                    value = current_value  # Keep original value if parsing fails
            elif attr_type is float:
                try:
                    value = float(value_str)
                except ValueError:
                    value = current_value  # Keep original value if parsing fails
            else:
                # Keep as string
                value = value_str
            setattr(self, attr, value)

    def print_data(self):
        data = self.tag_dict
        for key, value in data.items():
            print(f"{key}: {getattr(self, value)}") 