import json
import numpy as np


def color_from_index(index):
    # Generate a color from anindex using hash
    import hashlib
    color = hashlib.md5(str(index).encode()).hexdigest()
    return '#' + color[:6]

# Filter landmarks to exclude those that are out of the image bounds
def filter_landmarks(landmarks, image_size):
    valid_indices = (landmarks[:, 0] >= 0) & (landmarks[:, 0] < image_size) & \
                    (landmarks[:, 1] >= 0) & (landmarks[:, 1] < image_size)
    return landmarks[valid_indices]

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        
        return json.JSONEncoder.default(self, obj)
