if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage

def normalize_to_uint8(data):
    minimum_value = data.min()
    maximum_value = data.max()
    return ((data - minimum_value) / (maximum_value - minimum_value + 1e-12) * 255).astype(np.uint8)

def convert_to_qpixmap(image_array):
    image_contiguous = np.ascontiguousarray(image_array)
    
    if len(image_contiguous.shape) == 3:
        if image_contiguous.shape[2] == 3 and image_contiguous.dtype == np.uint8:
            image_contiguous = cv2.cvtColor(image_contiguous, cv2.COLOR_BGR2RGB)
        height, width, channels = image_contiguous.shape
        qt_image = QImage(image_contiguous.data, width, height, channels * width, QImage.Format.Format_BGR888)
    else:
        height, width = image_contiguous.shape
        qt_image = QImage(image_contiguous.data, width, height, width, QImage.Format.Format_Grayscale8)
        
    return QPixmap.fromImage(qt_image.copy())
