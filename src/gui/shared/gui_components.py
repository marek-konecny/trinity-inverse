if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from datetime import datetime
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea

class ViewerWindowWithFileTree(QMainWindow):
    def __init__(self, window_title):
        super().__init__()
        self.setWindowTitle(window_title)
        self.resize(1500, 1000)
        
        central_main_widget = QWidget()
        self.setCentralWidget(central_main_widget)
        self.application_main_layout = QHBoxLayout(central_main_widget)

        self.left_control_panel = QWidget()
        self.left_panel_layout = QVBoxLayout(self.left_control_panel)
        self.application_main_layout.addWidget(self.left_control_panel, stretch=1)

        self.right_file_explorer_panel = QWidget()
        self.right_file_explorer_panel.setFixedWidth(200)
        self.right_panel_layout = QVBoxLayout(self.right_file_explorer_panel)
        self.application_main_layout.addWidget(self.right_file_explorer_panel)

        self.file_explorer_scroll_area = QScrollArea()
        self.file_explorer_scroll_area.setWidgetResizable(True)
        self.file_explorer_scroll_content = QWidget()
        self.file_explorer_scroll_layout = QVBoxLayout(self.file_explorer_scroll_content)
        self.file_explorer_scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.file_explorer_scroll_area.setWidget(self.file_explorer_scroll_content)
        self.right_panel_layout.addWidget(self.file_explorer_scroll_area, stretch=1)

def save_image_prompt(parent_widget, prefix, image_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{prefix}_{timestamp}.png"
    file_path, _ = QFileDialog.getSaveFileName(parent_widget, "Save capture", default_name, "PNG Images (*.png)")

    if file_path:
        if isinstance(image_data, QPixmap) and not image_data.isNull():
            image_data.save(file_path)
            return True
        elif isinstance(image_data, np.ndarray):
            cv2.imwrite(file_path, image_data)
            return True
    return False

def set_scaled_pixmap(label, pixmap):
    if pixmap is not None and not pixmap.isNull():
        label.setPixmap(
            pixmap.scaled(
                label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
        ))