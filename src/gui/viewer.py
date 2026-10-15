if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import cv2
import sys
import argparse
import numpy as np
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QImage, QFont
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QGroupBox, QSlider, QButtonGroup, QTabWidget, QComboBox, QPushButton, QCheckBox

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.core_utils import solve_poisson_equation
from shared.image_utils import convert_to_qpixmap, normalize_to_uint8

SUPPRESSION_ANIM_STEPS = 60  # ticks per one-way sweep
SUPPRESSION_ANIM_EASING = 9  # higher = more dwell at 0/100
class AutoRotatingGLViewWidget(gl.GLViewWidget):
    CLICK_SLOP_PX = 5      # press-release movement tolerance for a click
    ROTATION_STEP_DEG = 1.0

    def __init__(self):
        super().__init__()
        self.auto_rotate_timer = QTimer(self)
        self.auto_rotate_timer.setInterval(30)
        self.auto_rotate_timer.timeout.connect(self.auto_rotate_step)
        self._pending_click = False
        self._press_pos = None

    def mousePressEvent(self, event):
        self._pending_click = True
        self._press_pos = event.position() if hasattr(event, 'position') else event.pos()
        self.stop_auto_rotation()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pending_click:
            current_pos = event.position() if hasattr(event, 'position') else event.pos()
            moved = (current_pos - self._press_pos).manhattanLength()
            if moved > self.CLICK_SLOP_PX:
                self._pending_click = False
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._pending_click:
            self._pending_click = False
            self.auto_rotate_timer.start()

    def wheelEvent(self, event):
        self.stop_auto_rotation()
        super().wheelEvent(event)

    def stop_auto_rotation(self):
        self.auto_rotate_timer.stop()

    def auto_rotate_step(self):
        params = self.opts
        self.setCameraPosition(
            azimuth=(params['azimuth'] + self.ROTATION_STEP_DEG) % 360.0,
            elevation=params['elevation'],
            distance=params['distance'])

from shared.gui_components import ViewerWindowWithFileTree, save_image_prompt, set_scaled_pixmap
from tools.utils import load_preinferred_latent_data, inference_from_latents

class InferredComponentsViewerApp(ViewerWindowWithFileTree):
    def __init__(self):
        super().__init__("Trinity-Inverse Inference Visualizer")

        self.det_p_vectors = None
        self.det_alpha = None
        self.se_beta = None
        self.det_se_w = None
        self.det_se_b = None

        self.active_detector_index = 0
        self.active_sample_directory = None
        self.current_display_pixmap = None
        self.current_component_pixmap = None

        self.render_height_map = None
        self.render_nx = None
        self.render_ny = None
        self.render_nz = None
        self.render_comp = None
        self.render_r_low_etd = None
        self.render_r_low_t1 = None
        self.render_r_low_t2 = None
        self.render_r_high = None
        self.cached_colors = None

        self.latent_comp = None
        self.latent_nx = None
        self.latent_ny = None
        self.latent_nz = None
        self.latent_r_low_etd = None
        self.latent_r_low_t1 = None
        self.latent_r_low_t2 = None
        self.latent_r_high = None
        self.latent_gain = None

        self.suppression_animation_timer = QTimer(self)
        self.suppression_animation_timer.setInterval(25)
        self.suppression_animation_timer.timeout.connect(self.animate_suppression_step)
        self.suppression_animation_direction = 1
        self.suppression_animation_step_index = 0

        if os.path.exists(args.latents_root_dir):
            self.root_directory_path = os.path.abspath(args.latents_root_dir)
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
            cand = os.path.join(repo_root, args.latents_root_dir)
            self.root_directory_path = cand if os.path.exists(cand) else os.path.join(os.path.dirname(os.path.realpath(__file__)), args.latents_root_dir)

        self.initialize_user_interface()
        self.build_sample_directory_list()

    def initialize_user_interface(self):
        detector_selection_group_box = QGroupBox("Detector")
        detector_selection_layout = QHBoxLayout(detector_selection_group_box)
        self.detector_button_group = QButtonGroup(self)
        available_detectors = [("ETD", 0), ("T1", 1), ("T2", 2)]
        for mode_name, mode_identifier in available_detectors:
            radio_button = QRadioButton(mode_name)
            if mode_identifier == 0: radio_button.setChecked(True)
            self.detector_button_group.addButton(radio_button, mode_identifier)
            detector_selection_layout.addWidget(radio_button)
        self.detector_button_group.idToggled.connect(self.handle_detector_changed)
        self.left_panel_layout.addWidget(detector_selection_group_box)

        self.main_tabs = QTabWidget()
        self.left_panel_layout.addWidget(self.main_tabs, stretch=1)

        self.main_result_image_label = QLabel()
        self.main_result_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_result_image_label.setMinimumSize(600, 600)
        self.main_tabs.addTab(self.main_result_image_label, "Inference")
        self.main_tabs.currentChanged.connect(self.handle_tab_changed)

        tab2_widget = QWidget()
        tab2_layout = QVBoxLayout(tab2_widget)

        self.component_selector_3d = QComboBox()
        self.component_selector_3d.addItems(["Compositional contrast (c)", "Surface normals (n)", "Low-frequency residuals (r_low)", "High-frequency residuals (r_high)"])
        self.component_selector_3d.currentIndexChanged.connect(self.sync_component_selectors)
        tab2_layout.addWidget(self.component_selector_3d)

        tab2_view_layout = QHBoxLayout()
        self.view_3d = AutoRotatingGLViewWidget()
        self.view_3d.setCameraPosition(distance=800, elevation=45, azimuth=45)
        tab2_view_layout.addWidget(self.view_3d, stretch=1)

        self.z_scale_slider = QSlider(Qt.Orientation.Vertical)
        self.z_scale_slider.setRange(1, 1000)
        self.z_scale_slider.setValue(100)
        self.z_scale_slider.valueChanged.connect(self.update_3d_z_scale)
        tab2_view_layout.addWidget(self.z_scale_slider)

        tab2_layout.addLayout(tab2_view_layout, stretch=1)
        self.surface_3d = None
        self.main_tabs.addTab(tab2_widget, "Height map reconstruction")

        tab3_widget = QWidget()
        tab3_layout = QVBoxLayout(tab3_widget)

        self.component_selector_2d = QComboBox()
        self.component_selector_2d.addItems(["Compositional contrast (c)", "Surface normals (n)", "Low-frequency residuals (r_low)", "High-frequency residuals (r_high)"])
        self.component_selector_2d.currentIndexChanged.connect(self.sync_component_selectors)
        tab3_layout.addWidget(self.component_selector_2d)

        self.component_image_label = QLabel()
        self.component_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.component_image_label.setMinimumSize(600, 600)
        tab3_layout.addWidget(self.component_image_label, stretch=1)

        self.main_tabs.addTab(tab3_widget, "Inferred components")

        bottom_controls_container = QWidget()
        bottom_controls_layout = QVBoxLayout(bottom_controls_container)
        self.left_panel_layout.addWidget(bottom_controls_container)

        adjustments_group_box = QGroupBox("Inference adjustments")
        adjustments_layout = QVBoxLayout(adjustments_group_box)

        orientation_layout = QHBoxLayout()
        orientation_layout.addWidget(QLabel("Detector rotation (-180° to 180°):"))
        self.orientation_slider = QSlider(Qt.Orientation.Horizontal)
        self.orientation_slider.setRange(-180, 180)
        self.orientation_slider.setValue(0)
        self.orientation_slider.valueChanged.connect(self.execute_inference_pipeline)
        orientation_layout.addWidget(self.orientation_slider)
        adjustments_layout.addLayout(orientation_layout)

        noise_suppression_layout = QHBoxLayout()
        noise_suppression_layout.addWidget(QLabel("Noise suppression:"))
        self.noise_suppression_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_suppression_slider.setRange(0, 100)
        self.noise_suppression_slider.setValue(0)
        self.noise_suppression_slider.valueChanged.connect(self.execute_inference_pipeline)
        noise_suppression_layout.addWidget(self.noise_suppression_slider)
        self.animate_suppression_checkbox = QCheckBox("Animate")
        self.animate_suppression_checkbox.toggled.connect(self.toggle_suppression_animation)
        noise_suppression_layout.addWidget(self.animate_suppression_checkbox)
        adjustments_layout.addLayout(noise_suppression_layout)

        enhancement_layout = QHBoxLayout()
        enhancement_layout.addWidget(QLabel("Enhancement:"))
        self.enhancement_slider = QSlider(Qt.Orientation.Horizontal)
        self.enhancement_slider.setRange(0, 100)
        self.enhancement_slider.setValue(0)
        self.enhancement_slider.valueChanged.connect(self.execute_inference_pipeline)
        enhancement_layout.addWidget(self.enhancement_slider)
        adjustments_layout.addLayout(enhancement_layout)

        bottom_controls_layout.addWidget(adjustments_group_box)

        directory_label = QLabel("Triplet selection (pre-inferred)")
        directory_label.setFont(QFont("Arial", 12))
        self.right_panel_layout.insertWidget(0, directory_label)

        self.capture_button = QPushButton("Capture")
        self.capture_button.clicked.connect(self.capture_viewport)
        self.right_panel_layout.addWidget(self.capture_button)

    def sync_component_selectors(self, index):
        self.component_selector_2d.blockSignals(True)
        self.component_selector_3d.blockSignals(True)
        self.component_selector_2d.setCurrentIndex(index)
        self.component_selector_3d.setCurrentIndex(index)
        self.component_selector_2d.blockSignals(False)
        self.component_selector_3d.blockSignals(False)
        self.update_3d_colors()
        self.update_component_display()

    def build_sample_directory_list(self):
        valid_sample_directories = []
        for item in sorted(os.listdir(self.root_directory_path)):
            full_path = os.path.join(self.root_directory_path, item)
            if os.path.isdir(full_path):
                expected_npz = os.path.join(full_path, "latents.npz")
                expected_json = os.path.join(full_path, "params.json")
                if os.path.exists(expected_npz) and os.path.exists(expected_json):
                    valid_sample_directories.append(full_path)

        self.directory_button_group = QButtonGroup(self)
        for index, directory_path in enumerate(valid_sample_directories):
            display_name = os.path.basename(directory_path)
            radio_button = QRadioButton(display_name)
            self.directory_button_group.addButton(radio_button, index)
            self.file_explorer_scroll_layout.addWidget(radio_button)
            radio_button.toggled.connect(lambda checked, path=directory_path: self.load_sample_data(path) if checked else None)
            if index == 0:
                radio_button.setChecked(True)

    def handle_detector_changed(self, identifier, checked):
        if checked:
            self.active_detector_index = identifier
            self.execute_inference_pipeline()
            self.update_3d_colors()
            self.update_component_display()

    def update_3d_colors(self):
        if self.render_height_map is None: return

        width, height = self.render_height_map.T.shape
        self.cached_colors = np.empty((width, height, 4), dtype=np.float32)
        index = self.component_selector_3d.currentIndex()

        if index == 1:
            self.cached_colors[..., 0] = (self.render_nx.T + 1.0) * 0.5
            self.cached_colors[..., 1] = (self.render_ny.T + 1.0) * 0.5
            self.cached_colors[..., 2] = np.clip(self.render_nz.T, 0.0, 1.0)
            self.cached_colors[..., 3] = 1.0
        elif index == 0:
            data = self.render_comp
            normalized_data_8u = normalize_to_uint8(data)
            colored_data = cv2.applyColorMap(normalized_data_8u, cv2.COLORMAP_INFERNO)
            colored_rgb = cv2.cvtColor(colored_data, cv2.COLOR_BGR2RGB)
            self.cached_colors[..., 0] = colored_rgb[..., 0].T / 255.0
            self.cached_colors[..., 1] = colored_rgb[..., 1].T / 255.0
            self.cached_colors[..., 2] = colored_rgb[..., 2].T / 255.0
            self.cached_colors[..., 3] = 1.0
        else:
            if index == 2:
                if self.active_detector_index == 0: data = self.render_r_low_etd
                elif self.active_detector_index == 1: data = self.render_r_low_t1
                else: data = self.render_r_low_t2
            elif index == 3:
                data = self.render_r_high[self.active_detector_index].squeeze()
            else: return

            minimum_value, maximum_value = data.min(), data.max()
            normalized_data = (data - minimum_value) / (maximum_value - minimum_value + 1e-12)

            self.cached_colors[..., 0] = normalized_data.T
            self.cached_colors[..., 1] = normalized_data.T
            self.cached_colors[..., 2] = normalized_data.T
            self.cached_colors[..., 3] = 1.0

        self.cached_colors = np.flip(self.cached_colors, axis=0)

        if self.surface_3d is not None:
            z_multiplier = self.z_scale_slider.value() / 100.0
            z_data = np.flip(self.render_height_map.T * z_multiplier, axis=0)
            self.surface_3d.setData(z=z_data, colors=self.cached_colors)

    def update_3d_z_scale(self):
        if self.render_height_map is None or self.cached_colors is None:
            return

        z_multiplier = self.z_scale_slider.value() / 100.0
        z_data = self.render_height_map.T * z_multiplier
        z_data = np.flip(z_data, axis=0)

        if self.surface_3d is None:
            width, height = z_data.shape
            x_coordinates = np.linspace(-width/2, width/2, width)
            y_coordinates = np.linspace(-height/2, height/2, height)

            self.surface_3d = gl.GLSurfacePlotItem(
                x=x_coordinates,
                y=y_coordinates,
                z=z_data,
                colors=self.cached_colors,
                computeNormals=True,
                smooth=True,
                shader='shaded',
            )
            self.view_3d.addItem(self.surface_3d)
        else:
            self.surface_3d.setData(z=z_data)

    def load_sample_data(self, directory_path):
        self.active_sample_directory = directory_path

        self.orientation_slider.blockSignals(True)
        self.noise_suppression_slider.blockSignals(True)
        self.enhancement_slider.blockSignals(True)
        self.orientation_slider.setValue(0)
        self.noise_suppression_slider.setValue(0)
        self.enhancement_slider.setValue(0)
        self.orientation_slider.blockSignals(False)
        self.noise_suppression_slider.blockSignals(False)
        self.enhancement_slider.blockSignals(False)

        try:
            (
                self.latent_comp, (self.latent_nx, self.latent_ny, self.latent_nz),
                (self.latent_r_low_etd, self.latent_r_low_t1, self.latent_r_low_t2), self.latent_r_high,
                self.det_alpha, self.det_se_w, self.det_se_b, self.se_beta, self.det_p_vectors, self.latent_gain
            ) = load_preinferred_latent_data(directory_path)
           
            p_component = -self.latent_nx / self.latent_nz
            q_component = -self.latent_ny / self.latent_nz
            raw_height_map = solve_poisson_equation(p_component, q_component)

            decimation_factor = 2
            self.render_height_map = raw_height_map[::decimation_factor, ::decimation_factor]
            self.render_nx = self.latent_nx[::decimation_factor, ::decimation_factor]
            self.render_ny = self.latent_ny[::decimation_factor, ::decimation_factor]
            self.render_nz = self.latent_nz[::decimation_factor, ::decimation_factor]
            self.render_comp = self.latent_comp[::decimation_factor, ::decimation_factor]
            self.render_r_low_etd = self.latent_r_low_etd[::decimation_factor, ::decimation_factor]
            self.render_r_low_t1 = self.latent_r_low_t1[::decimation_factor, ::decimation_factor]
            self.render_r_low_t2 = self.latent_r_low_t2[::decimation_factor, ::decimation_factor]
            self.render_r_high = self.latent_r_high[:, ::decimation_factor, ::decimation_factor]

            if self.surface_3d is not None:
                self.view_3d.removeItem(self.surface_3d)
                self.surface_3d = None

            self.update_3d_colors()
            self.update_3d_z_scale()

            self.execute_inference_pipeline()
            self.update_component_display()

        except Exception as error:
            print(f"Error loading {directory_path}: {error}")

    def toggle_suppression_animation(self, checked):
        if checked and self.main_tabs.currentIndex() == 0:
            self.suppression_animation_step_index = 0
            self.suppression_animation_direction = 1
            self.suppression_animation_timer.start()
        else:
            self.suppression_animation_timer.stop()

    def handle_tab_changed(self, tab_index):
        if tab_index == 0 and self.animate_suppression_checkbox.isChecked():
            if not self.suppression_animation_timer.isActive():
                self.suppression_animation_step_index = 0
                self.suppression_animation_direction = 1
                self.suppression_animation_timer.start()
        else:
            self.suppression_animation_timer.stop()

    def animate_suppression_step(self):
        slider = self.noise_suppression_slider
        value = slider.value()

        step_index = self.suppression_animation_step_index
        if self.suppression_animation_direction == 1:
            eased = int(round(100 * (step_index / SUPPRESSION_ANIM_STEPS) ** SUPPRESSION_ANIM_EASING))
        else:
            eased = int(round(100 * (1 - (step_index / SUPPRESSION_ANIM_STEPS) ** SUPPRESSION_ANIM_EASING)))
        eased = int(np.clip(eased, 0, 100))

        self.suppression_animation_step_index += 1
        if step_index >= SUPPRESSION_ANIM_STEPS:
            self.suppression_animation_direction *= -1
            self.suppression_animation_step_index = 0
            eased = 100 if self.suppression_animation_direction == -1 else 0

        slider.setValue(eased)

    def execute_inference_pipeline(self):
        if self.latent_comp is None or self.det_p_vectors is None: return

        detector_index = self.active_detector_index
        det_p_theta_rad = np.deg2rad(self.orientation_slider.value())
        r_high_factor = 1.0 - (self.noise_suppression_slider.value() / 100.0)
        r_low_factor = 1.0 - (self.enhancement_slider.value() / 100.0)

        inferred_image = inference_from_latents(
            detector_index, r_high_factor, r_low_factor, det_p_theta_rad,
            self.latent_comp, (self.latent_nx, self.latent_ny, self.latent_nz), (self.latent_r_low_etd, self.latent_r_low_t1, self.latent_r_low_t2), self.latent_r_high,
            self.det_alpha, self.det_se_w, self.det_se_b, self.se_beta, self.det_p_vectors, self.latent_gain
        )
        normalized_display_image = normalize_to_uint8(inferred_image)

        self.current_display_pixmap = convert_to_qpixmap(normalized_display_image)
        set_scaled_pixmap(self.main_result_image_label, self.current_display_pixmap)

    def update_component_display(self):
        if self.latent_comp is None: return

        index = self.component_selector_2d.currentIndex()

        if index == 1:
            height_image, width_image = self.latent_nx.shape
            rgb_normals = np.empty((height_image, width_image, 3), dtype=np.float32)
            rgb_normals[..., 0] = (self.latent_nx + 1.0) * 0.5
            rgb_normals[..., 1] = (self.latent_ny + 1.0) * 0.5
            rgb_normals[..., 2] = np.clip(self.latent_nz, 0.0, 1.0)
            normalized_data = (rgb_normals * 255).astype(np.uint8)
        elif index == 0:
            data = self.latent_comp
            normalized_data_8u = normalize_to_uint8(data)
            normalized_data_color = cv2.applyColorMap(normalized_data_8u, cv2.COLORMAP_INFERNO)
            normalized_data = cv2.cvtColor(normalized_data_color, cv2.COLOR_BGR2RGB)
        else:
            if index == 2:
                if self.active_detector_index == 0: data = self.latent_r_low_etd
                elif self.active_detector_index == 1: data = self.latent_r_low_t1
                else: data = self.latent_r_low_t2
            elif index == 3:
                data = self.latent_r_high[self.active_detector_index].squeeze()
            else: return

            normalized_data = normalize_to_uint8(data)

        self.current_component_pixmap = convert_to_qpixmap(normalized_data)
        set_scaled_pixmap(self.component_image_label, self.current_component_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        set_scaled_pixmap(self.main_result_image_label, self.current_display_pixmap)
        if hasattr(self, 'current_component_pixmap') and self.current_component_pixmap is not None:
            set_scaled_pixmap(self.component_image_label, self.current_component_pixmap)

    def capture_viewport(self):
        if self.latent_comp is None: return

        current_tab_index = self.main_tabs.currentIndex()

        if current_tab_index == 0:
            if self.current_display_pixmap:
                save_image_prompt(self, "inference", self.current_display_pixmap)

        elif current_tab_index == 1:
            qt_image = self.view_3d.grabFramebuffer()
            qt_image = qt_image.convertToFormat(QImage.Format.Format_RGBA8888)
            width, height = qt_image.width(), qt_image.height()

            pointer = qt_image.bits()
            pointer.setsize(qt_image.sizeInBytes())
            array = np.array(pointer, copy=True).reshape((height, width, 4))

            mask = np.any(array[:, :, :3] != [0, 0, 0], axis=-1)
            array[~mask] = [0, 0, 0, 0]

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if np.any(rows) and np.any(cols):
                row_min, row_max = np.where(rows)[0][[0, -1]]
                col_min, col_max = np.where(cols)[0][[0, -1]]
                array_cropped = array[row_min:row_max+1, col_min:col_max+1]
                final_array = cv2.cvtColor(array_cropped, cv2.COLOR_RGBA2BGRA)
            else:
                final_array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)

            save_image_prompt(self, "height_map", final_array)

        elif current_tab_index == 2:
            if self.current_component_pixmap:
                save_image_prompt(self, "component", self.current_component_pixmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents_root_dir', type=str, default="inferred_latents",
                        help="Path to directory containing inferred latents (with params.json and latents.npz subfolders)")
    args = parser.parse_args()

    application_instance = QApplication(sys.argv)
    application_instance.setStyle("Fusion")
    main_window_instance = InferredComponentsViewerApp()
    main_window_instance.show()
    sys.exit(application_instance.exec())
