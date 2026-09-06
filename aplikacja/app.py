import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QPushButton, QFileDialog, QLabel, QSpinBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

from src.models_brush import Generator

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODEL_PATH = "./checkpoints_brush_advanced/brush_adv_epoch_59.pth"
IMG_SIZE = 256

transform_img = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_mask = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])


def load_model():
    netG = Generator().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        return netG

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if 'ema_netG_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['ema_netG_state_dict'])
    elif 'netG_state_dict' in checkpoint:
        netG.load_state_dict(checkpoint['netG_state_dict'])
    else:
        netG.load_state_dict(checkpoint)

    netG.eval()
    return netG


class PaintCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(512, 512)
        self.setStyleSheet("border: 2px dashed #666; background-color: #1e1e1e;")

        self.image = QPixmap(512, 512)
        self.image.fill(Qt.black)

        self.mask_pixmap = QPixmap(512, 512)
        self.mask_pixmap.fill(Qt.transparent)

        self.original_pil = None
        self.drawing = False
        self.brush_size = 25
        self.last_point = QPoint()

    def set_base_image(self, pil_img):
        self.original_pil = pil_img.resize((512, 512))

        qimg = QImage(self.original_pil.tobytes(), 512, 512, QImage.Format_RGB888)
        self.image = QPixmap.fromImage(qimg)

        self.clear_mask()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.original_pil:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.mask_pixmap)
            pen = QPen(QColor(255, 0, 0, 180), self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.image)
        painter.drawPixmap(0, 0, self.mask_pixmap)

    def get_binary_mask_pil(self):
        if not self.original_pil:
            return None

        qimg = self.mask_pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        ptr = qimg.bits()
        ptr.setsize(512 * 512 * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((512, 512, 4))
        alpha = arr[:, :, 3]
        mask_binary = (alpha > 0).astype(np.uint8) * 255
        return Image.fromarray(mask_binary).convert("L")

    def clear_mask(self):
        self.mask_pixmap.fill(Qt.transparent)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kontrolowane wymazywanie danych MRI")
        self.setGeometry(100, 100, 1100, 660)

        self.model = load_model()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        left_box = QVBoxLayout()

        controls = QHBoxLayout()
        self.btn_load = QPushButton("Wczytaj skan MRI")
        self.btn_load.setStyleSheet("padding: 6px; font-weight: bold;")
        self.btn_load.clicked.connect(self.load_image)
        controls.addWidget(self.btn_load)

        controls.addWidget(QLabel("Rozmiar pędzla:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(5, 80)
        self.spin_brush.setValue(25)
        self.spin_brush.valueChanged.connect(self.change_brush_size)
        controls.addWidget(self.spin_brush)

        self.btn_clear = QPushButton("Wyczyść pędzel")
        self.btn_clear.clicked.connect(self.clear_mask)
        controls.addWidget(self.btn_clear)

        left_box.addLayout(controls)

        self.canvas = PaintCanvas()
        left_box.addWidget(self.canvas)

        self.btn_process = QPushButton("WYMAŻ I UZUPEŁNIJ TKANKĘ")
        self.btn_process.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd; 
                color: white; 
                font-weight: bold; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)
        self.btn_process.clicked.connect(self.process_image)
        left_box.addWidget(self.btn_process)

        layout.addLayout(left_box)

        right_box = QVBoxLayout()
        lbl_result_title = QLabel("<b>Wynik rekonstrukcji:</b>")
        lbl_result_title.setStyleSheet("font-size: 14px;")
        right_box.addWidget(lbl_result_title)

        self.result_label = QLabel()
        self.result_label.setFixedSize(512, 512)
        self.result_label.setStyleSheet("border: 2px solid #444; background-color: #1e1e1e;")
        self.result_label.setAlignment(Qt.AlignCenter)
        right_box.addWidget(self.result_label)

        layout.addLayout(right_box)

    def change_brush_size(self, val):
        self.canvas.brush_size = val

    def clear_mask(self):
        self.canvas.clear_mask()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz skan MRI", "", "Pliki obrazów (*.png *.jpg *.jpeg)"
        )
        if file_path:
            pil_img = Image.open(file_path).convert("RGB")
            self.canvas.set_base_image(pil_img)

    def process_image(self):
        if self.canvas.original_pil is None:
            return

        mask_pil = self.canvas.get_binary_mask_pil()
        if mask_pil is None:
            return

        real_img_t = transform_img(self.canvas.original_pil).unsqueeze(0).to(DEVICE)
        mask_t = transform_mask(mask_pil).unsqueeze(0).to(DEVICE)

        if mask_t.shape[1] > 1:
            mask_t = mask_t[:, 0:1, :, :]

        inv_mask_t = 1.0 - mask_t

        with torch.no_grad():
            fake_img_t = self.model(real_img_t, mask_t)
            erased_result_t = real_img_t * inv_mask_t + fake_img_t * mask_t

        result_norm = (erased_result_t.squeeze(0).cpu() + 1) / 2.0
        result_norm = torch.clamp(result_norm, 0.0, 1.0)

        to_pil = T.ToPILImage()
        result_pil = to_pil(result_norm).resize((512, 512))

        qimg = QImage(result_pil.tobytes(), 512, 512, QImage.Format_RGB888)
        self.result_label.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())