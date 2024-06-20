import os
import cv2
import xml.etree.ElementTree as ET
from glob import glob
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, \
    QProgressBar
import sys
import numpy as np

def draw_boxes(image, labels):
    for label in labels:
        x1, y1, x2, y2 = label
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image


def load_labels(label_path):
    labels = []
    tree = ET.parse(label_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        labels.append([x1, y1, x2, y2])

    return labels


def draw_boxes_on_images(input_folder, progress_callback):
    image_folder = os.path.join(input_folder, 'images')
    label_folder = os.path.join(input_folder, 'labels')
    output_folder = os.path.join(input_folder, 'output')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob(os.path.join(image_folder, '*.jpg')) + \
                  glob(os.path.join(image_folder, '*.jpeg')) + \
                  glob(os.path.join(image_folder, '*.png')) + \
                  glob(os.path.join(image_folder, '*.bmp')) + \
                  glob(os.path.join(image_folder, '*.tif')) + \
                  glob(os.path.join(image_folder, '*.tiff'))

    total_files = len(image_files)
    for index, image_path in enumerate(image_files):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_folder, base_name + '.xml')

        if os.path.exists(label_path):
            # 使用 cv2.imdecode 读取含中文路径的图像
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            labels = load_labels(label_path)
            image_with_boxes = draw_boxes(image, labels)

            # 使用 cv2.imencode 处理含中文路径的图像保存
            ext = os.path.splitext(image_path)[1]
            success, encoded_image = cv2.imencode(ext, image_with_boxes)
            if success:
                output_image_path = os.path.join(output_folder, base_name + '_with_boxes.jpg')
                with open(output_image_path, 'wb') as f:
                    f.write(encoded_image)
        else:
            print(f"Warning: Label file not found for image {image_path}")

        progress_callback(index + 1, total_files)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("选择文件夹")
        self.setGeometry(100, 100, 300, 200)

        self.label = QLabel("请选择包含图像和 XML 文件的文件夹", self)
        self.label.setGeometry(20, 20, 260, 40)

        self.button = QPushButton("选择文件夹", self)
        self.button.setGeometry(90, 100, 120, 40)
        self.button.clicked.connect(self.select_folder)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(20, 150, 260, 30)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹', '')
        if folder_path:
            self.progress_bar.setValue(0)
            draw_boxes_on_images(folder_path, self.update_progress)
            self.label.setText("处理完成！")

    def update_progress(self, current, total):
        self.progress_bar.setValue(int((current / total) * 100))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
