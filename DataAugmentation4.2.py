import cv2
import sys
import os
import random
import numpy as np
from glob import glob
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, \
    QGridLayout, QLineEdit, QComboBox, QListWidget, QAbstractItemView, QListWidgetItem, QMenuBar, QAction, QMessageBox, \
    QSlider, QMainWindow, QProgressBar
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import xml.etree.ElementTree as ET
import re


class DataAugmentation:
    def __init__(self):
        pass

    def scale(self, image, fx, fy):
        return cv2.resize(image, (0, 0), fx=fx, fy=fy)

    def rotate(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH)), M

    def flip(self, image, flip_code):
        return cv2.flip(image, flip_code)

    def adjust_brightness(self, image, value):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def histogram_equalization(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(img_y_cr_cb)
            cv2.equalizeHist(channels[0], channels[0])
            img_y_cr_cb = cv2.merge(channels)
            result_image = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)
        else:
            result_image = cv2.equalizeHist(image)

        return result_image

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def adjust_saturation(self, image, saturation_factor=1.0):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def translate_image(self, image, tx, ty):
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, M, (cols, rows))

        return translated_image

    def save_image(self, image, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        cv2.imwrite(path, image)

    def load_image_and_labels(self, image_path, label_path):
        image = cv2.imread(image_path)
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

        return image, labels

    def save_labels(self, label_path, labels, original_label_path, new_image):
        tree = ET.parse(original_label_path)
        root = tree.getroot()

        # 更新size节点中的width和height
        size_node = root.find('size')
        if size_node is not None:
            width_node = size_node.find('width')
            height_node = size_node.find('height')
            if width_node is not None:
                width_node.text = str(new_image.shape[1])
            if height_node is not None:
                height_node.text = str(new_image.shape[0])

        # 清除现有的object标签
        for obj in root.findall('object'):
            root.remove(obj)

        for label in labels:
            x1, y1, x2, y2 = label
            obj = ET.Element('object')
            original_obj = next((o for o in ET.parse(original_label_path).getroot().findall('object')
                                 if o.find('name').text), None)
            if original_obj is not None:
                for child in list(original_obj):
                    obj.append(child)

            bndbox = obj.find('bndbox')
            bndbox.find('xmin').text = str(int(x1))
            bndbox.find('ymin').text = str(int(y1))
            bndbox.find('xmax').text = str(int(x2))
            bndbox.find('ymax').text = str(int(y2))

            root.append(obj)

        tree.write(label_path)

    def process_image(self, image_path, label_path, fx, fy, angle_str, flip_code, brightness, he_flag, gamma,
                      saturation, tx_str, ty_str):
        if label_path is not None:
            image, labels = self.load_image_and_labels(image_path, label_path)
        else:
            image = cv2.imread(image_path)
            labels = []  # 如果没有标注文件，则标签为空列表
        if image is not None:
            try:
                if '~' in angle_str:
                    match = re.match(r'(-?\d+(\.\d+)?)~(-?\d+(\.\d+)?)', angle_str)
                    if match:
                        angle_min, angle_max = map(float, match.groups()[::2])
                        angle = random.uniform(angle_min, angle_max)
                    else:
                        angle = float(angle_str or 0)  # 默认值0
                else:
                    angle = float(angle_str or 0)  # 默认值0
            except ValueError:
                angle = 0  # 当无法转换为浮点数时使用默认值0

            rotated_image, M = self.rotate(image, angle)
            if flip_code != 11:
                flipped_image = self.flip(rotated_image, flip_code)
            else:
                flipped_image = rotated_image
            bright_image = self.adjust_brightness(flipped_image, brightness)
            if he_flag == 1:
                he_image = self.histogram_equalization(bright_image)
            else:
                he_image = bright_image
            gamma_image = self.adjust_gamma(he_image, gamma)
            saturation_image = self.adjust_saturation(gamma_image, saturation)
            try:
                if '~' in tx_str:
                    match = re.match(r'(-?\d+(\.\d+)?)~(-?\d+(\.\d+)?)', tx_str)
                    if match:
                        tx_min, tx_max = map(float, match.groups()[::2])
                        tx = random.uniform(tx_min, tx_max)
                    else:
                        tx = float(tx_str or 0)  # 默认值0
                else:
                    tx = float(tx_str or 0)  # 默认值0
            except ValueError:
                tx = 0  # 当无法转换为浮点数时使用默认值0

            try:
                if '~' in ty_str:
                    match = re.match(r'(-?\d+(\.\d+)?)~(-?\d+(\.\d+)?)', ty_str)
                    if match:
                        ty_min, ty_max = map(float, match.groups()[::2])
                        ty = random.uniform(ty_min, ty_max)
                    else:
                        ty = float(ty_str or 0)  # 默认值0
                else:
                    ty = float(ty_str or 0)  # 默认值0
            except ValueError:
                ty = 0  # 当无法转换为浮点数时使用默认值0
            translate_image = self.translate_image(saturation_image, tx, ty)
            scaled_image = self.scale(translate_image, fx, fy)

            # 更新标签
            if labels:
                augmented_labels = self.update_labels(labels, scaled_image, M, flip_code, tx, ty, fx, fy, angle)
            else:
                augmented_labels = []

            return scaled_image, augmented_labels
        else:
            print(f"Error loading image {image_path}")
            return None, None

    def update_labels(self, labels, image, M, flip_code, tx, ty, fx, fy, angle):
        augmented_labels = []

        # Step 1: Apply affine transformation (rotation and translation)
        for label in labels:
            x1, y1, x2, y2 = label
            points = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            points = np.hstack([points, np.ones((4, 1))])
            new_points = M.dot(points.T).T
            x_coords = new_points[:, 0]
            y_coords = new_points[:, 1]
            new_x1, new_y1 = x_coords.min(), y_coords.min()
            new_x2, new_y2 = x_coords.max(), y_coords.max()
            augmented_labels.append([new_x1, new_y1, new_x2, new_y2])
            # print(new_x1, new_y1, new_x2, new_y2)

        # Step 2: Apply flipping
        h, w, _ = image.shape
        h=h/fy
        w=w/fx
        flip_labels = []
        for label in augmented_labels:
            x_min, y_min, x_max, y_max = label
            if flip_code == 0:
                flip_labels.append([x_max, h - y_min, x_min, h - y_max])
            elif flip_code == 1:
                flip_labels.append([w - x_max, y_min, w - x_min, y_max])
            elif flip_code == -1:
                flip_labels.append([w - x_min, h - y_max, w - x_max, h - y_min])
            elif flip_code == 11:
                flip_labels = augmented_labels

        # Step 3: Apply translation
        translated_labels = []
        for label in flip_labels:
            new_x1, new_y1, new_x2, new_y2 = label
            # print(new_x1, new_y1, new_x2, new_y2)
            translated_labels.append([new_x1 + tx, new_y1 + ty, new_x2 + tx, new_y2 + ty])
            # print(new_x1 + tx, new_y1 + ty, new_x2 + tx, new_y2 + ty)

        # Step 4: Apply scaling
        scaled_labels = []
        for label in translated_labels:
            x_min, y_min, x_max, y_max = label
            scaled_labels.append([x_min * fx, y_min * fy, x_max * fx, y_max * fy])
            # print(x_min * fx, y_min * fy, x_max * fx, y_max * fy)

        return scaled_labels


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('数据增强器4.2')
        self.setGeometry(100, 100, 1000, 800)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.menuBar = QMenuBar(self)
        fileMenu = self.menuBar.addMenu('文件')

        loadImageFolderAction = QAction('打开图片文件夹', self)
        loadImageFolderAction.triggered.connect(self.loadImageFolder)
        fileMenu.addAction(loadImageFolderAction)

        loadFolderAction = QAction('打开含标注文件的文件夹', self)
        loadFolderAction.triggered.connect(self.loadFolder)
        fileMenu.addAction(loadFolderAction)

        self.layout.setMenuBar(self.menuBar)

        self.scaleInputFx = QLineEdit()
        self.scaleInputFy = QLineEdit()
        self.rotateInputAngle = QLineEdit()
        self.flipInputCode = QComboBox()
        self.flipInputCode.addItem("不翻转", 11)
        self.flipInputCode.addItem("水平翻转", 1)
        self.flipInputCode.addItem("垂直翻转", 0)
        self.flipInputCode.addItem("水平垂直翻转", -1)
        self.brightnessInputValue = QLineEdit()
        self.heInputAngle = QLineEdit()
        self.scaleInputtx = QLineEdit()
        self.scaleInputty = QLineEdit()

        self.selectAllButton = QPushButton('全选')
        self.selectAllButton.clicked.connect(self.selectAllImages)

        self.previewButton = QPushButton('预览图片')
        self.previewButton.clicked.connect(self.previewImage)

        self.saveButton = QPushButton('保存图片')
        self.saveButton.clicked.connect(self.saveImages)

        self.rotateInputAngle = QLineEdit()

        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(-100)
        self.brightnessSlider.setMaximum(100)
        self.brightnessSlider.setValue(0)
        self.brightnessSlider.valueChanged.connect(self.updateBrightnessValue)
        self.brightnessValueLabel = QLabel("0")

        self.gammaSlider = QSlider(Qt.Horizontal)
        self.gammaSlider.setMinimum(0)
        self.gammaSlider.setMaximum(20)
        self.gammaSlider.setValue(10)
        self.gammaSlider.valueChanged.connect(self.updategammaValue)
        self.gammaValueLabel = QLabel("1")

        self.saturationSlider = QSlider(Qt.Horizontal)
        self.saturationSlider.setMinimum(0)
        self.saturationSlider.setMaximum(30)
        self.saturationSlider.setValue(10)
        self.saturationSlider.valueChanged.connect(self.updatesaturationValue)
        self.saturationValueLabel = QLabel("1")

        gridLayout = QGridLayout()
        gridLayout.addWidget(QLabel("缩放因子X:"), 0, 0)
        gridLayout.addWidget(self.scaleInputFx, 0, 1)
        gridLayout.addWidget(QLabel("缩放因子Y:"), 0, 3)
        gridLayout.addWidget(self.scaleInputFy, 0, 4)
        gridLayout.addWidget(QLabel("亮度调整值:"), 1, 0)
        gridLayout.addWidget(self.brightnessSlider, 1, 1)
        gridLayout.addWidget(self.brightnessValueLabel, 1, 2)
        gridLayout.addWidget(QLabel("旋转角度（或范围）:"), 1, 3)
        gridLayout.addWidget(self.rotateInputAngle, 1, 4)
        gridLayout.addWidget(QLabel("饱和度:"), 2, 0)
        gridLayout.addWidget(self.saturationSlider, 2, 1)
        gridLayout.addWidget(self.saturationValueLabel, 2, 2)
        gridLayout.addWidget(QLabel("翻转:"), 2, 3)
        gridLayout.addWidget(self.flipInputCode, 2, 4)
        gridLayout.addWidget(QLabel("伽马校正:"), 3, 0)
        gridLayout.addWidget(self.gammaSlider, 3, 1)
        gridLayout.addWidget(self.gammaValueLabel, 3, 2)
        gridLayout.addWidget(QLabel("直方图均衡化:"), 3, 3)
        gridLayout.addWidget(self.heInputAngle, 3, 4)
        gridLayout.addWidget(QLabel("x轴偏移量:"), 4, 0)
        gridLayout.addWidget(self.scaleInputtx, 4, 1)
        gridLayout.addWidget(QLabel("y轴偏移量:"), 4, 3)
        gridLayout.addWidget(self.scaleInputty, 4, 4)
        self.layout.addLayout(gridLayout)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.selectAllButton)
        buttonLayout.addWidget(self.previewButton)
        buttonLayout.addWidget(self.saveButton)
        self.layout.addLayout(buttonLayout)

        self.imageList = QListWidget()
        self.imageList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.imageList.setFixedWidth(300)
        self.imageList.itemClicked.connect(self.toggleImageSelection)
        self.imageList.itemClicked.connect(self.onImageItemClicked)

        self.imageDisplayLabel = QLabel()
        self.imageDisplayLabel.setFixedSize(500, 500)
        self.imageDisplayLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        listAndImageLayout = QHBoxLayout()
        listAndImageLayout.addWidget(self.imageList)

        imageListLayout = QVBoxLayout()
        imageListLayout.addWidget(self.imageList)

        listAndImageLayout.addLayout(imageListLayout)
        listAndImageLayout.addWidget(self.imageDisplayLabel)
        # 添加进度条
        self.progressBar = QProgressBar(self)
        self.layout.addWidget(self.progressBar)

        self.layout.addLayout(listAndImageLayout)
        self.setLayout(self.layout)

    def toggleImageSelection(self, item):
        current_state = item.checkState()
        new_state = Qt.Unchecked if current_state == Qt.Checked else Qt.Checked
        item.setCheckState(new_state)

    def onImageItemClicked(self, item):
        currentState = item.checkState()
        newState = Qt.Unchecked if currentState == Qt.Checked else Qt.Checked
        item.setCheckState(newState)
        image_name = item.text()
        image_path = os.path.join(self.folderPath, image_name)
        self.displayImage(image_path)

    def updateBrightnessValue(self, value):
        self.brightnessValueLabel.setText(str(value))

    def updategammaValue(self, value):
        gamma_value = value / 10.0
        self.gammaValueLabel.setText(f"{gamma_value:.1f}")

    def updatesaturationValue(self, value):
        saturation_value = value / 10.0
        self.saturationValueLabel.setText(f"{saturation_value:.1f}")

    def loadImageFolder(self):
        folderPath = QFileDialog.getExistingDirectory(self, '选择文件夹', '')
        if folderPath:
            self.folderPath = folderPath
            self.imageList.clear()

            wildcard_expressions = [
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp",
                "*.tif",
                "*.tiff"
            ]

            found_files = False
            for wildcard in wildcard_expressions:
                matching_files = glob(os.path.join(folderPath, wildcard))
                if matching_files:
                    found_files = True
                for file_path in matching_files:
                    file_name = os.path.basename(file_path)
                    item = QListWidgetItem(file_name)
                    item.setCheckState(Qt.Unchecked)
                    self.imageList.addItem(item)

            if not found_files:
                QMessageBox.warning(self, "文件夹为空", "选定的文件夹中没有找到符合条件的图片！")

    def loadFolder(self):
        folderPath = QFileDialog.getExistingDirectory(self, '选择文件夹', '')
        if folderPath:
            self.folderPath = folderPath
            self.imageList.clear()
            self.xmlFiles = []

            wildcard_expressions = [
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp",
                "*.tif",
                "*.tiff",
                "*.xml"
            ]
            found_files = False
            for wildcard in wildcard_expressions:
                matching_files = glob(os.path.join(folderPath, wildcard))
                if matching_files:
                    found_files = True
                for file_path in matching_files:
                    file_name = os.path.basename(file_path)
                    if file_path.endswith('.xml'):
                        self.xmlFiles.append(file_path)
                    else:
                        item = QListWidgetItem(file_name)
                        item.setCheckState(Qt.Unchecked)
                        self.imageList.addItem(item)
            if not found_files:
                QMessageBox.warning(self, "文件夹为空", "选定的文件夹中没有找到符合条件的图片或XML文件！")

    def selectAllImages(self):
        already_selected = all(item.checkState() == Qt.Checked for i in range(self.imageList.count())
                               for item in [self.imageList.item(i)])
        new_state = Qt.Unchecked if already_selected else Qt.Checked
        for i in range(self.imageList.count()):
            item = self.imageList.item(i)
            item.setCheckState(new_state)

    def showImageOnLabel(self, image, label):
        height, width, channel = image.shape
        bytesPerLine = channel * width
        qImage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImage)
        scaledPixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaledPixmap)

    def displayImage(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            self.showImageOnLabel(image, self.imageDisplayLabel)
        else:
            QMessageBox.warning(self, "加载错误", f"无法加载图片 {image_path}")

    def previewImage(self):
        selected_items = [item for item in self.imageList.findItems("", Qt.MatchContains) if
                          item.checkState() == Qt.Checked]
        if len(selected_items) > 1:
            QMessageBox.warning(self, "预览错误", "不能选择多张图片进行预览")
        elif len(selected_items) < 1:
            QMessageBox.warning(self, "预览错误", "请先选择一张图片")
        else:
            image_name = selected_items[0].text()
            image_path = os.path.join(self.folderPath, image_name)

            # 如果存在同名XML文件则使用，否则设为None
            label_path = os.path.join(self.folderPath, image_name.replace(image_name.split('.')[-1], 'xml'))
            if not os.path.exists(label_path):
                label_path = None

            fx = float(self.scaleInputFx.text() or 1)
            fy = float(self.scaleInputFy.text() or 1)
            angle_str = self.rotateInputAngle.text()
            flipCode = self.flipInputCode.currentData() if self.flipInputCode.currentText() else 11
            brightness = self.brightnessSlider.value()
            he_flag = int(self.heInputAngle.text() or 0)
            gamma = self.gammaSlider.value() / 10.0
            saturation = self.saturationSlider.value() / 10.0
            tx = self.scaleInputtx.text()
            ty = self.scaleInputty.text()
            augmenter = DataAugmentation()
            processed_image, augmented_labels = augmenter.process_image(
                image_path, label_path, fx, fy, angle_str, flipCode, brightness, he_flag, gamma, saturation, tx, ty
            )

            if processed_image is not None:
                self.showImageOnLabel(processed_image, self.imageDisplayLabel)
            else:
                QMessageBox.warning(self, "处理错误", f"无法处理图片 {image_path}")

    def saveImages(self):
        selected_items = [item for item in self.imageList.findItems("", Qt.MatchContains) if
                          item.checkState() == Qt.Checked]
        if selected_items:
            saveFolder = QFileDialog.getExistingDirectory(self, '保存图片到', '')
            if saveFolder:
                augmenter = DataAugmentation()

                fx = float(self.scaleInputFx.text() or 1)
                fy = float(self.scaleInputFy.text() or 1)
                angle_str = self.rotateInputAngle.text()
                flipCode = self.flipInputCode.currentData() if self.flipInputCode.currentText() else 11
                brightness = self.brightnessSlider.value()
                he_flag = int(self.heInputAngle.text() or 0)
                gamma = self.gammaSlider.value() / 10.0
                saturation = self.saturationSlider.value() / 10.0
                tx = self.scaleInputtx.text()
                ty = self.scaleInputty.text()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # output_folder = f"{saveFolder}/processed_{timestamp}"
                output_folder_images=f"{saveFolder}/processed/images"
                output_folder_labels=f"{saveFolder}/processed/labels"
                # if not os.path.exists(output_folder):
                #     os.makedirs(output_folder)
                if not os.path.exists(output_folder_images):
                    os.makedirs(output_folder_images)
                if not os.path.exists(output_folder_labels):
                    os.makedirs(output_folder_labels)


                total_items = len(selected_items)
                self.progressBar.setValue(0)

                for count, item in enumerate(selected_items):
                    image_name = os.path.splitext(item.text())[0]
                    extension = os.path.splitext(item.text())[1]
                    image_path = os.path.join(self.folderPath, item.text())

                    # 根据参数是否有变化来构造文件名
                    new_image_name = image_name
                    if angle_str:
                        new_image_name += '_rot'
                    if flipCode != 11:
                        new_image_name += '_flip'
                    if brightness != 0:
                        new_image_name += '_lt'
                    if he_flag != 0:
                        new_image_name += '_he'
                    if gamma != 1:
                        new_image_name += '_gamma'
                    if saturation != 1:
                        new_image_name += '_sat'
                    if tx or ty:
                        new_image_name += '_move'
                    if fx != 1 or fy != 1:
                        new_image_name += '_scale'

                    new_image_name += f"_{timestamp}.png"
                    new_label_name = new_image_name.replace('.png', '.xml')
                    # print(new_image_name)

                    # 如果存在同名XML文件则使用，否则设为None
                    label_path = os.path.join(self.folderPath, image_name + ".xml")
                    if not os.path.exists(label_path):
                        label_path = None

                    processed_image, augmented_labels = augmenter.process_image(
                        image_path, label_path, fx, fy, angle_str, flipCode, brightness, he_flag, gamma, saturation, tx,
                        ty
                    )
                    if processed_image is not None:
                        save_image_path = os.path.join(output_folder_images, new_image_name)
                        augmenter.save_image(processed_image, save_image_path)

                        if label_path is not None:
                            save_label_path = os.path.join(output_folder_labels, new_label_name)
                            augmenter.save_labels(save_label_path, augmented_labels, label_path, processed_image)

                    # 更新进度条
                    self.progressBar.setValue(int((count + 1) / total_items * 100))

        else:
            QMessageBox.warning(self, "保存错误", "没有选择要保存的图片")

def draw_boxes(image, labels):
    for label in labels:
        x1, y1, x2, y2 = label
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
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
            image = cv2.imread(image_path)
            labels = load_labels(label_path)
            image_with_boxes = draw_boxes(image, labels)
            output_image_path = os.path.join(output_folder, base_name + '_with_boxes.jpg')
            cv2.imwrite(output_image_path, image_with_boxes)
        else:
            print(f"Warning: Label file not found for image {image_path}")

        progress_callback(index + 1, total_files)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("画标注框（用于数据增强后检验）")
        self.setGeometry(100, 100, 350, 200)

        self.label = QLabel("请选择要检验的文件夹", self)
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

def run_app():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    run_app()
    sys.exit(app.exec_())