import os
import sys
import cv2
import glob
import numpy as np
from datetime import datetime
from PyQt5.QtGui import QColor
from DAP_fuction import DataAugmentation
from PySide6.QtCore import Qt
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QImage, QAction, QPixmap, QWheelEvent, QPainter, QPalette, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QFrame, QStyleFactory, QListWidgetItem, QGraphicsView, \
    QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QMenuBar, QLabel, QHBoxLayout, QVBoxLayout, QMessageBox

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale_factor = 1.0

        self.setMinimumSize(300, 200)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self.scale(1.1, 1.1)
            self.scale_factor *= 1.1
        else:
            self.scale(1 / 1.1, 1 / 1.1)
            self.scale_factor /= 1.1

class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()
        self.centralwidget = None
        self.centralVerticalLayout = None
        self.apply_system_theme()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("Data Augmentation Process Tool - Version 2.0 ")
        MainWindow.resize(1200, 800)
        self.MainWindow = MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 菜单栏
        self.menuBar = QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menuBar)
        self.startMenu = self.menuBar.addMenu('开始')
        self.openAction = QAction('打开', MainWindow)
        self.openAction.triggered.connect(self.open_directory_dialog)
        self.select_save_path_Action = QAction("选择保存路径", MainWindow)
        self.select_save_path_Action.triggered.connect(self.select_save_path_directory_dialog)
        self.exitAction = QAction('退出', MainWindow)
        self.exitAction.triggered.connect(MainWindow.close)
        self.startMenu.addAction(self.openAction)
        self.startMenu.addAction(self.select_save_path_Action)
        self.startMenu.addAction(self.exitAction)
        # 总体水平布局
        self.mainHorizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)

        # 左侧选项工具栏
        self.optionsFrame = QtWidgets.QFrame(self.centralwidget)
        self.optionsLayout = QtWidgets.QVBoxLayout(self.optionsFrame)
        self.add_option_controls()
        self.optionsFrame.setFixedWidth(250)
        self.mainHorizontalLayout.addWidget(self.optionsFrame)

        # 中间区域的垂直布局
        self.centralVerticalLayout = QtWidgets.QVBoxLayout()

        # 顶部操作栏
        self.actionToolBar = QtWidgets.QToolBar()
        # 创建并添加操作到工具栏
        self.start_action = QAction("开始生成")
        self.preview_action = QAction("预览")
        self.pause_action = QAction("暂停")
        self.stop_action = QAction("停止")
        self.help_action = QAction("帮助")
        self.about_action = QAction("关于")
        self.exit_action = QAction("退出")

        self.actionToolBar.addAction(self.start_action)
        self.actionToolBar.addAction(self.preview_action)
        self.actionToolBar.addAction(self.pause_action)
        self.actionToolBar.addAction(self.stop_action)
        self.actionToolBar.addAction(self.help_action)
        self.actionToolBar.addAction(self.about_action)
        self.actionToolBar.addAction(self.exit_action)
        # 连接操作到槽函数
        self.preview_action.triggered.connect(self.get_radio_items)
        self.start_action.triggered.connect(self.make_process_picture)
        # self.actionToolBar.addAction("开始生成")
        # self.actionToolBar.addAction("预览")
        # self.actionToolBar.addAction("暂停")
        # self.actionToolBar.addAction("停止")
        # self.actionToolBar.addAction("帮助")
        # self.actionToolBar.addAction("关于")
        # self.actionToolBar.addAction("退出")
        self.centralVerticalLayout.addWidget(self.actionToolBar)

        # 中间画板区域
        self.imagePreviewLayout = QHBoxLayout()
        self.leftVlayout = QVBoxLayout()
        self.rightVlayout = QVBoxLayout()
        self.imagePreviewLayout.addLayout(self.leftVlayout)
        self.imagePreviewLayout.addLayout(self.rightVlayout)

        self.original_graphics_view = ZoomableGraphicsView(self.MainWindow)
        self.original_scene = QGraphicsScene(self.MainWindow)
        self.original_graphics_view.setScene(self.original_scene)
        self.originalSizeLabel = QLabel("原图尺寸")
        #
        self.leftVlayout.addWidget(self.originalSizeLabel)
        self.leftVlayout.addWidget(self.original_graphics_view, stretch=1)

        self.enhanced_graphics_view = ZoomableGraphicsView(self.MainWindow)
        self.enhanced_scene = QGraphicsScene(self.MainWindow)
        self.enhanced_graphics_view.setScene(self.enhanced_scene)
        self.enhancedSizeLabel = QLabel("增强图片尺寸")
        #
        self.rightVlayout.addWidget(self.enhancedSizeLabel)
        # self.rightVlayout.addWidget(self.enhancedImageLabel, stretch=1)
        self.rightVlayout.addWidget(self.enhanced_graphics_view, stretch=1)

        self.centralVerticalLayout.addLayout(self.imagePreviewLayout, stretch=3)  # Increase stretch factor for larger display

        # 底部信息输出栏和进度条
        self.outputTextEdit = QtWidgets.QTextEdit()
        self.outputTextEdit.setMaximumHeight(100)
        self.centralVerticalLayout.addWidget(self.outputTextEdit, stretch=1)  # Lower stretch factor to make it smaller
        self.progressBar = QtWidgets.QProgressBar()
        self.centralVerticalLayout.addWidget(self.progressBar)

        self.mainHorizontalLayout.addLayout(self.centralVerticalLayout)
        # 右侧文件列表
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.setFixedWidth(250)
        self.mainHorizontalLayout.addWidget(self.fileListWidget)

        MainWindow.setCentralWidget(self.centralwidget)

        # 状态栏
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusBar)

        self.opt = None  # 用于判断是生成当前图片还是所有图片
        self.save_directory_path = None  # 保存路径

    def add_option_controls(self):
        # 添加各种控制元素
        labels = ["缩放因子X:", "缩放因子Y:", "翻转:", "旋转:", "亮度:", "伽马校正:", "饱和度:", "直方图均衡化:",
                  "X轴偏移量:", "Y轴偏移量:", "当前图片", "所有图片"]
        self.widgets = [
            QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), QtWidgets.QComboBox(),
            QtWidgets.QLineEdit(), QtWidgets.QSlider(Qt.Horizontal), QtWidgets.QSlider(Qt.Horizontal),
            QtWidgets.QSlider(Qt.Horizontal), QtWidgets.QComboBox(), QtWidgets.QLineEdit(),
            QtWidgets.QLineEdit(), QtWidgets.QRadioButton(), QtWidgets.QRadioButton()
        ]
        data = [
            None, None, [(11, "不翻转"), (1, "水平翻转"), (0, "垂直翻转"), (-1, "水平垂直翻转")],
            None, None, None, None, [(0, "否"), (1, "是")], None, None, None, None
        ]

        self.label77 = [
            QtWidgets.QLabel("", self.optionsFrame), QtWidgets.QLabel("", self.optionsFrame), QtWidgets.QLabel("",
                                                                                                              self.optionsFrame),
            QtWidgets.QLabel("", self.optionsFrame), QtWidgets.QLabel("", self.optionsFrame),
            QtWidgets.QLabel("", self.optionsFrame), QtWidgets.QLabel("", self.optionsFrame), QtWidgets.QLabel("",
                                                                                                              self.optionsFrame),
            QtWidgets.QLabel("", self.optionsFrame), QtWidgets.QLabel("", self.optionsFrame),
            QtWidgets.QLabel("", self.optionsFrame), QtWidgets.QLabel("", self.optionsFrame)
        ]
        labels33 = [None, None, None, None, "0", "1", "1", None, None, None, None, None]
        for i, label in enumerate(labels):
            layout = QtWidgets.QHBoxLayout()
            label_widget = QtWidgets.QLabel(label, self.optionsFrame)
            layout.addWidget(label_widget)
            if isinstance(self.widgets[i], QtWidgets.QComboBox):
                for item in data[i]:
                    self.widgets[i].addItem(item[1], item[0])
            elif isinstance(self.widgets[i], QtWidgets.QSlider):
                if i == 4:  # 亮度
                    self.widgets[i].setMinimum(-100)
                    self.widgets[i].setMaximum(100)
                    self.widgets[i].setValue(0)
                if i == 5:  # 伽马
                    self.widgets[i].setMinimum(0)
                    self.widgets[i].setMaximum(30)
                    self.widgets[i].setValue(10)
                if i == 6:  # 饱和
                    self.widgets[i].setMinimum(0)
                    self.widgets[i].setMaximum(30)
                    self.widgets[i].setValue(10)

            self.label77[i].setText(labels33[i])
            layout.addWidget(self.widgets[i])
            layout.addWidget(self.label77[i])
            self.optionsLayout.addLayout(layout)
        # 滚动条连接函数
        self.widgets[4].valueChanged.connect(self.updateBrightnessValue)
        self.widgets[5].valueChanged.connect(self.updategammaValue)
        self.widgets[6].valueChanged.connect(self.updatesaturationValue)
        self.widgets[10].toggled.connect(
            lambda checked, opt="a": self.a_all(checked, opt))
        self.widgets[11].toggled.connect(
            lambda checked, opt="all": self.a_all(checked, opt))

    def open_directory_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        directory_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory", "", options=options)
        if directory_path:
            self.directory_path77_all = directory_path
            self.load_images_from_directory(directory_path)

    def select_save_path_directory_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        directory_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory", "", options=options)
        if directory_path:
            self.save_directory_path = directory_path

    def load_images_from_directory(self, directory):
        self.fileListWidget.clear()
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        for filename in os.listdir(directory):
            if filename.lower().endswith(image_extensions):
                list_item = QListWidgetItem(self.fileListWidget)
                radio_button = QtWidgets.QRadioButton(filename)
                radio_button.toggled.connect(
                    lambda checked, path=os.path.join(directory, filename): self.display_image(checked, path))
                self.fileListWidget.setItemWidget(list_item, radio_button)

    def get_radio_items(self):
        try:
            # 获取输入的参数并进行验证
            fx = float(self.widgets[0].text() or 1)
            fy = float(self.widgets[1].text() or 1)
            if fx <= 0 or fy <= 0:
                raise ValueError("缩放因子X和缩放因子Y必须大于0")

            flipCode = self.widgets[2].currentData() if self.widgets[2].currentText() else 11
            angle_str = self.widgets[3].text()
            brightness = self.widgets[4].value()
            gamma = self.widgets[5].value() / 10.0
            saturation = self.widgets[6].value() / 10.0
            he_flag = self.widgets[7].currentData() if self.widgets[7].currentText() else 0
            tx = self.widgets[8].text()
            ty = self.widgets[9].text()

            # 验证旋转角度和偏移量的输入格式
            if not self.is_valid_transform_value(angle_str):
                raise ValueError("旋转参数格式无效")
            if not self.is_valid_transform_value(tx):
                raise ValueError("X轴偏移量格式无效")
            if not self.is_valid_transform_value(ty):
                raise ValueError("Y轴偏移量格式无效")

            # 处理图像
            image_path = self.directory_path88_a
            image_name = os.path.basename(image_path)
            label_path = os.path.join(self.directory_path77_all, image_name.replace(image_name.split('.')[-1], 'xml'))
            if not os.path.exists(label_path):
                label_path = None

            augmenter = DataAugmentation()
            processed_image, augmented_labels = augmenter.process_image(
                image_path, label_path, fx, fy, angle_str, flipCode, brightness, he_flag, gamma, saturation, tx, ty
            )
            if processed_image is not None:
                self.showImageOnLabel(processed_image)
        except ValueError as e:
            self.show_input_error_message(str(e))

    def is_valid_transform_value(self, value):
        if not value:
            return True
        try:
            if '~' in value:
                min_val, max_val = value.split('~')
                return float(min_val) < float(max_val)
            else:
                float(value)
            return True
        except ValueError:
            return False

    def display_image(self, checked, image_path):
        if checked:
            self.directory_path88_a = image_path
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            height, width, channel = image.shape
            bytesPerLine = channel * width
            qImage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImage)
            self.image_original_item = QGraphicsPixmapItem(pixmap)
            self.original_scene.clear()
            self.original_scene.addItem(self.image_original_item)
            self.original_scene.setSceneRect(0, 0, width, height)  # 更新场景大小
            self.fit_image_to_original_view()
            self.originalSizeLabel.setText("原图尺寸：{} X {}".format(height, width))

    def showImageOnLabel(self, image):
        height, width, channel = image.shape
        bytesPerLine = channel * width
        qImage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImage)
        self.image_enhanced_item = QGraphicsPixmapItem(pixmap)
        self.enhanced_scene.clear()
        self.enhanced_scene.addItem(self.image_enhanced_item)
        self.enhanced_scene.setSceneRect(0, 0, width, height)  # 更新场景大小
        self.fit_image_to_enhanced_view()
        self.enhancedSizeLabel.setText("增强尺寸：{} X {}".format(height, width))

    def fit_image_to_original_view(self):
        scene_rect = self.original_scene.sceneRect()
        view_rect = self.original_graphics_view.viewport().rect()
        view_transform = self.original_graphics_view.transform()
        view_transform.reset()
        self.original_graphics_view.setTransform(view_transform)
        self.original_graphics_view.fitInView(scene_rect, Qt.KeepAspectRatio)

    def fit_image_to_enhanced_view(self):
        scene_rect = self.enhanced_scene.sceneRect()
        view_rect = self.enhanced_graphics_view.viewport().rect()
        view_transform = self.enhanced_graphics_view.transform()
        view_transform.reset()
        self.enhanced_graphics_view.setTransform(view_transform)
        self.enhanced_graphics_view.fitInView(scene_rect, Qt.KeepAspectRatio)

    def a_all(self, check, opt):
        if check:
            self.opt = opt

    def make_process_picture(self):
        try:
            # 获取输入的参数并进行验证
            self.fx = float(self.widgets[0].text() or 1)
            self.fy = float(self.widgets[1].text() or 1)
            if self.fx <= 0 or self.fy <= 0 :
                raise ValueError("缩放因子X和缩放因子Y小于0")

            self.flipCode = self.widgets[2].currentData() if self.widgets[2].currentText() else 11
            self.angle_str = self.widgets[3].text()
            self.brightness = self.widgets[4].value()
            self.gamma = self.widgets[5].value() / 10.0
            self.saturation = self.widgets[6].value() / 10.0
            self.he_flag = self.widgets[7].currentData() if self.widgets[7].currentText() else 0
            self.tx = self.widgets[8].text()
            self.ty = self.widgets[9].text()

            # 验证旋转角度和偏移量的输入格式
            if not self.is_valid_transform_value(self.angle_str):
                raise ValueError("旋转参数格式无效")
            if not self.is_valid_transform_value(self.tx):
                raise ValueError("X轴偏移量格式无效")
            if not self.is_valid_transform_value(self.ty):
                raise ValueError("Y轴偏移量格式无效")

            # 获取保存文件夹路径
            save_folder = self.save_directory_path
            timestamp = datetime.now().strftime("%m%d_%H%M")
            if not save_folder:
                self.show_save_path_warning_box()
                return

            if self.opt == "a":
                # 处理单张图片
                image_paths = [self.directory_path88_a]
            elif self.opt == "all":
                # 获取所有图片路径
                path = self.directory_path77_all
                image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
                image_paths = []
                for extension in image_extensions:
                    image_paths.extend(glob.glob(os.path.join(path, extension)))
            else:
                self.show_warning_box()
                return

            # 处理并保存图片
            self.process_and_save_images(image_paths, save_folder, timestamp)
        except ValueError as e:
            self.show_input_error_message(str(e))

    def show_warning_box(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText("This is a warning message")
        msg_box.setInformativeText("请先选择生成当前图片或所有图片")
        msg_box.setWindowTitle("警告")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def show_save_path_warning_box(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText("This is a warning message")
        msg_box.setInformativeText("请先选择保存文件夹路径")
        msg_box.setWindowTitle("警告")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def show_input_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText("输入错误")
        msg_box.setInformativeText(message)
        msg_box.setWindowTitle("错误")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def process_and_save_images(self, image_paths, save_folder, timestamp):
        augmenter = DataAugmentation()
        total_images = len(image_paths)
        count = 0
        for image_path in image_paths:
            image_name_suffix = os.path.basename(image_path)
            image_name, _ = os.path.splitext(image_name_suffix)
            label_path = os.path.join(self.directory_path77_all, image_name + ".xml")
            if not os.path.exists(label_path):
                label_path = None

            # 根据参数是否有变化来构造文件名
            new_image_name = image_name
            output_firstname = 'output'
            if self.angle_str:
                new_image_name += '_rot'
                output_firstname += '_rotate'
            if self.flipCode != 11:
                new_image_name += '_flip'
                output_firstname += '_flip'
                if self.brightness != 0:
                    new_image_name += '_lt'
                    output_firstname += '_light'
                if self.he_flag != 0:
                    new_image_name += '_he'
                    output_firstname += '_he'
                if self.gamma != 1:
                    new_image_name += '_gamma'
                    output_firstname += '_gamma'
                if self.saturation != 1:
                    new_image_name += '_sat'
                    output_firstname += '_saturation'
                if self.tx or self.ty:
                    new_image_name += '_move'
                    output_firstname += '_move'
                if self.fx != 1 or self.fy != 1:
                    new_image_name += '_scale'
                    output_firstname += '_scale'
                new_image_name += f"_{timestamp}.png"
                new_label_name = new_image_name.replace('.png', '.xml')

                output_folder_images = f"{save_folder}/{output_firstname}/{timestamp}/images"
                output_folder_labels = f"{save_folder}/{output_firstname}/{timestamp}/labels"

                if not os.path.exists(output_folder_images):
                    os.makedirs(output_folder_images)
                if not os.path.exists(output_folder_labels):
                    os.makedirs(output_folder_labels)

                # 处理图像
                processed_image, augmented_labels = augmenter.process_image(
                    image_path, label_path, self.fx, self.fy, self.angle_str, self.flipCode, self.brightness,
                    self.he_flag,
                    self.gamma, self.saturation, self.tx, self.ty
                )

                if processed_image is not None:
                    save_path = os.path.join(output_folder_images, new_image_name)
                    print(save_path)
                    augmenter.save_image(processed_image, save_path)
                    if label_path is not None:
                        save_label_path = os.path.join(output_folder_labels, new_label_name)
                        print(save_label_path)
                        augmenter.save_labels(save_label_path, augmented_labels, label_path, processed_image)
                count += 1
                self.progressBar.setValue(int(count / total_images * 100))

    def updateBrightnessValue(self, value):
        self.label77[4].setText(str(value))

    def updategammaValue(self, value):
        gamma_value = value / 10.0
        self.label77[5].setText(f"{gamma_value:.1f}")

    def updatesaturationValue(self, value):
        saturation_value = value / 10.0
        self.label77[6].setText(f"{saturation_value:.1f}")

    def apply_system_theme(self):
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        palette = QPalette()
        if QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128:
            # Dark theme
            palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.Base, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
            palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        else:
            # Light theme
            palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
            palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
            palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
            palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
            palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
            palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
            palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

        QApplication.setPalette(palette)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())