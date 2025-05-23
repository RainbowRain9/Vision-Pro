#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOvision Pro - 目标检测系统主程序
支持标准 YOLOv8 和 Drone-YOLO 模型
"""

import os
import sys
import cv2
import datetime
import logging
import yaml
from pathlib import Path
from ultralytics import YOLO
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox,
                            QTableWidgetItem, QStyledItemDelegate, QHeaderView)


class CenteredDelegate(QStyledItemDelegate):
    """表格内容居中显示的代理类"""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class YOLODetectionUI(QMainWindow):
    """YOLOvision Pro 目标检测系统主界面类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOvision Pro - 目标检测系统")
        self.resize(1400, 900)

        # 设置窗口图标
        if hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, 'icon.ico')
        else:
            icon_path = 'icon.ico'
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # 初始化变量
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.is_camera_running = False
        self.current_image = None
        self.current_result = None
        self.video_writer = None
        self.current_config = None

        # 设置项目路径 - 适应新的目录结构
        self.project_root = Path(__file__).parent
        self.models_path = self.project_root / "models"
        self.configs_path = self.project_root / "assets" / "configs"
        self.results_path = self.project_root / "results"
        self.outputs_path = self.project_root / "outputs"
        self.scripts_path = self.project_root / "scripts"
        self.docs_path = self.project_root / "docs"

        # 创建必要的输出目录
        self.ensure_directories()

        # 设置日志
        self.setup_logging()

        # 初始化UI
        self.setup_ui()

        # 连接信号槽
        self.connect_signals()

    def ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.results_path / "images",
            self.results_path / "videos",
            self.results_path / "camera",
            self.outputs_path / "models",
            self.outputs_path / "logs",
            self.outputs_path / "results"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """设置日志系统"""
        log_file = self.outputs_path / "logs" / f"yolovision_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_ui(self):
        """设置UI界面"""
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)

        # 主布局
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(15)

        # 创建左侧布局 (图像显示)
        self.setup_left_panel()

        # 创建右侧布局 (控制面板)
        self.setup_right_panel()

        # 设置状态栏
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setStyleSheet("QStatusBar { border-top: 1px solid #c0c0c0; }")
        self.setStatusBar(self.statusbar)

        # 设置全局样式
        self.set_style()

    def setup_left_panel(self):
        """设置左侧面板 - 图像显示区域"""
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.setSpacing(15)

        # 原始图像组
        self.original_group = QtWidgets.QGroupBox("原始图像")
        self.original_group.setMinimumHeight(400)
        self.original_img_label = QtWidgets.QLabel()
        self.original_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_img_label.setText("等待加载图像...")
        self.original_img_label.setStyleSheet("background-color: #F0F0F0; border: 1px solid #CCCCCC;")

        original_layout = QtWidgets.QVBoxLayout()
        original_layout.addWidget(self.original_img_label)
        self.original_group.setLayout(original_layout)
        self.left_layout.addWidget(self.original_group)

        # 检测结果图像组
        self.result_group = QtWidgets.QGroupBox("检测结果")
        self.result_group.setMinimumHeight(400)
        self.result_img_label = QtWidgets.QLabel()
        self.result_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_img_label.setText("检测结果将显示在这里")
        self.result_img_label.setStyleSheet("background-color: #F0F0F0; border: 1px solid #CCCCCC;")

        result_layout = QtWidgets.QVBoxLayout()
        result_layout.addWidget(self.result_img_label)
        self.result_group.setLayout(result_layout)
        self.left_layout.addWidget(self.result_group)

        self.main_layout.addLayout(self.left_layout, stretch=3)

    def setup_right_panel(self):
        """设置右侧面板 - 控制区域"""
        self.right_layout = QtWidgets.QVBoxLayout()
        self.right_layout.setSpacing(15)

        # 模型选择组
        self.setup_model_group()

        # 参数设置组
        self.setup_param_group()

        # 功能按钮组
        self.setup_function_group()

        # 检测结果表格组
        self.setup_result_table_group()

        self.main_layout.addLayout(self.right_layout, stretch=1)

    def setup_model_group(self):
        """设置模型选择组"""
        self.model_group = QtWidgets.QGroupBox("模型设置")
        self.model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.model_layout = QtWidgets.QVBoxLayout()

        # 模型类型选择
        model_type_layout = QtWidgets.QHBoxLayout()
        model_type_layout.addWidget(QtWidgets.QLabel("模型类型:"))
        self.model_type_combo = QtWidgets.QComboBox()
        self.model_type_combo.addItems(["预训练模型", "自定义配置"])
        model_type_layout.addWidget(self.model_type_combo)
        self.model_layout.addLayout(model_type_layout)

        # 模型/配置选择
        self.model_combo = QtWidgets.QComboBox()
        self.update_model_options()

        # 加载模型按钮
        self.load_model_btn = QtWidgets.QPushButton(" 加载模型")
        self.load_model_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_model_btn.setStyleSheet(
            "QPushButton { padding: 8px; background-color: #4CAF50; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )

        # 模型信息显示
        self.model_info_label = QtWidgets.QLabel("未加载模型")
        self.model_info_label.setStyleSheet("color: #666; font-size: 10px;")
        self.model_info_label.setWordWrap(True)

        self.model_layout.addWidget(QtWidgets.QLabel("选择模型/配置:"))
        self.model_layout.addWidget(self.model_combo)
        self.model_layout.addWidget(self.load_model_btn)
        self.model_layout.addWidget(self.model_info_label)
        self.model_group.setLayout(self.model_layout)
        self.right_layout.addWidget(self.model_group)

    def update_model_options(self):
        """更新模型选项"""
        self.model_combo.clear()

        if self.model_type_combo.currentText() == "预训练模型":
            # 加载预训练模型文件
            model_files = self.get_model_files()
            if model_files:
                self.model_combo.addItems(model_files)
            else:
                self.model_combo.addItems(["yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        else:
            # 加载配置文件
            config_files = self.get_config_files()
            if config_files:
                self.model_combo.addItems(config_files)
            else:
                self.model_combo.addItems(["yolov8s-drone.yaml"])

    def get_config_files(self):
        """获取配置文件列表"""
        config_files = []
        if self.configs_path.exists():
            for file in self.configs_path.glob("*.yaml"):
                config_files.append(file.name)
        return sorted(config_files)

    def setup_param_group(self):
        """设置参数设置组"""
        self.param_group = QtWidgets.QGroupBox("检测参数")
        self.param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.param_layout = QtWidgets.QFormLayout()
        self.param_layout.setLabelAlignment(Qt.AlignLeft)
        self.param_layout.setFormAlignment(Qt.AlignLeft)
        self.param_layout.setVerticalSpacing(15)

        # 置信度滑块
        self.conf_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_value = QtWidgets.QLabel("0.25")
        self.conf_value.setAlignment(Qt.AlignCenter)
        self.conf_value.setStyleSheet("font-weight: bold; color: #2196F3;")

        # IoU滑块
        self.iou_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)
        self.iou_value = QtWidgets.QLabel("0.45")
        self.iou_value.setAlignment(Qt.AlignCenter)
        self.iou_value.setStyleSheet("font-weight: bold; color: #2196F3;")

        self.param_layout.addRow("置信度阈值:", self.conf_slider)
        self.param_layout.addRow("当前值:", self.conf_value)
        self.param_layout.addRow(QtWidgets.QLabel(""))  # 空行
        self.param_layout.addRow("IoU阈值:", self.iou_slider)
        self.param_layout.addRow("当前值:", self.iou_value)

        self.param_group.setLayout(self.param_layout)
        self.right_layout.addWidget(self.param_group)

    def setup_function_group(self):
        """设置功能按钮组"""
        self.func_group = QtWidgets.QGroupBox("检测功能")
        self.func_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.func_layout = QtWidgets.QVBoxLayout()
        self.func_layout.setSpacing(10)

        # 图片检测按钮
        self.image_btn = QtWidgets.QPushButton(" 图片检测")
        self.image_btn.setIcon(QIcon.fromTheme("image-x-generic"))

        # 视频检测按钮
        self.video_btn = QtWidgets.QPushButton(" 视频检测")
        self.video_btn.setIcon(QIcon.fromTheme("video-x-generic"))

        # 摄像头检测按钮
        self.camera_btn = QtWidgets.QPushButton(" 摄像头检测")
        self.camera_btn.setIcon(QIcon.fromTheme("camera-web"))

        # 停止检测按钮
        self.stop_btn = QtWidgets.QPushButton(" 停止检测")
        self.stop_btn.setIcon(QIcon.fromTheme("process-stop"))
        self.stop_btn.setEnabled(False)

        # 保存结果按钮
        self.save_btn = QtWidgets.QPushButton(" 保存结果")
        self.save_btn.setIcon(QIcon.fromTheme("document-save"))
        self.save_btn.setEnabled(False)

        # 设置按钮样式
        button_style = """
        QPushButton {
            padding: 10px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            text-align: left;
        }
        QPushButton:hover {
            background-color: #0b7dda;
        }
        QPushButton:disabled {
            background-color: #cccccc;
        }
        """

        for btn in [self.image_btn, self.video_btn, self.camera_btn,
                    self.stop_btn, self.save_btn]:
            btn.setStyleSheet(button_style)
            self.func_layout.addWidget(btn)

        self.func_group.setLayout(self.func_layout)
        self.right_layout.addWidget(self.func_group)

    def setup_result_table_group(self):
        """设置检测结果表格组"""
        self.table_group = QtWidgets.QGroupBox("检测结果详情")
        self.table_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.table_layout = QtWidgets.QVBoxLayout()

        self.result_table = QtWidgets.QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["类别", "置信度", "左上坐标", "右下坐标"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.result_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # 设置表格样式
        self.result_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                alternate-background-color: #f5f5f5;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 5px;
                border: none;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)

        # 设置居中代理
        delegate = CenteredDelegate(self.result_table)
        self.result_table.setItemDelegate(delegate)

        self.table_layout.addWidget(self.result_table)
        self.table_group.setLayout(self.table_layout)
        self.right_layout.addWidget(self.table_group, stretch=1)

    def set_style(self):
        """设置全局样式"""
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QLabel {
            color: #333333;
        }
        QComboBox {
            padding: 5px;
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 16px;
            height: 16px;
            margin: -5px 0;
            background: #2196F3;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal {
            background: #2196F3;
            border-radius: 3px;
        }
        """
        self.centralwidget.setStyleSheet(style)

    def connect_signals(self):
        """连接信号槽"""
        self.model_type_combo.currentTextChanged.connect(self.update_model_options)
        self.load_model_btn.clicked.connect(self.load_model)
        self.image_btn.clicked.connect(self.detect_image)
        self.video_btn.clicked.connect(self.detect_video)
        self.camera_btn.clicked.connect(self.detect_camera)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.save_btn.clicked.connect(self.save_result)
        self.conf_slider.valueChanged.connect(self.update_conf_value)
        self.iou_slider.valueChanged.connect(self.update_iou_value)
        self.timer.timeout.connect(self.update_camera_frame)

    def get_model_files(self):
        """获取models目录下的模型文件"""
        model_files = []
        if self.models_path.exists():
            for file in self.models_path.glob("*.pt"):
                model_files.append(file.name)
        return sorted(model_files)

    def load_model(self):
        """加载YOLO模型"""
        try:
            model_name = self.model_combo.currentText()

            if self.model_type_combo.currentText() == "预训练模型":
                # 加载预训练模型
                model_path = self.models_path / model_name

                if model_path.exists():
                    self.model = YOLO(str(model_path))
                    self.current_config = None
                    model_info = f"预训练模型: {model_name}"
                else:
                    # 尝试从 ultralytics 下载
                    self.model = YOLO(model_name)
                    self.current_config = None
                    model_info = f"在线模型: {model_name}"

            else:
                # 加载自定义配置
                config_path = self.configs_path / model_name

                if config_path.exists():
                    # 读取配置文件信息
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)

                    self.model = YOLO(str(config_path))
                    self.current_config = config_data

                    # 显示配置信息
                    if 'nc' in config_data:
                        model_info = f"自定义配置: {model_name}\n类别数: {config_data['nc']}"
                    else:
                        model_info = f"自定义配置: {model_name}"

                    # 特别标识 Drone-YOLO
                    if 'drone' in model_name.lower():
                        model_info += "\n🚁 Drone-YOLO (小目标优化)"
                else:
                    raise FileNotFoundError(f"配置文件不存在: {config_path}")

            # 更新UI状态
            self.model_info_label.setText(model_info)
            self.statusbar.showMessage(f"模型加载成功: {model_name}", 3000)
            self.image_btn.setEnabled(True)
            self.video_btn.setEnabled(True)
            self.camera_btn.setEnabled(True)

            # 记录日志
            self.logger.info(f"模型加载成功: {model_name}")

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            QMessageBox.critical(self, "错误", error_msg)
            self.model_info_label.setText("模型加载失败")
            self.logger.error(error_msg)

    def update_conf_value(self):
        """更新置信度值显示"""
        conf = self.conf_slider.value() / 100
        self.conf_value.setText(f"{conf:.2f}")

    def update_iou_value(self):
        """更新IoU值显示"""
        iou = self.iou_slider.value() / 100
        self.iou_value.setText(f"{iou:.2f}")

    def display_image(self, img, label):
        """在标签控件中显示图像"""
        h, w, c = img.shape
        bytes_per_line = c * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 根据标签大小缩放图像
        label_size = label.size()
        scaled_img = q_img.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label.setPixmap(QPixmap.fromImage(scaled_img))

    def update_result_table(self, result):
        """更新检测结果表格"""
        self.result_table.setRowCount(0)

        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for _, box in enumerate(boxes):  # 使用_忽略索引变量
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = result.names[class_id]

                # 添加新行
                row_position = self.result_table.rowCount()
                self.result_table.insertRow(row_position)

                # 设置单元格内容
                self.result_table.setItem(row_position, 0, QTableWidgetItem(class_name))
                self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{conf:.2f}"))
                self.result_table.setItem(row_position, 2, QTableWidgetItem(f"({x1}, {y1})"))
                self.result_table.setItem(row_position, 3, QTableWidgetItem(f"({x2}, {y2})"))

    def detect_image(self):
        """图片检测功能"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if file_path:
            try:
                # 读取图片
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 显示原始图片
                self.display_image(img, self.original_img_label)
                self.current_image = img.copy()

                # 检测图片
                conf = self.conf_slider.value() / 100
                iou = self.iou_slider.value() / 100

                self.statusbar.showMessage("正在检测图片...")
                QtWidgets.QApplication.processEvents()  # 更新UI

                results = self.model.predict(img, conf=conf, iou=iou)
                result_img = results[0].plot()

                # 显示检测结果
                self.display_image(result_img, self.result_img_label)
                self.current_result = result_img.copy()

                # 更新结果表格
                self.update_result_table(results[0])

                self.save_btn.setEnabled(True)
                self.statusbar.showMessage(f"图片检测完成: {os.path.basename(file_path)}", 3000)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"图片检测失败: {str(e)}")
                self.statusbar.showMessage("图片检测失败", 3000)

    def detect_video(self):
        """视频检测功能"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if file_path:
            try:
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    raise Exception("无法打开视频文件")

                # 获取视频信息
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # 创建视频结果目录
                video_results_dir = self.results_path / "videos"
                video_results_dir.mkdir(parents=True, exist_ok=True)

                # 创建视频写入器
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = video_results_dir / f"output_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

                # 启用停止按钮，禁用其他按钮
                self.stop_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                self.image_btn.setEnabled(False)
                self.video_btn.setEnabled(False)
                self.camera_btn.setEnabled(False)

                # 开始处理视频
                self.timer.start(30)  # 30ms间隔
                self.statusbar.showMessage(f"正在处理视频: {os.path.basename(file_path)}...")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"视频检测失败: {str(e)}")
                self.statusbar.showMessage("视频检测失败", 3000)

    def detect_camera(self):
        """摄像头检测功能"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        try:
            self.cap = cv2.VideoCapture(0)  # 使用默认摄像头
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")

            # 获取摄像头信息
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建摄像头结果目录
            camera_results_dir = self.results_path / "camera"
            camera_results_dir.mkdir(parents=True, exist_ok=True)

            # 创建视频写入器
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = camera_results_dir / f"camera_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(output_file), fourcc, 20, (width, height))

            # 启用停止按钮，禁用其他按钮
            self.stop_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.image_btn.setEnabled(False)
            self.video_btn.setEnabled(False)
            self.camera_btn.setEnabled(False)

            # 设置摄像头运行标志
            self.is_camera_running = True

            # 开始处理视频
            self.timer.start(30)  # 30ms间隔
            self.statusbar.showMessage("正在使用摄像头检测...")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"摄像头检测失败: {str(e)}")
            self.statusbar.showMessage("摄像头检测失败", 3000)

    def update_camera_frame(self):
        """更新摄像头/视频帧"""
        if self.cap is None or not self.cap.isOpened():
            self.stop_detection()
            return

        ret, frame = self.cap.read()
        if not ret:
            # 视频结束
            self.stop_detection()
            self.statusbar.showMessage("视频处理完成", 3000)
            return

        # 显示原始帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_image(frame_rgb, self.original_img_label)
        self.current_image = frame_rgb.copy()

        # 检测帧
        conf = self.conf_slider.value() / 100
        iou = self.iou_slider.value() / 100

        results = self.model.predict(frame_rgb, conf=conf, iou=iou)
        result_img = results[0].plot()

        # 显示检测结果
        self.display_image(result_img, self.result_img_label)
        self.current_result = result_img.copy()

        # 更新结果表格
        self.update_result_table(results[0])

        # 写入视频
        if self.video_writer is not None:
            self.video_writer.write(cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    def stop_detection(self):
        """停止检测"""
        # 停止定时器
        self.timer.stop()

        # 释放视频资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # 关闭视频写入器
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # 重置摄像头标志
        self.is_camera_running = False

        # 恢复按钮状态
        self.stop_btn.setEnabled(False)
        self.image_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.camera_btn.setEnabled(True)

        self.statusbar.showMessage("检测已停止", 3000)

    def save_result(self):
        """保存检测结果"""
        if self.current_result is None:
            QMessageBox.warning(self, "警告", "没有可保存的检测结果")
            return

        # 创建图片结果目录
        image_results_dir = self.results_path / "images"
        image_results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"result_{timestamp}.jpg"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", str(image_results_dir / default_name),
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )

        if file_path:
            try:
                # 保存检测结果图像
                cv2.imwrite(file_path, cv2.cvtColor(self.current_result, cv2.COLOR_RGB2BGR))

                # 同时保存到 outputs/results 目录
                outputs_results_dir = self.outputs_path / "results"
                outputs_results_dir.mkdir(parents=True, exist_ok=True)
                backup_path = outputs_results_dir / f"result_{timestamp}.jpg"
                cv2.imwrite(str(backup_path), cv2.cvtColor(self.current_result, cv2.COLOR_RGB2BGR))

                self.statusbar.showMessage(f"结果已保存至: {file_path}", 3000)
                self.logger.info(f"检测结果已保存: {file_path}")

            except Exception as e:
                error_msg = f"保存结果失败: {str(e)}"
                QMessageBox.critical(self, "错误", error_msg)
                self.statusbar.showMessage("保存结果失败", 3000)
                self.logger.error(error_msg)

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止所有正在进行的检测
        self.stop_detection()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = YOLODetectionUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
