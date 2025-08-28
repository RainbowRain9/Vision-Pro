
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
from typing import Dict, Any

# 添加本地 ultralytics 到 Python 路径
current_dir = Path(__file__).parent
ultralytics_path = current_dir / "ultralytics"
if ultralytics_path.exists():
    ultralytics_str = str(ultralytics_path.absolute())
    if ultralytics_str not in sys.path:
        sys.path.insert(0, ultralytics_str)
        print(f"✅ 添加 ultralytics 路径: {ultralytics_str}")

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

        # Supervision 集成
        self.supervision_wrapper = None
        self.supervision_enabled = False
        self.supervision_analyzer = None

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

        # 初始化 Supervision
        self.init_supervision()

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
        self.main_layout.setSpacing(0)

        # 创建可调整大小的分割器
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(8)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
            }
            QSplitter::handle:hover {
                background-color: #d0d0d0;
            }
        """)

        # 创建左侧面板 (图像显示)
        self.left_widget = self.setup_left_panel()
        self.splitter.addWidget(self.left_widget)

        # 创建右侧面板 (控制面板) - 带滚动功能
        self.right_widget = self.setup_right_panel()
        self.splitter.addWidget(self.right_widget)

        # 设置初始分割比例 (左侧70%, 右侧30%)
        self.splitter.setSizes([700, 300])
        self.splitter.setStretchFactor(0, 1)  # 左侧可拉伸
        self.splitter.setStretchFactor(1, 0)  # 右侧固定最小宽度

        # 将分割器添加到主布局
        self.main_layout.addWidget(self.splitter)

        # 设置菜单栏
        self.setup_menubar()

        # 设置状态栏
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setStyleSheet("QStatusBar { border-top: 1px solid #c0c0c0; }")
        self.setStatusBar(self.statusbar)

        # 设置全局样式
        self.set_style()

    def setup_menubar(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # 视图菜单
        view_menu = menubar.addMenu('视图(&V)')

        # 重置布局动作
        reset_layout_action = QtWidgets.QAction('重置布局', self)
        reset_layout_action.setShortcut('Ctrl+R')
        reset_layout_action.setStatusTip('重置界面布局到默认状态')
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)

        view_menu.addSeparator()

        # 布局预设
        layout_submenu = view_menu.addMenu('布局预设')

        # 图像优先布局
        image_priority_action = QtWidgets.QAction('图像优先 (80:20)', self)
        image_priority_action.triggered.connect(lambda: self.set_layout_ratio(80, 20))
        layout_submenu.addAction(image_priority_action)

        # 平衡布局
        balanced_action = QtWidgets.QAction('平衡布局 (70:30)', self)
        balanced_action.triggered.connect(lambda: self.set_layout_ratio(70, 30))
        layout_submenu.addAction(balanced_action)

        # 控制优先布局
        control_priority_action = QtWidgets.QAction('控制优先 (60:40)', self)
        control_priority_action.triggered.connect(lambda: self.set_layout_ratio(60, 40))
        layout_submenu.addAction(control_priority_action)

        view_menu.addSeparator()

        # 全屏切换
        fullscreen_action = QtWidgets.QAction('全屏', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

    def reset_layout(self):
        """重置布局到默认状态"""
        self.splitter.setSizes([700, 300])
        self.statusbar.showMessage("布局已重置", 2000)

    def set_layout_ratio(self, left_percent, right_percent):
        """设置布局比例"""
        total_width = self.splitter.width()
        left_width = int(total_width * left_percent / 100)
        right_width = int(total_width * right_percent / 100)
        self.splitter.setSizes([left_width, right_width])
        self.statusbar.showMessage(f"布局已调整为 {left_percent}:{right_percent}", 2000)

    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
            self.statusbar.showMessage("退出全屏模式", 2000)
        else:
            self.showFullScreen()
            self.statusbar.showMessage("进入全屏模式", 2000)

    def setup_left_panel(self):
        """设置左侧面板 - 图像显示区域"""
        left_widget = QtWidgets.QWidget()
        left_widget.setMinimumWidth(400)  # 设置最小宽度

        self.left_layout = QtWidgets.QVBoxLayout(left_widget)
        self.left_layout.setSpacing(15)
        self.left_layout.setContentsMargins(10, 10, 10, 10)

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

        return left_widget

    def setup_right_panel(self):
        """设置右侧面板 - 控制区域（带滚动功能）"""
        # 创建滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(350)  # 设置最小宽度
        scroll_area.setMaximumWidth(500)  # 设置最大宽度

        # 创建滚动内容widget
        scroll_content = QtWidgets.QWidget()
        scroll_content.setObjectName("scroll_content")

        # 设置滚动区域样式
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
        """)

        # 创建右侧控制面板布局
        self.right_layout = QtWidgets.QVBoxLayout(scroll_content)
        self.right_layout.setSpacing(15)
        self.right_layout.setContentsMargins(10, 10, 10, 10)

        # 模型选择组
        self.setup_model_group()

        # 参数设置组
        self.setup_param_group()

        # 小目标检测设置组
        self.setup_small_object_group()

        # 标注器设置组
        self.setup_annotator_group()

        # 功能按钮组
        self.setup_function_group()

        # 检测结果表格组
        self.setup_result_table_group()

        # 添加弹性空间，确保内容顶部对齐
        self.right_layout.addStretch()

        # 将内容widget设置到滚动区域
        scroll_area.setWidget(scroll_content)

        return scroll_area

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

    def setup_small_object_group(self):
        """设置小目标检测参数组"""
        self.small_obj_group = QtWidgets.QGroupBox("小目标检测设置")
        self.small_obj_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.small_obj_layout = QtWidgets.QVBoxLayout()
        self.small_obj_layout.setSpacing(10)

        # 启用小目标检测复选框
        self.enable_small_obj_checkbox = QtWidgets.QCheckBox("启用小目标检测 (InferenceSlicer)")
        self.enable_small_obj_checkbox.setToolTip("使用切片推理技术提高小目标检测精度")
        self.small_obj_layout.addWidget(self.enable_small_obj_checkbox)

        # 检测模式选择
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("检测模式:"))
        self.detection_mode_combo = QtWidgets.QComboBox()
        self.detection_mode_combo.addItems(["标准切片", "多尺度检测", "自适应切片"])
        self.detection_mode_combo.setToolTip(
            "标准切片: 固定尺寸切片\n"
            "多尺度检测: 多个尺度组合检测\n"
            "自适应切片: 根据图像尺寸自动调整"
        )
        mode_layout.addWidget(self.detection_mode_combo)
        self.small_obj_layout.addLayout(mode_layout)

        # 切片尺寸设置
        slice_layout = QtWidgets.QHBoxLayout()
        slice_layout.addWidget(QtWidgets.QLabel("切片尺寸:"))
        self.slice_size_combo = QtWidgets.QComboBox()
        self.slice_size_combo.addItems(["320x320", "640x640", "800x800", "1024x1024"])
        self.slice_size_combo.setCurrentText("640x640")
        self.slice_size_combo.setToolTip("切片尺寸越小，小目标检测效果越好，但处理时间更长")
        slice_layout.addWidget(self.slice_size_combo)
        self.small_obj_layout.addLayout(slice_layout)

        # 重叠尺寸设置
        overlap_layout = QtWidgets.QHBoxLayout()
        overlap_layout.addWidget(QtWidgets.QLabel("重叠尺寸:"))
        self.overlap_size_combo = QtWidgets.QComboBox()
        self.overlap_size_combo.addItems(["64x64", "128x128", "192x192", "256x256"])
        self.overlap_size_combo.setCurrentText("128x128")
        self.overlap_size_combo.setToolTip("重叠区域有助于检测边界目标，但会增加计算量")
        overlap_layout.addWidget(self.overlap_size_combo)
        self.small_obj_layout.addLayout(overlap_layout)

        # 性能提示标签
        self.performance_hint_label = QtWidgets.QLabel("💡 提示: 启用小目标检测会增加处理时间")
        self.performance_hint_label.setStyleSheet("color: #666; font-size: 10px;")
        self.performance_hint_label.setWordWrap(True)
        self.small_obj_layout.addWidget(self.performance_hint_label)

        self.small_obj_group.setLayout(self.small_obj_layout)
        self.right_layout.addWidget(self.small_obj_group)

    def setup_annotator_group(self):
        """设置标注器控制组"""
        self.annotator_group = QtWidgets.QGroupBox("标注器设置")
        self.annotator_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.annotator_layout = QtWidgets.QVBoxLayout()
        self.annotator_layout.setSpacing(8)

        # 预设选择
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("预设:"))
        self.annotator_preset_combo = QtWidgets.QComboBox()
        self.annotator_preset_combo.addItems([
            "basic", "detailed", "privacy", "analysis", "segmentation", "presentation"
        ])
        self.annotator_preset_combo.setToolTip("选择标注器预设配置")
        preset_layout.addWidget(self.annotator_preset_combo)
        self.annotator_layout.addLayout(preset_layout)

        # 单个标注器开关
        annotators_layout = QtWidgets.QGridLayout()

        # 创建标注器复选框
        self.annotator_checkboxes = {}
        annotator_configs = [
            ("box", "边界框", "显示目标边界框"),
            ("label", "标签", "显示类别和置信度"),
            ("mask", "掩码", "显示分割掩码"),
            ("polygon", "多边形", "显示多边形轮廓"),
            ("heatmap", "热力图", "显示目标密度热力图"),
            ("blur", "模糊", "模糊检测区域"),
            ("pixelate", "像素化", "像素化检测区域")
        ]

        for i, (key, name, tooltip) in enumerate(annotator_configs):
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setToolTip(tooltip)
            checkbox.setObjectName(f"annotator_{key}")
            self.annotator_checkboxes[key] = checkbox

            # 默认启用基础标注器
            if key in ["box", "label"]:
                checkbox.setChecked(True)

            # 布局：2列
            row = i // 2
            col = i % 2
            annotators_layout.addWidget(checkbox, row, col)

        self.annotator_layout.addLayout(annotators_layout)

        # 控制按钮
        button_layout = QtWidgets.QHBoxLayout()

        self.apply_preset_btn = QtWidgets.QPushButton("应用预设")
        self.apply_preset_btn.setStyleSheet(
            "QPushButton { padding: 4px 8px; background-color: #2196F3; color: white; border-radius: 3px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )

        self.clear_heatmap_btn = QtWidgets.QPushButton("清除热力图")
        self.clear_heatmap_btn.setStyleSheet(
            "QPushButton { padding: 4px 8px; background-color: #FF9800; color: white; border-radius: 3px; }"
            "QPushButton:hover { background-color: #F57C00; }"
        )

        button_layout.addWidget(self.apply_preset_btn)
        button_layout.addWidget(self.clear_heatmap_btn)
        self.annotator_layout.addLayout(button_layout)

        # 状态显示
        self.annotator_status_label = QtWidgets.QLabel("状态: 基础模式")
        self.annotator_status_label.setStyleSheet("color: #666; font-size: 10px;")
        self.annotator_layout.addWidget(self.annotator_status_label)

        self.annotator_group.setLayout(self.annotator_layout)
        self.right_layout.addWidget(self.annotator_group)

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

        # 标注器控制信号
        self.apply_preset_btn.clicked.connect(self.apply_annotator_preset)
        self.clear_heatmap_btn.clicked.connect(self.clear_heatmap_history)

        # 标注器复选框信号
        for key, checkbox in self.annotator_checkboxes.items():
            checkbox.stateChanged.connect(lambda state, annotator=key: self.toggle_annotator(annotator, state))

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
            # 检查是否启用小目标检测
            if (self.supervision_enabled and self.supervision_wrapper and
                hasattr(self, 'enable_small_obj_checkbox') and
                self.enable_small_obj_checkbox.isChecked()):
                self.detect_image_with_small_objects(file_path)
                return

            # 如果启用了 Supervision，使用增强检测
            elif self.supervision_enabled and self.supervision_wrapper:
                self.detect_image_with_supervision(file_path)
                return

            # 原始检测方法（向后兼容）
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

    def init_supervision(self):
        """初始化 Supervision 功能"""
        try:
            from scripts.modules.supervision_wrapper import SupervisionWrapper, SupervisionAnalyzer

            # VisDrone 类别名称
            visdrone_classes = [
                'pedestrian', 'people', 'bicycle', 'car', 'van',
                'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
            ]

            self.supervision_wrapper = SupervisionWrapper(class_names=visdrone_classes)
            self.supervision_analyzer = SupervisionAnalyzer()
            self.supervision_enabled = True

            self.statusbar.showMessage("Supervision 增强功能已启用", 3000)
            self.logger.info("Supervision 集成成功")

        except ImportError as e:
            self.supervision_enabled = False
            self.logger.warning(f"Supervision 未安装或导入失败: {e}")
            QMessageBox.information(
                self, "提示",
                "Supervision 功能未启用\n"
                "如需使用增强可视化功能，请安装: pip install supervision"
            )
        except Exception as e:
            self.supervision_enabled = False
            self.logger.error(f"Supervision 初始化失败: {e}")

    def apply_annotator_preset(self):
        """应用标注器预设"""
        if not self.supervision_enabled or not self.supervision_wrapper:
            QMessageBox.warning(self, "警告", "Supervision 功能未启用")
            return

        preset_name = self.annotator_preset_combo.currentText()
        try:
            self.supervision_wrapper.set_annotator_preset(preset_name)

            # 更新复选框状态
            enabled_annotators = self.supervision_wrapper.get_enabled_annotators()
            for key, checkbox in self.annotator_checkboxes.items():
                checkbox.blockSignals(True)  # 阻止信号避免递归
                checkbox.setChecked(key in enabled_annotators)
                checkbox.blockSignals(False)

            # 更新状态显示
            self.annotator_status_label.setText(f"状态: {preset_name} 模式")
            self.statusbar.showMessage(f"已应用预设: {preset_name}", 2000)
            self.logger.info(f"应用标注器预设: {preset_name}")

        except Exception as e:
            error_msg = f"应用预设失败: {str(e)}"
            QMessageBox.critical(self, "错误", error_msg)
            self.logger.error(error_msg)

    def toggle_annotator(self, annotator_type: str, state: int):
        """切换标注器状态"""
        if not self.supervision_enabled or not self.supervision_wrapper:
            return

        try:
            if state == 2:  # Qt.Checked
                self.supervision_wrapper.enable_annotator(annotator_type)
                action = "启用"
            else:  # Qt.Unchecked
                self.supervision_wrapper.disable_annotator(annotator_type)
                action = "禁用"

            # 更新状态显示
            enabled_count = len(self.supervision_wrapper.get_enabled_annotators())
            self.annotator_status_label.setText(f"状态: 自定义模式 ({enabled_count} 个标注器)")

            self.logger.info(f"{action}标注器: {annotator_type}")

        except Exception as e:
            self.logger.error(f"切换标注器 {annotator_type} 失败: {e}")

    def clear_heatmap_history(self):
        """清除热力图历史数据"""
        if not self.supervision_enabled or not self.supervision_wrapper:
            QMessageBox.warning(self, "警告", "Supervision 功能未启用")
            return

        try:
            self.supervision_wrapper.clear_heatmap_history()
            self.statusbar.showMessage("热力图历史数据已清除", 2000)
            self.logger.info("清除热力图历史数据")

        except Exception as e:
            error_msg = f"清除热力图历史失败: {str(e)}"
            QMessageBox.critical(self, "错误", error_msg)
            self.logger.error(error_msg)

    def detect_image_with_supervision(self, file_path: str):
        """使用 Supervision 增强的图像检测"""
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

            self.statusbar.showMessage("正在使用 Supervision 增强检测...")
            QtWidgets.QApplication.processEvents()

            # YOLO 检测
            results = self.model.predict(img, conf=conf, iou=iou)

            # Supervision 增强处理
            processed_result = self.supervision_wrapper.process_ultralytics_results(
                results[0], img
            )

            # 显示增强结果
            enhanced_image = processed_result['annotated_image']
            self.display_image(enhanced_image, self.result_img_label)
            self.current_result = enhanced_image.copy()

            # 更新结果表格（使用原始结果）
            self.update_result_table(results[0])

            # 添加到分析器
            self.supervision_analyzer.add_detection_result(processed_result)

            # 显示统计信息
            self.show_supervision_statistics(processed_result['statistics'])

            self.save_btn.setEnabled(True)
            detection_count = processed_result['detection_count']
            self.statusbar.showMessage(
                f"Supervision 增强检测完成: {os.path.basename(file_path)} "
                f"(检测到 {detection_count} 个目标)", 3000
            )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Supervision 检测失败: {str(e)}")
            self.statusbar.showMessage("Supervision 检测失败", 3000)
            self.logger.error(f"Supervision 检测错误: {e}")

    def show_supervision_statistics(self, statistics: Dict):
        """显示 Supervision 统计信息"""
        if not statistics:
            return

        # 生成统计摘要
        summary = self.supervision_wrapper.generate_detection_summary(statistics)

        # 在状态栏显示简要信息
        total = statistics.get('total_detections', 0)
        avg_conf = statistics.get('confidence_stats', {}).get('mean', 0)

        status_msg = f"检测统计: {total} 个目标, 平均置信度: {avg_conf:.3f}"
        self.statusbar.showMessage(status_msg, 5000)

        # 记录详细统计到日志
        self.logger.info(f"检测统计信息:\n{summary}")

    def detect_image_with_small_objects(self, file_path: str):
        """使用小目标检测功能的图像检测"""
        try:
            # 读取图片
            img = cv2.imread(file_path)
            if img is None:
                raise Exception("无法读取图像文件")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 显示原始图片
            self.display_image(img, self.original_img_label)
            self.current_image = img.copy()

            # 获取检测参数
            conf = self.conf_slider.value() / 100
            iou = self.iou_slider.value() / 100

            # 获取小目标检测配置
            detection_mode = self.detection_mode_combo.currentText()
            slice_size_text = self.slice_size_combo.currentText()
            overlap_size_text = self.overlap_size_combo.currentText()

            # 解析尺寸
            slice_w, slice_h = map(int, slice_size_text.split('x'))
            overlap_w, overlap_h = map(int, overlap_size_text.split('x'))

            self.statusbar.showMessage(f"正在使用{detection_mode}进行小目标检测...")
            QtWidgets.QApplication.processEvents()

            # 根据检测模式选择方法
            if detection_mode == "多尺度检测":
                result = self.supervision_wrapper.detect_with_multiple_scales(
                    img, self.model, conf, iou
                )
            elif detection_mode == "自适应切片":
                # 获取最优配置
                optimal_config = self.supervision_wrapper.get_optimal_slice_config(img.shape[:2])
                result = self.supervision_wrapper.detect_small_objects(
                    img, self.model, conf, iou,
                    slice_wh=optimal_config['slice_wh'],
                    overlap_wh=optimal_config['overlap_wh']
                )
            else:  # 标准切片
                result = self.supervision_wrapper.detect_small_objects(
                    img, self.model, conf, iou,
                    slice_wh=(slice_w, slice_h),
                    overlap_wh=(overlap_w, overlap_h)
                )

            # 显示检测结果
            if 'error' not in result:
                enhanced_image = result['annotated_image']
                self.display_image(enhanced_image, self.result_img_label)
                self.current_result = enhanced_image.copy()

                # 更新结果表格（如果有原始检测结果）
                if result['detections'] is not None:
                    # 创建一个模拟的 ultralytics 结果对象用于表格显示
                    self.update_small_object_result_table(result)

                # 显示统计信息
                self.show_small_object_statistics(result['statistics'])

                self.save_btn.setEnabled(True)
                detection_count = result['detection_count']
                method = result.get('method', '小目标检测')
                processing_time = result['statistics'].get('processing_time', 0)

                self.statusbar.showMessage(
                    f"{method}完成: {os.path.basename(file_path)} "
                    f"(检测到 {detection_count} 个目标, 耗时 {processing_time:.2f}s)", 5000
                )
            else:
                raise Exception(result['error'])

        except Exception as e:
            QMessageBox.critical(self, "错误", f"小目标检测失败: {str(e)}")
            self.statusbar.showMessage("小目标检测失败", 3000)
            self.logger.error(f"小目标检测错误: {e}")

    def update_small_object_result_table(self, result: Dict):
        """更新小目标检测结果表格"""
        self.result_table.setRowCount(0)

        detections = result['detections']
        if detections is None or len(detections.xyxy) == 0:
            return

        labels = result['labels']

        for i in range(len(detections.xyxy)):
            # 获取检测信息
            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = map(int, bbox)

            confidence = detections.confidence[i] if detections.confidence is not None else 0.0
            class_id = int(detections.class_id[i]) if detections.class_id is not None else 0

            # 获取类别名称
            if i < len(labels):
                # 从标签中提取类别名称（格式: "class_name: confidence"）
                class_name = labels[i].split(':')[0].strip()
            elif class_id < len(self.supervision_wrapper.class_names):
                class_name = self.supervision_wrapper.class_names[class_id]
            else:
                class_name = f"Class_{class_id}"

            # 添加新行
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)

            # 设置单元格内容
            self.result_table.setItem(row_position, 0, QTableWidgetItem(class_name))
            self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{confidence:.2f}"))
            self.result_table.setItem(row_position, 2, QTableWidgetItem(f"({x1}, {y1})"))
            self.result_table.setItem(row_position, 3, QTableWidgetItem(f"({x2}, {y2})"))

    def show_small_object_statistics(self, statistics: Dict):
        """显示小目标检测统计信息"""
        if not statistics:
            return

        # 更新性能提示标签
        if hasattr(self, 'performance_hint_label'):
            processing_time = statistics.get('processing_time', 0)
            detection_count = statistics.get('total_detections', 0)

            if 'slice_config' in statistics:
                slice_info = statistics['slice_config']
                total_slices = slice_info.get('total_slices', 0)
                hint_text = (f"✅ 检测完成: {detection_count} 个目标, "
                           f"处理 {total_slices} 个切片, 耗时 {processing_time:.2f}s")
            else:
                hint_text = f"✅ 检测完成: {detection_count} 个目标, 耗时 {processing_time:.2f}s"

            self.performance_hint_label.setText(hint_text)
            self.performance_hint_label.setStyleSheet("color: #4CAF50; font-size: 10px;")

        # 记录详细统计到日志
        summary = self.supervision_wrapper.generate_detection_summary(statistics)
        self.logger.info(f"小目标检测统计信息:\n{summary}")

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
