
# YOLO 项目

## 1. 概述

本文档旨在为使用 LabelMe 进行图像标注并为 YOLO (You Only Look Once) 目标检测项目准备数据集提供一份详细的指南。内容涵盖 LabelMe 的安装、YOLO 项目的推荐目录结构、详细的标注流程（从原始图像到 YOLO 格式数据集的转换），以及相关的 Python 脚本和标注技巧。

## 2. 项目目录结构

为了在 YOLO 目标检测项目中高效管理数据和脚本，建议采用以下目录结构：
```
yolo_project/
├── data/
│   ├── raw_images/              # 存放原始未标注图像
│   ├── annotations/             # 存放 LabelMe 生成的 JSON 格式标注文件
│   ├── yolo_dataset/            # 存放转换后的 YOLO 格式数据集
│   │   ├── images/
│   │   │   ├── train/          # 训练集图像
│   │   │   ├── val/            # 验证集图像
│   │   │   └── test/           # 测试集图像（可选）
│   │   ├── labels/
│   │   │   ├── train/          # 训练集 YOLO 格式标签
│   │   │   ├── val/            # 验证集 YOLO 格式标签
│   │   │   └── test/           # 测试集 YOLO 格式标签（可选）
│   │   └── data.yaml           # YOLO 数据集配置文件 (例如 COCO 格式)
│   └── classes.txt              # 类别列表文件，每行一个类别名称
├── scripts/
│   ├── labelme2yolo.py          # 转换脚本：将 LabelMe JSON 标注转换为 YOLO 格式
│   └── split_dataset.py         # 数据集划分脚本：将数据集划分为训练集、验证集和测试集
├── models/                      # 存放训练好的模型权重文件
└── results/                     # 存放模型测试结果、可视化输出等
```

## 3. LabelMe 安装指南

LabelMe 是一个强大的图像标注工具，支持多边形、矩形、圆形、线条和点的标注。

### 3.1. 方法一：使用 pip 安装（推荐）

1.  **确保已安装 Python** (推荐 Python 3.7 或更高版本)。
2.  打开命令行终端 (Windows 用户可使用 CMD 或 PowerShell)。
3.  执行安装命令：
    ```bash
    pip install labelme
	```
4.  验证安装：
    ```bash
    labelme --help
    ```
    如果显示帮助信息，则表示安装成功。

### 3.2. 方法二：从 GitHub 安装最新版本

如果您需要最新的功能或修复：
```bash
pip install git+https://github.com/wkentaro/labelme.git
```

### 3.3. 方法三：使用 conda 安装（适合 Anaconda 用户）

如果您使用 Anaconda 管理 Python 环境：
```bash
conda create -n labelme python=3.8
conda activate labelme
pip install labelme
```

### 3.4. 可能遇到的问题及解决方案

1.  **权限问题**：
    *   Windows: 以管理员身份运行命令提示符。
    *   Linux/Mac: 使用 `sudo pip install labelme`。
2.  **PyQt 依赖问题**：如果安装过程中提示 PyQt 相关错误，可先尝试安装 PyQt5：
    ```bash
    pip install PyQt5
    ```
3.  **版本冲突**：建议在虚拟环境中安装 LabelMe 以避免与其他包的冲突：
    ```bash
    python -m venv labelme_env
    # Windows
    labelme_env\Scripts\activate
    # Linux/Mac
    source labelme_env/bin/activate

    pip install labelme
    ```

## 4. 数据准备与标注流程

### 4.1. 准备原始图像

将所有待标注的图像文件放入项目根目录下的 `data/raw_images/` 文件夹中。

### 4.2. 准备类别文件 `classes.txt`

在 `data/classes.txt` 文件中列出您项目所需的所有对象类别名称，每行一个类别。例如：
```
person
car
bicycle
traffic_light
```
此文件将用于 LabelMe 标注时的标签预设以及后续转换为 YOLO 格式时的类别索引。

### 4.3. 使用 LabelMe 进行图像标注

1.  启动 LabelMe。您可以直接在命令行中输入 `labelme` 打开图形界面，或者指定图像目录和输出目录：
    ```bash
    cd yolo_project  # 进入项目根目录
    labelme data/raw_images/ --output data/annotations/ --labels data/classes.txt
    ```
    *   `data/raw_images/`: 包含待标注图像的文件夹。
    *   `--output data/annotations/`: 指定 LabelMe 生成的 JSON 标注文件的保存位置。
    *   `--labels data/classes.txt`: 加载预定义的类别列表，方便标注。

2.  在 LabelMe 界面中：
    *   点击 "Open Dir" 打开 `data/raw_images/` 目录。
    *   选择标注工具（对于目标检测，通常使用 "Create Rectangle"）。
    *   在图像上框出目标对象，并从右侧列表中选择或输入正确的类别标签。
    *   完成一张图像的标注后，点击 "Save" (Ctrl+S)，JSON 文件将保存在 `data/annotations/` 目录中，文件名与图像名对应。
    *   使用 "Next Image" (D) 和 "Prev Image" (A) 切换图像。

### 4.4. 将 LabelMe 标注转换为 YOLO 格式

LabelMe 生成的是 JSON 格式的标注文件，而 YOLO 需要特定格式的 `.txt` 标签文件（每个图像对应一个 `.txt` 文件，每行包含 `class_id x_center y_center width height`，均为归一化值）。

创建一个 Python 脚本 `scripts/labelme2yolo.py` 来执行此转换。

**`scripts/labelme2yolo.py` 示例代码：**
```python
import os
import json
import glob
import shutil
from PIL import Image

def labelme_to_yolo(labelme_json_dir, yolo_output_dir, classes_file_path):
    """
    将 LabelMe 标注的 JSON 文件转换为 YOLO 格式的 .txt 文件。
    同时将原始图像复制到 YOLO 数据集的 images 目录中。

    参数:
    labelme_json_dir (str): 存放 LabelMe JSON 标注文件的目录。
    yolo_output_dir (str): 存放转换后的 YOLO 格式数据 (images 和 labels 子目录) 的根目录。
    classes_file_path (str): 存放类别列表的 .txt 文件路径。
    """

    # 读取类别文件
    try:
        with open(classes_file_path, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        if not classes:
            print(f"错误: 类别文件 '{classes_file_path}' 为空或格式不正确。")
            return
    except FileNotFoundError:
        print(f"错误: 类别文件 '{classes_file_path}' 未找到。")
        return

    # 确保 YOLO 输出目录存在
    yolo_images_dir = os.path.join(yolo_output_dir, "images_temp") # 临时存放所有转换后的图片和标签
    yolo_labels_dir = os.path.join(yolo_output_dir, "labels_temp")
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # 处理每个 JSON 文件
    json_files = glob.glob(os.path.join(labelme_json_dir, "*.json"))
    if not json_files:
        print(f"警告: 在目录 '{labelme_json_dir}' 中未找到 JSON 文件。")
        return

    for json_file_path in json_files:
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"错误: 无法读取或解析 JSON 文件 '{json_file_path}': {e}")
            continue

        # 获取图像路径和尺寸
        image_path_in_json = data.get('imagePath')
        if not image_path_in_json:
            print(f"警告: JSON 文件 '{json_file_path}' 中缺少 'imagePath' 字段。跳过此文件。")
            continue

        # 构建原始图像的实际路径
        # 假设 JSON 文件中的 imagePath 是相对于 JSON 文件本身的相对路径，或者是一个基本文件名
        if os.path.isabs(image_path_in_json):
            original_image_path = image_path_in_json
        else:
            # 尝试基于 JSON 文件目录或原始图像目录 (raw_images) 寻找
            potential_path1 = os.path.join(os.path.dirname(json_file_path), image_path_in_json)
            # 假设 raw_images 目录与 annotations 目录同级
            potential_path2 = os.path.join(os.path.dirname(labelme_json_dir), "raw_images", os.path.basename(image_path_in_json))

            if os.path.exists(potential_path1):
                original_image_path = potential_path1
            elif os.path.exists(potential_path2):
                original_image_path = potential_path2
            else:
                print(f"警告: 无法找到图像 '{image_path_in_json}' (来自 JSON '{json_file_path}')。尝试路径: {potential_path1}, {potential_path2}。跳过此文件。")
                continue

        if not os.path.exists(original_image_path):
            print(f"警告: 原始图像文件 '{original_image_path}' 不存在。跳过 JSON 文件 '{json_file_path}'。")
            continue

        try:
            img = Image.open(original_image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"错误: 无法打开或读取图像 '{original_image_path}': {e}。跳过 JSON 文件 '{json_file_path}'。")
            continue

        # 创建 YOLO 标签文件名 (与图像文件名一致，扩展名为 .txt)
        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
        yolo_label_file_path = os.path.join(yolo_labels_dir, f"{base_name}.txt")

        # 复制图像到 YOLO images 目录 (保留原始扩展名，或统一为 .jpg)
        # 为了简单起见，这里统一保存为 .jpg
        new_image_filename = f"{base_name}.jpg"
        yolo_image_file_path = os.path.join(yolo_images_dir, new_image_filename)
        try:
            if original_image_path.lower().endswith(('.png', '.jpeg', '.bmp', '.gif', '.tiff')):
                 img.convert('RGB').save(yolo_image_file_path, "JPEG")
            else:
                shutil.copy(original_image_path, yolo_image_file_path)
        except Exception as e:
            print(f"错误: 复制或转换图像 '{original_image_path}' 到 '{yolo_image_file_path}' 失败: {e}")
            continue

        with open(yolo_label_file_path, 'w') as f_yolo:
            for shape in data.get('shapes', []):
                label = shape.get('label')
                if not label:
                    print(f"警告: 在 JSON '{json_file_path}' 的一个标注中缺少 'label'。跳过此标注。")
                    continue

                if label not in classes:
                    print(f"警告: 标签 '{label}' (来自 JSON '{json_file_path}') 不在预定义的类别列表 '{classes_file_path}' 中。跳过此标注。")
                    continue

                class_id = classes.index(label)

                # 仅处理矩形框 (rectangle)
                if shape.get('shape_type') == 'rectangle':
                    points = shape.get('points')
                    if not points or len(points) != 2:
                        print(f"警告: 矩形标注的点不正确 (JSON '{json_file_path}', 标签 '{label}')。跳过此标注。")
                        continue

                    # LabelMe 矩形框的两个对角点 (x1, y1) 和 (x2, y2)
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x1 = min(x_coords)
                    y1 = min(y_coords)
                    x2 = max(x_coords)
                    y2 = max(y_coords)

                    # 转换为 YOLO 格式 (中心点 x, 中心点 y, 宽度 w, 高度 h - 均为归一化值)
                    bb_x_center = (x1 + x2) / 2.0 / img_width
                    bb_y_center = (y1 + y2) / 2.0 / img_height
                    bb_width = (x2 - x1) / img_width
                    bb_height = (y2 - y1) / img_height

                    # 写入 YOLO 格式文件
                    f_yolo.write(f"{class_id} {bb_x_center:.6f} {bb_y_center:.6f} {bb_width:.6f} {bb_height:.6f}\n")
                else:
                    print(f"提示: 跳过非矩形标注 (类型: '{shape.get('shape_type')}', 标签: '{label}') 在文件 '{json_file_path}'。YOLO 目标检测通常使用矩形框。")

    print(f"转换完成。YOLO 格式的图像和标签已保存到 '{yolo_images_dir}' 和 '{yolo_labels_dir}'。")
    print(f"请检查输出，并使用 'split_dataset.py' 脚本来划分训练/验证/测试集。")

if __name__ == '__main__':
    # --- 配置路径 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # scripts 目录的上级目录 (yolo_project)

    labelme_json_input_dir = os.path.join(project_root, "data", "annotations")
    yolo_dataset_output_dir = os.path.join(project_root, "data", "yolo_dataset") # 转换后的数据将放在此目录的 images_temp 和 labels_temp
    classes_txt_file = os.path.join(project_root, "data", "classes.txt")
    # --- 执行转换 ---
    if not os.path.exists(labelme_json_input_dir):
        print(f"错误: LabelMe JSON 输入目录 '{labelme_json_input_dir}' 不存在。请先进行标注。")
    elif not os.path.exists(classes_txt_file):
        print(f"错误: 类别文件 '{classes_txt_file}' 不存在。请先创建。")
    else:
        labelme_to_yolo(labelme_json_input_dir, yolo_dataset_output_dir, classes_txt_file)

```

**运行转换脚本：**
确保您已进入 `yolo_project` 根目录。
```bash
python scripts/labelme2yolo.py
```
执行后，`data/yolo_dataset/images_temp/` 和 `data/yolo_dataset/labels_temp/` 目录下将分别存放转换后的图像和对应的 YOLO 格式标签文件。

### 4.5. 划分数据集 (训练集、验证集、测试集)

转换完成后，通常需要将数据集划分为训练集 (train)、验证集 (val) 和可选的测试集 (test)。

创建一个 Python 脚本 `scripts/split_dataset.py` 来执行此操作。

**`scripts/split_dataset.py` 示例代码：**
```python
import os
import glob
import random
import shutil

def split_dataset(base_dir, images_input_dir, labels_input_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将数据集划分为训练集、验证集和测试集。
    图像和标签文件将从 images_input_dir 和 labels_input_dir 移动到
    base_dir 下的 train/val/test 子目录中。

    参数:
    base_dir (str): YOLO 数据集的根目录 (例如 'data/yolo_dataset')。
                    划分后的 train/val/test 子目录将创建在此目录下。
    images_input_dir (str): 存放所有待划分图像的目录。
    labels_input_dir (str): 存放所有待划分标签的目录。
    train_ratio (float): 训练集比例。
    val_ratio (float): 验证集比例。
    test_ratio (float): 测试集比例。
    """

    # 确保比例加起来约等于 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        print("错误: 训练、验证和测试集的比例之和必须为 1.0。")
        return

    # 创建目标目录结构
    sets = ['train', 'val', 'test']
    for s in sets:
        os.makedirs(os.path.join(base_dir, "images", s), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", s), exist_ok=True)

    # 获取所有图像文件 (假设图像和标签文件名一一对应，只是扩展名不同)
    all_image_files = sorted(glob.glob(os.path.join(images_input_dir, "*.*"))) # 支持多种图像格式

    if not all_image_files:
        print(f"错误: 在目录 '{images_input_dir}' 中未找到图像文件。")
        return

    # 随机打乱
    random.shuffle(all_image_files)

    # 计算划分数量
    num_images = len(all_image_files)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    # num_test = num_images - num_train - num_val # 剩余的作为测试集

    # 划分图像列表
    train_images = all_image_files[:num_train]
    val_images = all_image_files[num_train : num_train + num_val]
    test_images = all_image_files[num_train + num_val:]

    datasets_map = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    def move_files(image_list, target_set_name):
        if not image_list and target_set_name in ['train', 'val'] and ( (target_set_name == 'train' and train_ratio > 0) or (target_set_name == 'val' and val_ratio > 0) ):
            print(f"警告: {target_set_name} 集为空，但其比例大于0。请检查图像数量和划分比例。")
        elif not image_list and target_set_name == 'test' and test_ratio > 0 :
             print(f"警告: {target_set_name} 集为空，但其比例大于0。请检查图像数量和划分比例。")


        for img_path in image_list:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_file_name = f"{base_name}.txt"
            original_label_path = os.path.join(labels_input_dir, label_file_name)

            # 目标路径
            dst_img_dir = os.path.join(base_dir, "images", target_set_name)
            dst_label_dir = os.path.join(base_dir, "labels", target_set_name)

            dst_img_path = os.path.join(dst_img_dir, os.path.basename(img_path))
            dst_label_path = os.path.join(dst_label_dir, label_file_name)

            # 移动图像文件
            shutil.move(img_path, dst_img_path)

            # 移动对应的标签文件 (如果存在)
            if os.path.exists(original_label_path):
                shutil.move(original_label_path, dst_label_path)
            else:
                print(f"警告: 未找到图像 '{os.path.basename(img_path)}' 对应的标签文件 '{label_file_name}'。")
        print(f"已将 {len(image_list)} 个文件移动到 {target_set_name} 集。")


    # 移动文件到各自的目录
    for set_name, image_paths in datasets_map.items():
        if (set_name == 'test' and test_ratio == 0) or not image_paths: # 如果测试集比例为0或文件列表为空，则跳过
            if test_ratio == 0 and set_name == 'test':
                 print(f"提示: 测试集比例为0，跳过创建测试集。")
            continue
        move_files(image_paths, set_name)

    # 清理临时的 images_temp 和 labels_temp 目录 (如果它们为空)
    if not os.listdir(images_input_dir):
        os.rmdir(images_input_dir)
        print(f"已清理空目录: {images_input_dir}")
    else:
        print(f"警告: 目录 '{images_input_dir}' 在划分后非空，请检查是否有未处理文件。")

    if not os.listdir(labels_input_dir):
        os.rmdir(labels_input_dir)
        print(f"已清理空目录: {labels_input_dir}")
    else:
        print(f"警告: 目录 '{labels_input_dir}' 在划分后非空，请检查是否有未处理文件。")

    print("数据集划分完成。")
    create_data_yaml(base_dir, os.path.join(os.path.dirname(base_dir), "classes.txt"))


def create_data_yaml(yolo_dataset_dir, classes_file_path):
    """为 YOLO 创建 data.yaml 配置文件"""
    try:
        with open(classes_file_path, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"错误: 类别文件 '{classes_file_path}' 未找到，无法创建 data.yaml。")
        return

    data_yaml_content = {
        'path': os.path.abspath(yolo_dataset_dir), # 数据集根目录的绝对路径
        'train': os.path.join("images", "train"),  #相对于 path 的训练集图像目录
        'val': os.path.join("images", "val"),      # 相对于 path 的验证集图像目录
    }
    if os.path.exists(os.path.join(yolo_dataset_dir, "images", "test")) and os.listdir(os.path.join(yolo_dataset_dir, "images", "test")):
        data_yaml_content['test'] = os.path.join("images", "test") # 相对于 path 的测试集图像目录 (可选)

    data_yaml_content.update({
        'nc': len(classes),
        'names': classes
    })

    yaml_path = os.path.join(yolo_dataset_dir, "data.yaml")
    try:
        import yaml # 需要 PyYAML 包
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=None)
        print(f"YOLO 数据集配置文件 'data.yaml' 已创建于: {yaml_path}")
    except ImportError:
        print("警告: PyYAML 未安装。无法自动创建 'data.yaml'。请手动创建或安装 PyYAML (`pip install PyYAML`) 后重新运行。")
        print("data.yaml 内容应如下所示:")
        print(f"path: {data_yaml_content['path']}")
        print(f"train: {data_yaml_content['train']}")
        print(f"val: {data_yaml_content['val']}")
        if 'test' in data_yaml_content:
            print(f"test: {data_yaml_content['test']}")
        print(f"nc: {data_yaml_content['nc']}")
        print(f"names: {data_yaml_content['names']}")
    except Exception as e:
        print(f"创建 'data.yaml' 文件时出错: {e}")


if __name__ == '__main__':
    # --- 配置路径 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # scripts 目录的上级目录 (yolo_project)

    yolo_data_root = os.path.join(project_root, "data", "yolo_dataset")
    # 输入目录是 labelme2yolo.py 脚本的输出
    source_images_dir = os.path.join(yolo_data_root, "images_temp")
    source_labels_dir = os.path.join(yolo_data_root, "labels_temp")

    # --- 配置划分比例 ---
    # 确保 train_ratio + val_ratio + test_ratio = 1.0
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1 # 如果不需要测试集，可以设为 0，则 VAL_RATIO 相应调整为 0.2 (或 TRAIN_RATIO 为 0.9)

    if not os.path.exists(source_images_dir) or not os.listdir(source_images_dir):
        print(f"错误: 原始图像目录 '{source_images_dir}' 不存在或为空。请先运行 'labelme2yolo.py' 进行转换。")
    elif not os.path.exists(source_labels_dir) or not os.listdir(source_labels_dir):
         print(f"错误: 原始标签目录 '{source_labels_dir}' 不存在或为空。请先运行 'labelme2yolo.py' 进行转换。")
    else:
        split_dataset(yolo_data_root, source_images_dir, source_labels_dir, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

```

**运行划分脚本：**
确保您已进入 `yolo_project` 根目录。
```bash
python scripts/split_dataset.py
```
执行后，`data/yolo_dataset/images_temp/` 和 `data/yolo_dataset/labels_temp/` 中的文件将被移动到 `data/yolo_dataset/` 下相应的 `images/train`, `images/val`, `images/test`, `labels/train`, `labels/val`, `labels/test` 目录中。同时，会在 `data/yolo_dataset/` 目录下生成一个 `data.yaml` 文件，供 YOLO 训练时使用。

### 4.6. 使用转换后的数据集训练 YOLO 模型

数据集准备完毕后，您可以使用它来训练您的 YOLO 模型（例如 YOLOv8）。

示例训练命令 (YOLOv8):
```bash
# 确保您的 YOLOv8 环境已激活
# yolo task=detect mode=train model=yolov8s.pt data=path/to/yolo_project/data/yolo_dataset/data.yaml epochs=100 imgsz=640 batch=16 ...
yolo task=detect mode=train model=yolov8s.pt data=data/yolo_dataset/data.yaml epochs=100 imgsz=640
```
请将 `yolov8s.pt` 替换为您选择的预训练模型，并根据需要调整 `epochs`, `imgsz`, `batch` 等超参数。`data` 参数应指向您生成的 `data.yaml` 文件。

## 5. LabelMe 标注技巧与建议

1.  **预定义标签**：启动 LabelMe 时使用 `--labels data/classes.txt` 参数加载标签列表，可以避免手动输入类别名称，减少拼写错误，确保标签一致性。
2.  **善用快捷键**：
    *   `Ctrl+S`: 保存当前标注。
    *   `W` 或 "Create Rectangle" / "Create Polygon": 创建标注。
    *   `Ctrl+Z`: 撤销上一步操作。
    *   `Del` 或 `Backspace` (选中标注后): 删除选中的标注。
    *   `A` / `D`: 切换到上一张/下一张图像。
    *   `Space` (拖动图像时): 激活拖动模式。
3.  **批量处理**：直接打开包含多个图像的目录 (`labelme data/raw_images/`) 进行标注，比单张打开效率更高。
4.  **保持标注一致性**：
    *   **边界框紧密性**：确保标注框尽可能紧密地包围目标对象，不多余也不遗漏。
    *   **遮挡处理**：对于部分被遮挡的对象，通常标注其可见部分。根据项目需求决定是否标注严重遮挡的对象。
    *   **标签准确性**：确保为每个对象选择正确的类别标签。
    *   **命名规范**：在 `classes.txt` 中使用一致的命名（例如，全小写，单数形式）。
5.  **检查标注质量**：在转换格式之前，花时间回顾和检查已完成的标注，确保准确性和一致性。错误的标注会严重影响模型性能。

## 6. 从 LabelMe 到 YOLO 的注意事项

1.  **坐标系转换**：
    *   LabelMe 保存的是每个标注点在图像中的绝对像素坐标。
    *   YOLO 需要的是归一化坐标：目标边界框的中心点 `(x_center, y_center)` 以及框的 `width` 和 `height`，所有这些值都相对于图像的总宽度和总高度进行归一化 (范围在 0 到 1 之间)。转换脚本 (`labelme2yolo.py`) 会处理这个转换。
2.  **标注类型**：
    *   对于标准的 YOLO 目标检测任务，主要使用 LabelMe 的**矩形 (rectangle)** 标注工具。
    *   如果使用了多边形 (polygon) 或其他形状进行分割任务的标注，转换到 YOLO 目标检测格式时，通常会取这些形状的最小外接矩形。提供的 `labelme2yolo.py` 脚本示例仅处理矩形标注。
3.  **类别索引**：
    *   YOLO 标签文件中的类别 ID 是从 0 开始的整数索引。
    *   `classes.txt` 文件中的类别顺序决定了这个索引。例如，`classes.txt` 中第一行的类别对应 ID 0，第二行对应 ID 1，以此类推。转换脚本会根据 `classes.txt` 来分配正确的类别 ID。

---