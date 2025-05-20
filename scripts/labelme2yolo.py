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
                elif shape.get('shape_type') == 'polygon':
                    points = shape.get('points')
                    if not points or len(points) < 3: # 多边形至少需要3个点
                        print(f"警告: 多边形标注的点不正确 (JSON '{json_file_path}', 标签 '{label}')。跳过此标注。")
                        continue

                    # 计算多边形的外接矩形
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x1 = min(x_coords)
                    y1 = min(y_coords)
                    x2 = max(x_coords)
                    y2 = max(y_coords)

                    # 转换为 YOLO 格式
                    bb_x_center = (x1 + x2) / 2.0 / img_width
                    bb_y_center = (y1 + y2) / 2.0 / img_height
                    bb_width = (x2 - x1) / img_width
                    bb_height = (y2 - y1) / img_height

                    # 写入 YOLO 格式文件
                    f_yolo.write(f"{class_id} {bb_x_center:.6f} {bb_y_center:.6f} {bb_width:.6f} {bb_height:.6f}\n")
                else:
                    print(f"提示: 跳过非矩形或非多边形标注 (类型: '{shape.get('shape_type')}', 标签: '{label}') 在文件 '{json_file_path}'。")
  
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
