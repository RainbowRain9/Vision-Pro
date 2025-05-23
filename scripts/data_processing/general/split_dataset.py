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
