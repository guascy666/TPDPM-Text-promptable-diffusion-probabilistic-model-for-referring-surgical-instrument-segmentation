import shutil
import os

import shutil

def copy_folder(src_folder, dst_folder):
    # try:
    #     # 如果目标文件夹不存在，会直接创建并复制
    #     shutil.copytree(src_folder, dst_folder)
    #     print(f"文件夹已成功从 {src_folder} 复制到 {dst_folder}")
    # except FileExistsError:
    #     print(f"错误：目标文件夹 {dst_folder} 已存在")
    # except Exception as e:
    #     print(f"复制失败：{e}")
    shutil.copytree(src_folder, dst_folder,dirs_exist_ok=True)

# 示例用法
new_data='Endovis-RS17all'
os.makedirs(new_data,exist_ok=True)
os.makedirs(os.path.join(new_data,'Annotations'),exist_ok=True)
os.makedirs(os.path.join(new_data,'JPEGImages'),exist_ok=True)
old_dir1 = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS17/train/Annotations'
old_dir2 = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS17/train/JPEGImages'
# for i in os.listdir(old_dir1):
#     print(i)
copy_folder(old_dir1,os.path.join(new_data,'Annotations'))
print("loading")
copy_folder(old_dir2,(os.path.join(new_data,'JPEGImages')))
print("loading")
old_dir1 = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS17/valid/Annotations'
old_dir2 = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS17/valid/JPEGImages'

copy_folder(old_dir1,os.path.join(new_data,'Annotations'))
print("loading")
copy_folder(old_dir2,(os.path.join(new_data,'JPEGImages')))
print("loading")

