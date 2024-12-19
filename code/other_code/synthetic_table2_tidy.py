import os
import shutil

# 設定原始資料夾路徑
folder_path = "C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_2/ERRNet-master/test_data/testdata_reflection_synthetic_table2/blended"

# 創建目標資料夾
output_folders = ['input_images', 'label1_images', 'label2_images']
for folder in output_folders:
    os.makedirs(os.path.join(folder_path, folder), exist_ok=True)

# 取得所有檔案
files = os.listdir(folder_path)

# 尋找每一組檔案並移動與重新命名
for file in files:
    if file.endswith('-input.png'):
        prefix = file.replace('-input.png', '')
        input_path = os.path.join(folder_path, file)
        label1_path = os.path.join(folder_path, prefix + '-label1.png')
        label2_path = os.path.join(folder_path, prefix + '-label2.png')

        # 移動檔案到對應的資料夾並重新命名
        shutil.move(input_path, os.path.join(folder_path, 'input_images', prefix + '.png'))
        shutil.move(label1_path, os.path.join(folder_path, 'label1_images', prefix + '.png'))
        shutil.move(label2_path, os.path.join(folder_path, 'label2_images', prefix + '.png'))

print("檔案已經移動並重新命名完成！")

