import os
import shutil

# 定義路徑
base_path = r"C:\Users\asd47\OneDrive - rui0828\master\CV\Final_Project\Code\paper_2\ERRNet-master\test_data\SIR2\SolidObjectDataset"

# 建立存放檔案的資料夾
output_folder_g = os.path.join(base_path, "output_g")
output_folder_m = os.path.join(base_path, "output_m")
output_folder_r = os.path.join(base_path, "output_r")

os.makedirs(output_folder_g, exist_ok=True)
os.makedirs(output_folder_m, exist_ok=True)
os.makedirs(output_folder_r, exist_ok=True)

# 遍歷第一層資料夾
for first_folder in os.listdir(base_path):
    first_folder_path = os.path.join(base_path, first_folder)
    if os.path.isdir(first_folder_path):
        # 遍歷第二層資料夾
        for second_folder in os.listdir(first_folder_path):
            second_folder_path = os.path.join(first_folder_path, second_folder)
            if os.path.isdir(second_folder_path):
                # 遍歷第三層資料夾
                for third_folder in os.listdir(second_folder_path):
                    third_folder_path = os.path.join(second_folder_path, third_folder)
                    if os.path.isdir(third_folder_path):
                        # 找到PNG檔案並處理
                        for file in os.listdir(third_folder_path):
                            if "g." in file:
                                shutil.copy(os.path.join(third_folder_path, file), os.path.join(output_folder_g, f"{first_folder}_{second_folder}_{third_folder}.png"))
                                # print(file,"g")
                            elif "m." in file:
                                shutil.copy(os.path.join(third_folder_path, file), os.path.join(output_folder_m, f"{first_folder}_{second_folder}_{third_folder}.png"))
                                # print(file,"m")
                            elif "r." in file:
                                shutil.copy(os.path.join(third_folder_path, file), os.path.join(output_folder_r, f"{first_folder}_{second_folder}_{third_folder}.png"))
                                # print(file,"r")

print("完成資料整理")
