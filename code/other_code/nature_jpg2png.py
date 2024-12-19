from PIL import Image
import os

def jpg_to_png(input_folder, output_folder):
    # 如果輸出資料夾不存在，創建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷資料夾中所有檔案
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        # 檢查是否為JPG檔案
        if filename.endswith(".jpg"):
            # 讀取JPG檔案
            with Image.open(filepath) as img:
                # 確保檔案副檔名為.png
                output_filepath = os.path.splitext(filename)[0] + ".png"
                output_filepath = os.path.join(output_folder, output_filepath)
                # 將圖片儲存為PNG
                img.save(output_filepath, "PNG")

# 指定要轉換的資料夾路徑
input_folder_path = "C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_2/ERRNet-master/test_data/Nature/blended/"
output_folder_path = "C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_2/ERRNet-master/test_data/Nature_PNG/blended/"

# 呼叫函式進行轉換
jpg_to_png(input_folder_path, output_folder_path)