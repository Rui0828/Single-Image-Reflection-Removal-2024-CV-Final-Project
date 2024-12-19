from PIL import Image
import shutil

source_path = "C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/download/training_set/Pascal VOC dataset/VOCdevkit/VOC2012/JPEGImages"
f = open("C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/download/training_set/Pascal VOC dataset/VOC2012_224_train_png.txt", "r")
p_list = f.readlines()

output_dir = "./data/training_set/VOC_224/"

for i in range(len(p_list)):
    temp = p_list[i].replace("/mnt/data/","/").replace(".png\n", ".jpg").strip()
    source_image_path = source_path + temp
    destination_image_path = output_dir + str(i) + ".jpg"
    
    shutil.copyfile(source_image_path, destination_image_path)
    
    # Resize the image to 224x224
    try:
        img = Image.open(destination_image_path)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img.save(destination_image_path)
    except Exception as e:
        print(f"Error resizing image {source_image_path}: {e}")
