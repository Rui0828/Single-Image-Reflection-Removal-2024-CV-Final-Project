import os

s = open("/mnt/c/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/training_output.txt").read()

os.system("nvidia-smi")
print(s[s.rfind("iter:"):])

