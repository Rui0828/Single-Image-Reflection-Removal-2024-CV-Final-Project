import cv2
import lpips
import numpy as np
from scipy.signal import convolve2d
import os

# PSNR
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# SSIM: https://cloud.tencent.com/developer/article/2218582
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.04, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=0.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def img_show(similarity, img1, img2, name1, name2):
    img1 = cv2.resize(img1, (520, 520))
    img2 = cv2.resize(img2, (520, 520))
    # 拼接兩張圖片
    imgs = np.hstack([img1, img2])
    path = "{0}".format('{0}VS{1}相似指数{2}%.jpg'.format(name1, name2, round(similarity, 2)))
    cv2.imencode('.jpg', imgs)[1].tofile(path)
    return path

# LPIPS
loss_fn = lpips.LPIPS(net='alex')
def calculate_lpips(image1, image2):
    tensor1 = lpips.im2tensor(image1)
    tensor2 = lpips.im2tensor(image2)
    lpips_value = loss_fn(tensor1, tensor2)
    return lpips_value.item()



# 計算所有指標
def calculate_all(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # 轉換為灰度圖像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 計算PSNR
    psnr = calculate_psnr(img1_gray, img2_gray)
    # print(f'PSNR: {psnr}')

    # 計算SSIM
    ssim = compute_ssim(img1_gray, img2_gray)
    # print(f'SSIM: {ssim}')

    # 計算LPIPS
    lpips_value = calculate_lpips(img1, img2)
    # print(f'LPIPS: {lpips_value}')
    
    return psnr, ssim, lpips_value


# 計算 synthetic_table2 (png)
# C: correct; E: estimate
print('Estimating synthetic_table2')
synthetic_table2_C_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_images/testdata_reflection_synthetic_table2/transmission_layer'
synthetic_table2_E_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_results/synthetic_table2'

synthetic_table2_eval_list = os.listdir(synthetic_table2_C_path)

synthetic_table2_PSNR = []
synthetic_table2__SSIM = []
synthetic_table2_LPIPS = []

for i in synthetic_table2_eval_list:
    synthetic_table2_C = os.path.join(synthetic_table2_C_path, i)
    synthetic_table2_E = os.path.join(synthetic_table2_E_path, i.replace('.png', '/t_output.png'))
    psnr, ssim, lpips_value = calculate_all(synthetic_table2_C, synthetic_table2_E)
    synthetic_table2_PSNR.append(psnr)
    synthetic_table2__SSIM.append(ssim)
    synthetic_table2_LPIPS.append(lpips_value)

print(f'PSNR: {np.mean(synthetic_table2_PSNR)}', f'SSIM: {np.mean(synthetic_table2__SSIM)}', f'LPIPS: {np.mean(synthetic_table2_LPIPS)}')
print('')

#計算 SolidObjectDataset (png)
# C: correct; E: estimate
print('Estimating SolidObjectDataset')
sod_C_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_images/SIR2/SolidObjectDataset/transmission_layer'
sod_E_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_results/SolidObjectDataset'

sod_eval_list = os.listdir(sod_C_path)

sod_PSNR = []
sod_SSIM = []
sod_LPIPS = []

for i in sod_eval_list:
    sod_C = os.path.join(sod_C_path, i)
    sod_E = os.path.join(sod_E_path, i.replace('.png', '/t_output.png'))
    psnr, ssim, lpips_value = calculate_all(sod_C, sod_E)
    sod_PSNR.append(psnr)
    sod_SSIM.append(ssim)
    sod_LPIPS.append(lpips_value)

print(f'PSNR: {np.mean(sod_PSNR)}', f'SSIM: {np.mean(sod_SSIM)}', f'LPIPS: {np.mean(sod_LPIPS)}')
print('')

#計算 Postcard (png)
# C: correct; E: estimate
print('Estimating Postcard')
postcard_C_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_images/SIR2/Postcard Dataset/transmission_layer'
postcard_E_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_results/Postcard'

postcard_eval_list = os.listdir(postcard_C_path)

postcard_PSNR = []
postcard_SSIM = []
postcard_LPIPS = []

for i in postcard_eval_list:
    postcard_C = os.path.join(postcard_C_path, i)
    postcard_E = os.path.join(postcard_E_path, i.replace('.png', '/t_output.png'))
    psnr, ssim, lpips_value = calculate_all(postcard_C, postcard_E)
    postcard_PSNR.append(psnr)
    postcard_SSIM.append(ssim)
    postcard_LPIPS.append(lpips_value)

print(f'PSNR: {np.mean(postcard_PSNR)}', f'SSIM: {np.mean(postcard_SSIM)}', f'LPIPS: {np.mean(postcard_LPIPS)}')
print('')

#計算 Wild (png)
# C: correct; E: estimate
print('Estimating Wild')
wild_C_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_images/SIR2/Wildscene/transmission_layer'
wild_E_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_results/Wildscene'

wild_eval_list = os.listdir(wild_C_path)

wild_PSNR = []
wild_SSIM = []
wild_LPIPS = []

for i in wild_eval_list:
    wild_C = os.path.join(wild_C_path, i)
    wild_E = os.path.join(wild_E_path, i.replace('.png', '/t_output.png'))
    psnr, ssim, lpips_value = calculate_all(wild_C, wild_E)
    wild_PSNR.append(psnr)
    wild_SSIM.append(ssim)
    wild_LPIPS.append(lpips_value)

print(f'PSNR: {np.mean(wild_PSNR)}', f'SSIM: {np.mean(wild_SSIM)}', f'LPIPS: {np.mean(wild_LPIPS)}')
print('')

#計算 Nature (jpg)
# C: correct; E: estimate
print('Estimating Nature')
nature_C_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_images/Nature/transmission_layer'
nature_E_path = 'C:/Users/asd47/OneDrive - rui0828/master/CV/Final_Project/Code/paper_1/test_results/Nature'

nature_eval_list = os.listdir(nature_C_path)

nature_PSNR = []
nature_SSIM = []
nature_LPIPS = []

for i in nature_eval_list:
    nature_C = os.path.join(nature_C_path, i)
    nature_E = os.path.join(nature_E_path, i.replace('.jpg', '/t_output.png'))
    psnr, ssim, lpips_value = calculate_all(nature_C, nature_E)
    nature_PSNR.append(psnr)
    nature_SSIM.append(ssim)
    nature_LPIPS.append(lpips_value)

print(f'PSNR: {np.mean(nature_PSNR)}', f'SSIM: {np.mean(nature_SSIM)}', f'LPIPS: {np.mean(nature_LPIPS)}')
print('')