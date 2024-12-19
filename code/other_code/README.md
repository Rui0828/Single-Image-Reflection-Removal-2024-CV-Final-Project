# Other Code - ReadMe

## Script Descriptions

### Dataset Preprocessing
- **`crop_VOC224.py`**  
  Crops images from the Pascal VOC dataset (provided IDs) to a size of 224×224.

- **`nature_jpg2png.py`**  
  Converts all images in the Nature Dataset to `.png` format, as there are mismatches between `.jpg` and `.png` file extensions that could affect implementation.

- **`Obj_tidy.py`**  
  Standardizes folder structures and file naming conventions in the SolidObjectDataset for easier usage.

- **`Postcard_tidy.py`**  
  Standardizes folder structures and file naming conventions in the Postcard Dataset for easier usage.

- **`Wile_tidy.py`**  
  Standardizes folder structures and file naming conventions in the Wildscene Dataset for easier usage.

- **`synthetic_table2_tidy.py`**  
  Standardizes folder structures and file naming conventions in the `testdata_reflection_synthetic_table2` dataset for easier usage.

### Evaluation Metrics
- **`eval_paper_1.py`**, **`eval_paper_3.py`**  
  Calculates PSNR, SSIM, and LPIPS metrics for evaluating reflection removal performance.
