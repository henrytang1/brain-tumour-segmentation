# Brain-Tumour-Segmentation
This webapp can be used to perform segmentation on brain MRI images. It features a UNet model with added batch normalization layers for optimal performance. The model is trained on a publicly available Kaggle dataset (see credits below), and can be used for fast inferencing through the Flask webapp. The app will create a segmentation mask filling any tumours or masses in a FLAIR enhanced MRI image, and if no tumours are detected, it will create a blank (default) mask.

## Illustration
<img src="images/output.gif">

### Credits
Kaggle dataset: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
