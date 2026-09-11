import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

def apply_unsharp_mask(image, sigma, scale_s):
    """
    Applies unsharp masking to sharpen an image based on the exact slide algorithm.
    
    Parameters:
    - image: 3D numpy array (input image).
    - sigma: Standard deviation for Gaussian kernel (blur amount).
    - scale_s: Scaling factor 's' for the unsharp mask.
    """
    image = image.astype(float)
    
    blurred_image = gaussian_filter(image, sigma=(sigma, sigma, 0))
    
    unsharp_mask = image - blurred_image
    
    sharpened_image = image + (scale_s * unsharp_mask)
    
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    sharpened_clipped = np.clip(sharpened_image, min_intensity, max_intensity)
    
    return sharpened_clipped

if __name__ == '__main__':

    sigma = 2.0       # Blur radius for the unsharp mask
    scale_1 = 1.5     # Level 1 sharpening
    scale_2 = 4.0     # Level 2 sharpening (stronger, clearly perceivable)

    image_files = [
        "../data/sharpen/moon.png",
        "../data/sharpen/peacock.png",
        "../data/sharpen/tiger.png"
    ]

    os.makedirs("./img", exist_ok=True)

    for path in image_files:
        with Image.open(path) as img:

            filename = os.path.basename(path)
            name_only, ext = os.path.splitext(filename)

            # To confirm all the images are RGB (hence 3-D) and none are gray scale
            # print(path)
            # print(np.array(img).shape)

            # img_rgb = img.convert('RGB')
            img_rgb = img
                
            img_array = np.array(img_rgb)
            
            sharpened_1 = apply_unsharp_mask(img_array, sigma, scale_1)
            sharpened_2 = apply_unsharp_mask(img_array, sigma, scale_2)
            # Save images
            Image.fromarray(sharpened_1.astype(np.uint8)).save(f"./img/{name_only}_scale1{ext}")
            Image.fromarray(sharpened_2.astype(np.uint8)).save(f"./img/{name_only}_scale2{ext}")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            vmin = np.min(img_array)
            vmax = np.max(img_array)
            
            im0 = axes[0].imshow(np.round(img_array).astype(np.uint8))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            im1 = axes[1].imshow(np.round(sharpened_1).astype(np.uint8))
            axes[1].set_title(f'Sharpened (Scale s = {scale_1})')
            axes[1].axis('off')
            
            im2 = axes[2].imshow(np.round(sharpened_2).astype(np.uint8))
            axes[2].set_title(f'Sharpened (Scale s = {scale_2})')
            axes[2].axis('off')
            
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"./img/q1_{name_only}{ext}")
            plt.close()
    