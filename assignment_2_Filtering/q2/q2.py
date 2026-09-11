import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import correlate

def compute_ncc_channel(scene, template):
    """
    Computes Normalized Cross-Correlation for a single channel.
    Follows strictly:
    1) F' = Fpatch / RMS(Fpatch)
    2) H' = H / RMS(H)
    3) NCC = sum(F' * H')
    Using scipy.signal.correlate with FFT for fast computation.
    """
    N = template.size
    H_sum_sq = np.sum(template**2)
    if H_sum_sq == 0:
        # Avoid division by zero if template is completely black
        return np.zeros((scene.shape[0] - template.shape[0] + 1, 
                         scene.shape[1] - template.shape[1] + 1))
    
    H_rms = np.sqrt(H_sum_sq / N)
    
    cross_corr = correlate(scene, template, mode='valid', method='fft')
    
    window = np.ones_like(template)
    sum_F_sq = correlate(scene**2, window, mode='valid', method='fft')
    
    # Fix minor numerical stability issues (negative values from FFT roundoff)
    sum_F_sq[sum_F_sq < 0] = 0
    
    F_rms = np.sqrt(sum_F_sq / N)
    F_rms[F_rms == 0] = 1e-10  # Prevent division by zero
    
    ncc = cross_corr / (F_rms * H_rms)
    return ncc

def compute_masked_ncc_channel(scene, template, mask):
    """
    Computes NCC while ignoring the pixels outside the mask (i.e. background).
    """
    N_valid = np.sum(mask)
    if N_valid == 0:
        return np.zeros((scene.shape[0] - template.shape[0] + 1, 
                         scene.shape[1] - template.shape[1] + 1))
        
    H_masked = template * mask
    
    H_sum_sq = np.sum(H_masked**2)
    H_rms = np.sqrt(H_sum_sq / N_valid)
    
    cross_corr = correlate(scene, H_masked, mode='valid', method='fft')
    
    sum_F_sq = correlate(scene**2, mask, mode='valid', method='fft')
    sum_F_sq[sum_F_sq < 0] = 0
    
    F_rms = np.sqrt(sum_F_sq / N_valid)
    F_rms[F_rms == 0] = 1e-10
    
    ncc = cross_corr / (F_rms * H_rms)
    return ncc

if __name__ == '__main__':
    
    os.makedirs("./img", exist_ok=True)
    
    scene_path = "../data/templateMatch/parking.png"
    template_path = "../data/templateMatch/templateNoPark.png"
    
    with Image.open(scene_path) as img_scene, Image.open(template_path) as img_temp:
        print(np.array(img_scene).shape, np.array(img_temp).shape)
        # Reduce scene image size by a factor of 5
        new_w = img_scene.width // 5
        new_h = img_scene.height // 5
        scene_b = img_scene.resize((new_w, new_h), Image.Resampling.LANCZOS)
        scene_arr_b = np.array(scene_b, dtype=np.float64)
        
        sizes_b = [(41, 41), (51, 51), (61, 61)]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        channels = ['Red', 'Green', 'Blue']
        
        for i, (size_w, size_h) in enumerate(sizes_b):
            temp_resized = img_temp.resize((size_w, size_h), Image.Resampling.LANCZOS)
            temp_arr = np.array(temp_resized, dtype=np.float64)
            
            for c in range(3):
                ncc_result = compute_ncc_channel(scene_arr_b[:,:,c], temp_arr[:,:,c])
                
                # Plot
                ax = axes[i, c]
                im = ax.imshow(ncc_result, cmap='viridis')
                ax.set_title(f'Size {size_w}x{size_h} | {channels[c]} Ch')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("Part (b) - NCC Images for Reduced Scene Image", fontsize=16)
        plt.tight_layout()
        plt.savefig("./img/part_b_ncc_results.png")
        plt.close()

    print("Running Part (d): Masked NCC with Original Scene")
    with Image.open(scene_path) as img_scene, Image.open(template_path) as img_temp:
        # For part d, use original scale scene image
        scene_arr_d = np.array(img_scene, dtype=np.float64)
        
        sizes_d = [(201, 201), (251, 251), (301, 301)]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        for i, (size_w, size_h) in enumerate(sizes_d):
            # Resize template
            temp_resized = img_temp.resize((size_w, size_h), Image.Resampling.LANCZOS)
            temp_arr = np.array(temp_resized, dtype=np.float64)
            
            # The background of the template is white, so we exclude RGB values close to (255, 255, 255)
            # using a strict threshold for isolating the white background.
            white_thresh = 245
            mask = ~((temp_arr[:,:,0] > white_thresh) & 
                     (temp_arr[:,:,1] > white_thresh) & 
                     (temp_arr[:,:,2] > white_thresh))
            mask = mask.astype(np.float64)
            
            for c in range(3):
                ncc_result = compute_masked_ncc_channel(scene_arr_d[:,:,c], temp_arr[:,:,c], mask)
                
                ax = axes[i, c]
                im = ax.imshow(ncc_result, cmap='viridis')
                ax.set_title(f'Masked Size {size_w}x{size_h} | {channels[c]} Ch')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("./img/part_d_masked_ncc_results.png")
        plt.close()