import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from PIL import Image
import os

# ==========================================
# (a) Canny Edge Detection Implementation
# ==========================================
def canny_edge_detection(image_array, sigma=1.5, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    """
    Applies Canny Edge Detection algorithm to an image array.
    """
    # Convert to grayscale if it is an RGB image
    # Standard luminance conversion
    img_gray = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

    # 1) Gaussian smoothing to reduce noise
    smoothed = gaussian_filter(img_gray, sigma=sigma)

    # 2) Gradient-magnitude G and orientation computation at each pixel
    Gx = sobel(smoothed, axis=1)
    Gy = sobel(smoothed, axis=0)
    
    G = np.hypot(Gx, Gy) # G = sqrt(Gx^2 + Gy^2)
    G = (G / G.max()) * 255.0 # Normalize to 0-255
    
    theta = np.arctan2(Gy, Gx)

    # 3) Edge thinning by non-maximum suppression
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.float32)
    
    # Convert angles from radians to degrees
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    # Quantize the unit vector u(p) to the 4 primary directions (0, 45, 90, 135)
    # to compare against p+u(p) and p-u(p)
    for i in range(1, M-1):
        for j in range(1, N-1):
            ang = angle[i, j]
            mag = G[i, j]
            q = 255
            r = 255

            # Angle 0 degrees (Horizontal edge)
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            # Angle 45 degrees
            elif (22.5 <= ang < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            # Angle 90 degrees (Vertical edge)
            elif (67.5 <= ang < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            # Angle 135 degrees
            elif (112.5 <= ang < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]

            # If pixel gradient is greater than neighbors along the gradient direction, keep it
            if (mag >= q) and (mag >= r):
                Z[i, j] = mag
            else:
                Z[i, j] = 0

    # 4) Edge tracing via hysteresis thresholding
    high_thresh = Z.max() * high_threshold_ratio
    low_thresh = high_thresh * low_threshold_ratio

    res = np.zeros((M, N), dtype=np.int32)
    
    WEAK = np.int32(50)
    STRONG = np.int32(255)

    # First, select candidate pixels where ||v(p)||^2 > thresholdHigh
    strong_i, strong_j = np.where(Z >= high_thresh)
    # Select weak edges
    weak_i, weak_j = np.where((Z <= high_thresh) & (Z >= low_thresh))

    res[strong_i, strong_j] = STRONG
    res[weak_i, weak_j] = WEAK

    # Find all connected paths starting from strong edge pixels
    changed = True
    while changed:
        changed = False
        for i in range(1, M-1):
            for j in range(1, N-1):
                if res[i, j] == WEAK:
                    # Check 8-connected neighborhood for a strong edge pixel
                    if ((res[i+1, j-1] == STRONG) or (res[i+1, j] == STRONG) or (res[i+1, j+1] == STRONG)
                        or (res[i, j-1] == STRONG) or (res[i, j+1] == STRONG)
                        or (res[i-1, j-1] == STRONG) or (res[i-1, j] == STRONG) or (res[i-1, j+1] == STRONG)):
                        res[i, j] = STRONG
                        changed = True

    # Suppress remaining weak edges that were not connected to strong edges
    res[res == WEAK] = 0
    
    # Return binary boolean array
    return res == STRONG

# ==========================================
# (b) Display Original and Edge Detected Versions
# ==========================================
if __name__ == '__main__':

    # Define parameters (Tune these free parameters for reasonable outputs)
    sigma = 1.5               # Gaussian blur radius
    low_thresh_ratio = 0.05   # Hysteresis lower threshold ratio
    high_thresh_ratio = 0.15  # Hysteresis higher threshold ratio

    image_files = [
        "../data/edge/butterfly.png",
        "../data/edge/paithaniEdge.png",
        "../data/edge/rangoli.png"
    ]
    
    # Ensure save directory exists
    os.makedirs("./img", exist_ok=True)

    for path in image_files:
        with Image.open(path) as img:

            # 1. Extract just "butterfly.png" (removes "../data/edge/")
            filename = os.path.basename(path)
            # 2. Split "butterfly.png" into "butterfly" and ".png"
            name_only, ext = os.path.splitext(filename)

            # To confirm all the images are RGB (hence 3-D) and none are gray scale
            print(path)
            print(np.array(img).shape)

            # img_rgb = img.convert('RGB')
            img_rgb = img
            
            # Convert the Pillow object into a NumPy array
            img_array = np.array(img_rgb)
            
            # Apply the canny edge detection function
            edges_binary = canny_edge_detection(
                img_array, 
                sigma=sigma, 
                low_threshold_ratio=low_thresh_ratio, 
                high_threshold_ratio=high_thresh_ratio
            )
            
            # Create the overlay image: detected edges drawn in black on original
            img_overlay = img_array.copy()
            img_overlay[edges_binary] = [0, 0, 0] # Set edge pixels to black
            
            # Save images
            Image.fromarray((edges_binary * 255).astype(np.uint8)).save(f"./img/{name_only}_binary_edges{ext}")
            Image.fromarray(img_overlay.astype(np.uint8)).save(f"./img/{name_only}_edges_overlay{ext}")
            
            # Set up the plot with 1 row and 3 columns
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # 1. Plot Original Image
            axes[0].imshow(img_array)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 2. Plot Output Image (Binary showing detected edges)
            axes[1].imshow(edges_binary, cmap='gray')
            axes[1].set_title('Detected Edges (Binary)')
            axes[1].axis('off')
            
            # 3. Plot Detected Edges drawn in black on the input image
            axes[2].imshow(img_overlay)
            axes[2].set_title('Edges Overlay (Black)')
            axes[2].axis('off')
            
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"./img/q3_{name_only}{ext}")
            plt.close()
                