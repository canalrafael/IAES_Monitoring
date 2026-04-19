import os
from PIL import Image

RESULTS_DIR = 'results/phase2_simplified'

def combine_images():
    img1_path = os.path.join(RESULTS_DIR, 'loss_curve.png')
    img2_path = os.path.join(RESULTS_DIR, 'learning_curve.png')
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Error: Could not find the original model's curve plots.")
        return

    # Open the original exported model's exact curves
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Combine them horizontally to save space
    dst_width = img1.width + img2.width
    dst_height = max(img1.height, img2.height)
    dst = Image.new('RGB', (dst_width, dst_height), color='white')
    
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))

    out_path = os.path.join(RESULTS_DIR, 'combined_learning_loss_curve.png')
    dst.save(out_path)
    print(f"Successfully created dual plot: {out_path}")

if __name__ == "__main__":
    combine_images()
