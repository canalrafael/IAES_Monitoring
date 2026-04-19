import os
from PIL import Image

RESULTS_DIR = 'results/phase2_simplified'
n_values = [1, 3, 5, 7]

def aggregate_matrices():
    images = []
    print("Loading individual confusion matrices...")
    for n in n_values:
        path = os.path.join(RESULTS_DIR, f'cm_N{n}.png')
        if os.path.exists(path):
            images.append(Image.open(path))
            print(f" - Loaded {path}")
        else:
            print(f"Warning: Could not find {path}")

    if len(images) == 4:
        # Assuming all images were saved with similar dimensions by matplotlib
        w, h = images[0].width, images[0].height
        
        # Create a new white canvas for the 2x2 grid
        grid = Image.new('RGB', (w * 2, h * 2), color='white')
        
        # Paste them in (Top-Left: N1, Top-Right: N3, Bottom-Left: N5, Bottom-Right: N7)
        grid.paste(images[0], (0, 0))       
        grid.paste(images[1], (w, 0))       
        grid.paste(images[2], (0, h))       
        grid.paste(images[3], (w, h))       
        
        out_path = os.path.join(RESULTS_DIR, 'aggregated_confusion_matrices.png')
        
        # Resize it slightly down so the file isn't massive in the IEEE paper
        grid = grid.resize((int(w * 1.5), int(h * 1.5)), Image.Resampling.LANCZOS)
        grid.save(out_path, quality=95)
        print(f"\nSuccess! 2x2 aggregated figure saved at: {out_path}")
    else:
        print("Error: Not all 4 matrix images were found. Did you re-run the phase2_simplified plot generation?")

if __name__ == "__main__":
    aggregate_matrices()
