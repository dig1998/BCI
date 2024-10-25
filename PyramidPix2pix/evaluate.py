import os
import cv2 as cv
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

# Function to compute PSNR and SSIM
def psnr_and_ssim(result_path):
    psnr = []
    ssim = []

    for filename in tqdm(os.listdir(result_path)):
        if 'merged_Generated IHC' in filename:
            try:
                # Construct paths for generated and real images
                generated_image_path = os.path.join(result_path, filename)
                real_image_path = os.path.join(result_path, filename.replace('merged_Generated IHC', 'merged_IHC'))

                # Read both images
                generated_img = cv.imread(generated_image_path)
                real_img = cv.imread(real_image_path)

                # Check if both images are successfully loaded
                if generated_img is None or real_img is None:
                    print(f"Could not read {filename} or its real counterpart.")
                    continue

                # Ensure images are at least 7x7
                if min(generated_img.shape[:2]) < 7 or min(real_img.shape[:2]) < 7:
                    print(f"Skipping {filename}: Image size too small.")
                    continue

                # Calculate PSNR
                PSNR = peak_signal_noise_ratio(real_img, generated_img)
                psnr.append(PSNR)

                # Calculate SSIM with specified window size and multichannel handling
                SSIM = structural_similarity(real_img, generated_img, multichannel=True, win_size=7, channel_axis=-1)
                ssim.append(SSIM)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Compute and print the average PSNR and SSIM
    if psnr and ssim:
        average_psnr = sum(psnr) / len(psnr)
        average_ssim = sum(ssim) / len(ssim)
        print(f"Average PSNR: {average_psnr}")
        print(f"Average SSIM: {average_ssim}")
    else:
        print("No valid images found or processed.")

if __name__ == "__main__":
    result_path = './results/pyramidpix2pix/test_latest/images'  # Update this path if needed
    psnr_and_ssim(result_path)
