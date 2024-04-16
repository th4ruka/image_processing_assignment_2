import cv2
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


def add_gaussian_noise(image, mean, variance):
    noise = np.random.normal(mean, np.sqrt(variance), image.shape)
    noisy_image = image + noise
    return noisy_image.astype(np.uint8)


def otsu_thresholding(image):
    threshold_value = threshold_otsu(image)
    binary_image = image > threshold_value
    return binary_image


def region_growing(image, seeds, threshold):
    segmented_image = np.zeros_like(image)
    rows, cols = image.shape
    image = image.astype(np.int16)
    for seed_x, seed_y in seeds:
        segmented_image[seed_x, seed_y] = 255

        current_pixels = [(seed_x, seed_y)]

        while current_pixels:
            x, y = current_pixels.pop()

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue  # Skip the current pixel

                    new_x = x + dx
                    new_y = y + dy

                    # Check boundaries and if it's already segmented
                    if (rows > new_x >= 0 == segmented_image[new_x, new_y] and
                            0 <= new_y < cols and
                            abs(image[new_x, new_y] - image[x, y]) < threshold):
                        segmented_image[new_x, new_y] = 255
                        current_pixels.append((new_x, new_y))

    return segmented_image


if __name__ == "__main__":
    image_path = 'src/cat.jpg'
    image = load_image(image_path)

    # Task 1 - Otsu's Thresholding
    noisy_image = add_gaussian_noise(image, 0, 0.1)
    binary_image = otsu_thresholding(noisy_image)

    # Task 2 - Region Growing
    seeds = [(50, 50), (100, 80)]
    segmented_image = region_growing(image.copy(), seeds, threshold=5)

    plt.figure(figsize=(10, 6))
    plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(132), plt.imshow(binary_image, cmap='gray'), plt.title('Otsu Implementation')
    plt.subplot(133), plt.imshow(segmented_image, cmap='gray'), plt.title('Region Grown')
    plt.show()
