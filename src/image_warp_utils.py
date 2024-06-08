import cv2
import numpy as np


def warp_image(image_path, src_points, dst_points):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    M = cv2.getAffineTransform(src_points, dst_points)

    warped_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return warped_image


def horizontal_stretch(image, stretch_factor):
    height, width = image.shape[:2]
    M = np.float32([[stretch_factor, 0, 0], [0, 1, 0]])
    return cv2.warpAffine(image, M, (int(width * stretch_factor), height))


def vertical_stretch(image, stretch_factor):
    height, width = image.shape[:2]
    M = np.float32([[1, 0, 0], [0, stretch_factor, 0]])
    return cv2.warpAffine(image, M, (width, int(height * stretch_factor)))


def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)


def plot_images(images, titles):
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


def main():
    ## image = cv2.imread()
    ## image = warp_image(image)
    pass


if __name__ == "__main__":
    main()
