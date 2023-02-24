import sys

from skimage.metrics import structural_similarity
import cv2

def calc_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2)


def main():
    print(f'Comparing {sys.argv[1]} and {sys.argv[2]}')
    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    print(f'ssim={calc_ssim(img1, img2)}')


if __name__=='__main__':
    main()


