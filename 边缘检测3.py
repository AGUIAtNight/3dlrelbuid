from typing import Iterable, List, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def look_shape(a: Iterable) -> Tuple:
    # for debug
    return np.array(a).shape


def length_within_points(a: Iterable, empty_value: Union[int, float] = 0) -> int:
    """
        a simple instance:
            array : [empty_value, empty_value, empty_value, 1, empty_value, 0, 1, 2, empty_value]
            Then length_within_points(array) will return index diff between 1 and 2, which is 5
    """
    a = list(a)
    l_pivot, r_pivot = -1, -2
    for index, (l_val, r_val) in enumerate(zip(a[::1], a[::-1])):
        if l_val != empty_value and l_pivot == -1:
            l_pivot = index
        if r_val != empty_value and r_pivot == -2:
            r_pivot = len(a) - index

    return r_pivot - l_pivot + 1


def dump_rings_from_image(image: np.ndarray, output_path: str, plot_dict: dict = {"color": "k", "linewidth": 2.0}, default_height: float = 8) -> List[np.ndarray]:
    # regular operation, no more explainations
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 150)

    # get ratio between width and height to adjust the final output
    valid_width = length_within_points(edge.sum(axis=0))
    valid_height = length_within_points(edge.sum(axis=1))
    true_ratio = valid_width / valid_height

    # get contour of the edge image
    contour_tuple = cv2.findContours(
        edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = contour_tuple[0]
    rings = [np.array(c).reshape([-1, 2]) for c in contours]

    # adjust coordinate system to the image coordinate system
    max_x, max_y, min_x, min_y = 0, 0, 0, 0
    for ring in rings:
        max_x = max(max_x, ring.max(axis=0)[0])
        max_y = max(max_y, ring.max(axis=0)[1])
        min_x = max(min_x, ring.min(axis=0)[0])
        min_y = max(min_y, ring.min(axis=0)[1])

    # adjust ratio
    plt.figure(figsize=[default_height * true_ratio, default_height])

    # plot to the matplotlib
    for _, ring in enumerate(rings):
        close_ring = np.vstack((ring, ring[0]))
        xx = close_ring[..., 0]
        yy = max_y - close_ring[..., 1]
        plt.plot(xx, yy, **plot_dict)

    plt.axis("off")
    plt.savefig(output_path)


if __name__ == '__main__':
    img = cv2.imread('input.jpg')
    dump_rings_from_image(img, "KID.svg")
