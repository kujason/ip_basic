import cv2


def cv2_show_image(window_name, image,
                   size_wh=None, location_xy=None):
    """Helper function for specifying window size and location when
    displaying images with cv2.

    Args:
        window_name: str window name
        image: ndarray image to display
        size_wh: window size (w, h)
        location_xy: window location (x, y)
    """

    if size_wh is not None:
        cv2.namedWindow(window_name,
                        cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, *size_wh)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    if location_xy is not None:
        cv2.moveWindow(window_name, *location_xy)

    cv2.imshow(window_name, image)
