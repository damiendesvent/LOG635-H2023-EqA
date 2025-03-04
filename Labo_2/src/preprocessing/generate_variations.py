import cv2

# returns 8 tuples (name_suffix, image)
def generate_variations(image):
    return (
        ('', image),
        ('_90', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ('_180', cv2.rotate(image, cv2.ROTATE_180)),
        ('_270', cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ('_flip', cv2.flip(image, 0)),
        ('_flip_90', cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_90_CLOCKWISE)),
        ('_flip_180', cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_180)),
        ('_flip_270', cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_90_COUNTERCLOCKWISE)),
    )

def get_flat_variations(image):
    return (
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.flip(image, 0),
        cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_180),
        cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_90_COUNTERCLOCKWISE),
    )