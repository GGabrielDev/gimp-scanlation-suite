import numpy as np

def resize_image(img, new_w, new_h):
    """
    Bilinear interpolation to resize a numpy image of shape (H, W, C).
    Done in pure NumPy to avoid dependency on OpenCV or PIL.
    """
    h, w, c = img.shape
    x = np.linspace(0, w - 1, new_w)
    y = np.linspace(0, h - 1, new_h)

    x_indices = np.floor(x).astype(np.int32)
    y_indices = np.floor(y).astype(np.int32)

    x_indices_next = np.minimum(x_indices + 1, w - 1)
    y_indices_next = np.minimum(y_indices + 1, h - 1)

    x_weight = (x - x_indices).reshape(1, new_w, 1)
    y_weight = (y - y_indices).reshape(new_h, 1, 1)

    p00 = img[y_indices][:, x_indices]
    p10 = img[y_indices][:, x_indices_next]
    p01 = img[y_indices_next][:, x_indices]
    p11 = img[y_indices_next][:, x_indices_next]

    interpolated = (
        p00 * (1 - x_weight) * (1 - y_weight) +
        p10 * x_weight * (1 - y_weight) +
        p01 * (1 - x_weight) * y_weight +
        p11 * x_weight * y_weight
    )
    return interpolated.astype(np.uint8)

def to_grayscale(img_np):
    """
    Convert RGB/RGBA input to grayscale using numpy luminosity math,
    and stack back to 3 channels to preserve the expected model shape.
    """
    # Use standard luminosity coefficients
    gray_img = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return np.stack((gray_img,) * 3, axis=-1)

def prepare_input_tensor(img_np, image_size):
    """
    Resizes the image preserving aspect ratio, pads with white pixels (255) to image_size,
    and converts to float32 NCHW tensor layout normalized to [0.0, 1.0].
    Returns (input_tensor, resized_w, resized_h).
    """
    orig_h, orig_w, _ = img_np.shape

    if orig_w >= orig_h:
        resized_w = image_size
        resized_h = int(image_size * orig_h / orig_w)
    else:
        resized_w = int(image_size * orig_w / orig_h)
        resized_h = image_size

    resized_w = max(1, resized_w)
    resized_h = max(1, resized_h)

    resized_img = resize_image(img_np, resized_w, resized_h)

    # Pad with white pixels (255) instead of black to prevent edge-detection failures
    padded_img = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    padded_img[0:resized_h, 0:resized_w, :] = resized_img

    input_data = padded_img.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    return input_data, resized_w, resized_h
