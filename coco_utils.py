from pycocotools.mask import encode
import numpy as np


def encode_mask(mask: np.ndarray):
    fortran_binary_mask = np.asfortranarray(mask)
    encoded_mask = encode(fortran_binary_mask)
    return encoded_mask
