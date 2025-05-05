# /home/delliott@int-episci.com/sam2/demo/backend/server/inference/mask_utils.py
"""
Module providing helper functions for working with segmentation masks.
These functions convert masks to RLE data values using pycocotools.
"""

from typing import List
import numpy as np
from inference.data_types import PropagateDataValue, Mask
from pycocotools.mask import encode as encode_masks

def get_mask_for_object(object_id: int, mask: np.ndarray) -> PropagateDataValue:
    """
    Create a PropagateDataValue instance for the given object and mask.

    Args:
        object_id (int): The ID of the object.
        mask (np.ndarray): Binary mask (as a NumPy array).

    Returns:
        PropagateDataValue: Data value with the RLE mask.
    """
    mask_rle = encode_masks(np.array(mask, dtype=np.uint8, order="F"))
    # Decode the counts from bytes to string.
    mask_rle["counts"] = mask_rle["counts"].decode()
    return PropagateDataValue(
        object_id=object_id,
        mask=Mask(
            size=mask_rle["size"],
            counts=mask_rle["counts"],
        ),
    )

def get_rle_mask_list(object_ids: List[int], masks: np.ndarray) -> List[PropagateDataValue]:
    """
    Generate a list of RLE masks for multiple objects.

    Args:
        object_ids (List[int]): List of object IDs.
        masks (np.ndarray): Array of binary masks.

    Returns:
        List[PropagateDataValue]: List of data values for each object.
    """
    return [get_mask_for_object(object_id, mask) for object_id, mask in zip(object_ids, masks)]