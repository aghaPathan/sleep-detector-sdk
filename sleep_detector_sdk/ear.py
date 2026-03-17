"""Eye Aspect Ratio (EAR) computation.

The EAR formula measures the ratio of vertical to horizontal eye distances:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

Where p1-p6 are the 6 landmark points of an eye:
    p1, p4: eye corners (horizontal)
    p2, p3: upper eyelid points
    p5, p6: lower eyelid points
"""

import numpy as np
from scipy.spatial import distance as dist


def compute_ear(eye: np.ndarray) -> float:
    """Compute the Eye Aspect Ratio for a single eye.

    Args:
        eye: Array of shape (6, 2) containing the 6 eye landmark coordinates.

    Returns:
        The EAR value (float >= 0). Higher values indicate more open eyes.
    """
    vertical_a = dist.euclidean(eye[1], eye[5])
    vertical_b = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])

    if horizontal == 0:
        return 0.0

    return (vertical_a + vertical_b) / (2.0 * horizontal)
