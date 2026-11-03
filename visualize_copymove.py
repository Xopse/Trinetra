"""
visualize_copymove.py — Draws the copy-move keypoint matches found by
image_forensics.py directly onto the image, so we can see WHERE they
are and judge whether they're a genuine duplicated patch or a
false-positive triggered by a legitimate repetitive security pattern
(common on real ID cards — guilloche patterns, repeated micro-text,
watermark-style backgrounds).

Usage: python visualize_copymove.py <path_to_image>
Saves: copymove_visualization.png (safe to inspect locally; matched
point COORDINATES are not personal data, but the underlying image
still is — view locally, don't need to share it unless you want to)
"""

import sys
import cv2
from image_forensics import copy_move_detection, _get_qr_region_boxes

if len(sys.argv) != 2:
    print("Usage: python visualize_copymove.py <path_to_image>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
matches = copy_move_detection(img, _get_qr_region_boxes(img))

print(f"Found {len(matches)} matches")

vis = img.copy()
for m in matches:
    pt_a = tuple(map(int, m["point_a"]))
    pt_b = tuple(map(int, m["point_b"]))
    color = (0, 0, 255)  # red
    cv2.circle(vis, pt_a, 6, color, 2)
    cv2.circle(vis, pt_b, 6, (0, 255, 0), 2)  # green for the paired point
    cv2.line(vis, pt_a, pt_b, (255, 255, 0), 1)  # cyan line connecting the pair

cv2.imwrite("copymove_visualization.png", vis)
print("Saved to copymove_visualization.png — open it locally to see where matches cluster.")
print("Red circles = point A of each match, green circles = point B, cyan lines connect pairs.")
