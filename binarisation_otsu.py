import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

image_path = "test_3.png"
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(f"Could not load {image_path}")

h, w = gray.shape
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

margin = int(w * 0.04)                      
core = thresh[:, margin:w - margin]
profile = np.count_nonzero(core, axis=1).astype(float)

#  Smoothen so tiny dips inside a line don't count as valleys
smooth = gaussian_filter1d(profile, sigma=4)

#  Estimate line spacing automatically 
p = smooth - smooth.mean()
ac = np.correlate(p, p, mode="full")[len(p) - 1:]
ac_peaks, _ = find_peaks(ac)
line_pitch = ac_peaks[0] if len(ac_peaks) else h // 10

#  Valleys of the profile = cut points between lines
valleys, _ = find_peaks(
    -smooth,
    distance=int(line_pitch * 0.6),
    prominence=0.08 * smooth.max(),
)

#  Build segments between cuts, drop blank ones (top/bottom margins)
cuts = np.concatenate(([0], valleys, [h]))
line_boundaries = []
for s, e in zip(cuts[:-1], cuts[1:]):
    if e - s > 0.4 * line_pitch and smooth[s:e].max() > 0.3 * smooth.max():
        line_boundaries.append((int(s), int(e)))

#  Save lines
output_dir = "output_lines"
os.makedirs(output_dir, exist_ok=True)
for i, (s, e) in enumerate(line_boundaries, 1):
    cv2.imwrite(os.path.join(output_dir, f"extracted_line_{i}.png"),
                gray[max(0, s - 2):min(h, e + 2), :])
print(f"Saved {len(line_boundaries)} lines (pitch ~ {line_pitch}px)")

#  Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharey=True,
                               gridspec_kw={"width_ratios": [3, 1]})
ax1.imshow(gray, cmap="gray")
for v in valleys:
    ax1.axhline(v, color="red", linewidth=1)
ax1.set_title(f"Detected lines: {len(line_boundaries)}")
ax2.plot(smooth, np.arange(h), color="blue")
ax2.plot(smooth[valleys], valleys, "rx")
ax2.set_title("Graph")
plt.tight_layout()
plt.savefig("segmentation_debug.png", dpi=150)
plt.show()