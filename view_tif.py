# %%
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rioxarray as rxr
from matplotlib.colors import BoundaryNorm, ListedColormap

class_labels = [
    "1: Water",
    "2: Tree Canopy",
    "3: Low Vegetation",
    "4: Barren Land",
    "5: Impervious (Other)",
    "6: Impervious (Road)",
    "15: No Data",
]

colors = [
    (0 / 255, 0 / 255, 255 / 255, 1),  # Deep Blue for water
    (34 / 255, 139 / 255, 34 / 255, 1),  # Forest Green for tree canopy / forest
    (154 / 255, 205 / 255, 50 / 255, 1),  # Yellow Green for low vegetation / field
    (210 / 255, 180 / 255, 140 / 255, 1),  # Tan for barren land
    (169 / 255, 169 / 255, 169 / 255, 1),  # Dark Gray for impervious (other)
    (105 / 255, 105 / 255, 105 / 255, 1),  # Dim Gray for impervious (road)
    (255 / 255, 255 / 255, 255 / 255, 1),  # White for no data
]
cmap = ListedColormap(colors)

# %% view model prediction
CHIP_PATH = Path.home() / "data_science/geospatial/clay/model/data/cvpr/ny/val/chips"
LABEL_PATH = Path.home() / "data_science/geospatial/clay/model/data/cvpr/ny/val/labels"
PRED_PATH = Path.home() / "data_science/geospatial/clay/model/data/cvpr/ny/val/predictions"
all_val_chips = sorted(list(CHIP_PATH.glob("*.npy")))


# %%
input_file = random.choice(all_val_chips)
label_file = LABEL_PATH / input_file.name.replace("naip-new_chip", "lc_chip")
# prediction = PRED_PATH / f"{input_file.stem}_prediction.png"
prediction = CHIP_PATH / f"{input_file.stem}_prediction.png"

# %%
!python deployment/aws_lambda/test_local.py {input_file}

# %%
image_in = np.load(input_file)[:3, :, :].transpose(1, 2, 0)
label_in = np.load(label_file).squeeze()
pred = mpimg.imread(prediction)

# %% plot results
_, ax = plt.subplots(2, 2, figsize=(9, 9))
ax = ax.ravel()

ax[0].imshow(image_in)
ax[0].set_title("Input Image (RGB)")

norm = BoundaryNorm(
    boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 15.5], ncolors=cmap.N
)

ax[2].imshow(label_in, cmap=cmap, norm=norm)
ax[2].set_title("Ground Truth Labels")

ax[1].imshow(pred)
ax[1].set_title("Model Prediction")

ax[3].set_axis_off()

# draw a legend for the labels in ax[3]
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10)
    for color in colors
]
ax[3].legend(handles, class_labels, title="Classes", loc="center")


# %%
files = Path("data/cvpr/files/train").glob("*.tif")
files = sorted(list(files))
files

# %% this is a single band landcover classification
da = rxr.open_rasterio(files[0]).squeeze()
da

# %%
# !gdalinfo "{files[0]}"

# %%
# np.unique(da)

da.plot.imshow(cmap=cmap, add_colorbar=False)
plt.title(files[0].name)
# fake a legend.  there are 6 classes from 1 to 6
colors = [cmap(i) for i in range(cmap.N)]

handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10)
    for color in colors
]
plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")


# %%
da = rxr.open_rasterio(files[1]).squeeze()
da

# %%
da.plot.imshow()
plt.title(files[1].name)

# %%
da

# %% run gdalifno on the first file
# !gdalinfo data/cvpr/files/train/m_4207532_se_18_1_naip-new.tif


# %%
