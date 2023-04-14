import os
import sys

import matplotlib as mpl
import numpy as np
import tensorflow as tf

from pinn import standard_plots
from pinn import training_data

epoch = sys.argv[1]

# Load the model.
base_path = os.path.join("linecurrent_BxBy")
path = os.path.join(base_path, "models", epoch, "model_Bx")
model_Bx = tf.keras.models.load_model(path)
path = os.path.join(base_path, "models", epoch, "model_By")
model_By = tf.keras.models.load_model(path)

# Create the points for the movie.
nt, nx, ny = 50, 50, 50
nxy = nx*ny
tmin, tmax = 0, 1
xmin, xmax = -1, 1
ymin, ymax = -1, 1
ng = [nt, nx, ny]
bg = [
    [tmin, tmax],
    [xmin, xmax],
    [ymin, ymax]
]
txy = training_data.create_training_points_gridded(ng, bg)

# Create the frame directory.
frame_dir = "./frames"
try:
    os.mkdir(frame_dir)
except:
    pass

# Create the frames.
mpl.use("Agg")
for i in range(nt):
    i0 = i*nxy
    i1 = i0 + nxy
    x = txy[i0:i1, 1]
    y = txy[i0:i1, 2]
    Bx = model_Bx(txy[i0:i1])
    By = model_By(txy[i0:i1])
    mpl.pyplot.clf()
    standard_plots.plot_BxBy_quiver(x, y, Bx, By)
    frame = mpl.pyplot.gcf()
    frame_name = f"frame_{i:06d}.png"
    mpl.pyplot.gca().set_title(f"t = {txy[i0, 0]:0.2f}")
    frame_path = os.path.join(frame_dir, frame_name)
    mpl.pyplot.savefig(frame_path)

# Create the movie.
os.system(f"convert -delay 10 -loop 0 {frame_dir}/frame_00*.png BxBy.gif")
