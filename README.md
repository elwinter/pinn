# pinn
Physics-informed neural networks

Known good parameters for problems:

lagaris/lagaris01
-----------------
n_layers = 1, n_hid = 10, w_dat = 0.5, max_epochs = 2000
Takes ~1 minute on mollie.

loop2d/loop2d_BxBy
------------------
n_layers = 4, n_hid = 100, w_dat = 0.95, max_epochs = 25000
Takes ~ 23 minutes on mollie.

loop2d/loop2d_nPuxuyuzBxByBz
----------------------------
n_layers = 4, n_hid = 100, w_dat = 0.95, max_epochs = 50000
Takes ~ 3.5-4 hours on mollie.
