# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# data_folder = Path(r"/home/gabriel/OneDrive/Doctorado-DESKTOP-JBOMLCA/Archivos/Data_19_06_25/Data")
data_folder = Path("./Data")
file_to_open = data_folder / "superfluid_density_in_the_lattice_B_in_(0.0-0.6)_Delta_s=0_Delta_S=0.2_lambda=0_points=16_theta=1.57_points=16_phi_angle=1.57_h=0.001_w_s=2.5_w_S=0.5_E_0=-6.npz"


Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
Lambda = Data["Lambda"]
Delta_S = Data["Delta_S"]
theta = Data["theta"]
mu = Data["mu"]
L_x = Data["L_x"]
h = Data["h"]

 

fig, ax = plt.subplots()
ax.plot(B_values/Delta_S, n_B_y[:,0], "-*g",  label=r"$n_{s,xx}(\theta=$"+f"{np.round(theta,2)}, 5.7GHz, L=3000)")
ax.plot(B_values/Delta_S, n_B_y[:,1], "-sg",  label=r"$n_{s,yy}(\theta=$"+f"{np.round(theta,2)}, \lambda_R=0.056)")
