"""
Check that the KOI rotation periods seem reasonable.
"""

import numpy as np
import pandas as pd
import check_code as cc
from starspot import rotation_tools as rt
import starspot as ss
import matplotlib.pyplot as plt

# Load big CKS table
df = pd.read_csv("cks_gaia_mazeh.csv")

print("Load light curve, sigma clip and remove NaNs.")
i = 0
kepid = df.kepid.values[i]
starname = "KIC {}".format(kepid)
time, flux, flux_err = cc.load_lc(starname, 2)
d, t0, p, n, fig0, mask = cc.transit_mask_plot(time, flux, flux_err, kepid)
fig0.savefig("plots/transit_mask_plot")

time, flux, flux_err = time[mask], flux[mask], flux_err[mask]

print("Make period plots")
ls_period, acf_period, pdm_period, period_err, fig1, fig2 = \
    cc.make_rotation_plots(time, flux, flux_err)
fig1.savefig("plots/ls_acf_plot")
fig2.savefig("plots/pdm_plot")
