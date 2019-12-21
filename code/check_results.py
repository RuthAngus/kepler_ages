"""
Script for checking the results of age inference.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

import check_code as cc

# Load file with info for all CKS stars
ro = pd.read_csv("cks_gaia_mazeh.csv")

# Load successful posteriors
gyro_filenames = glob.glob("samples/*gyro.h5")
iso_filenames = glob.glob("samples/*iso.h5")
print(len(gyro_filenames), len(iso_filenames))

gyro_inds, iso_inds = [], []
for i in range(len(gyro_filenames)):
    ind = cc.get_index_from_filename(ro, gyro_filenames[i])[0]
    gyro_inds.append(ind)

for i in range(len(iso_filenames)):
    ind = cc.get_index_from_filename(ro, iso_filenames[i])[0]
    iso_inds.append(ind)

# Merge so you're only looking at stars where both gyro and iso ran.
gyro_ro = ro.iloc[gyro_inds]
iso_ro = ro.iloc[iso_inds]
iso_df = pd.DataFrame(dict({"kepid": iso_ro.kepid.values}))
df = pd.merge(gyro_ro, iso_df, on="kepid", how="inner")


