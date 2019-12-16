#!/usr/bin/python3

import os
import sys
import numpy as np
import pandas as pd
import h5py
import tqdm
import emcee

import stardate as sd
from stardate.lhf import age_model
from isochrones import get_ichrone
mist = get_ichrone('mist')

from multiprocessing import Pool

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

def infer_stellar_age(df):

    # Set up the parameter dictionary.

    iso_params = {"G": (df["phot_g_mean_mag"], df["G_err"]),
                  "BP": (df["phot_bp_mean_mag"], df["bp_err"]),
                  "RP": (df["phot_rp_mean_mag"], df["rp_err"]),
                  "teff": (df["cks_steff"], df["cks_steff_err1"]),
                  "feh": (df["cks_smet"], df["cks_smet_err1"]),
                  "logg": (df["cks_slogg"], df["cks_slogg_err1"]),
                  "parallax": (df["parallax"], df["parallax_error"]}

    # Infer an age with isochrones and gyrochronology.

    gyro_fn = "{}_gyro".format(str(int(df["ID"])).zfill(9))
    iso_fn = "{}_iso".format(str(int(df["ID"])).zfill(9))

    # Get initialization
    bprp = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag.values"]
    log10_period = np.log10(df["Prot"])
    log10_age_yrs = age_model(log10_period, bprp)
    gyro_age = (10**log10_age_yrs)*1e-9

    eep = mist.get_eep(df["koi_smass"], np.log10(gyro_age*1e9),
                       df"[cks_smet"], accurate=True)

    inits = [eep, np.log10(gyro_age*1e9), df["cks_smet"],
             (1./df["parallax"])*1e3, df["Av"]]

    # Set up the star object
    iso_star = sd.Star(iso_params, Av=df["Av"], Av_err=df["Av_std"],
                       filename=iso_fn)
    gyro_star = sd.Star(iso_params, prot=df["Prot"], prot_err=df["e_Prot"],
                        Av=df["Av"], Av_err=df["Av_std"], filename=gyro_fn)

    # Run the MCMC
    iso_sampler = iso_star.fit(inits=inits, max_n=300000, save_samples=True)
    gyro_sampler = gyro_star.fit(inits=inits, max_n=300000, save_samples=True)


if __name__ == "__main__":
    #  Load the data file.
    df = pd.read_csv("cks_gaia_mazeh.csv")

    list_of_dicts = []
    for i in range(len(df)):
        list_of_dicts.append(df.iloc[i].to_dict())

    print(list_of_dicts[0])
    print(len(list_of_dicts))

    p = Pool(24)
    list(p.map(infer_stellar_age, list_of_dicts))
