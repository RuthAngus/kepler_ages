"""
Check that the KOI rotation periods and ages seem reasonable.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glob
import corner

import check_code as cc
from starspot import rotation_tools as rt
import starspot as ss
from stardate import load_samples, read_samples
from stardate.lhf import sigma


def make_period_plots_for_one_star(df, timespan, mask_transit_plot=True):

    print("Load light curve, sigma clip and remove NaNs.")
    kepid = df.kepid
    starname1 = "KIC {}".format(kepid)
    starname = str(int(df.kepid)).zfill(9)
    time, flux, flux_err = cc.load_lc(starname, timespan=timespan)

    if mask_transit_plot:
        print("Mask transits")
        d, p, t0, n, fig0, mask = cc.transit_mask_plot(time, flux, flux_err,
                                                    kepid)
        print("duration = ", d, "t0 = ", t0, "porb = ", p)
        fig0.savefig("plots/{}_atransit_mask_plot".format(starname))
        plt.close()

        time, flux, flux_err = time[mask], flux[mask], flux_err[mask]

    print("Make period plots")
    ls_period, acf_period, pdm_period, period_err, fig = \
    cc.make_rotation_plots(time, flux, flux_err)
    plt.axvline(df.Prot, color="C1")
    fig.savefig("plots/{}_big_plot".format(starname))
    plt.close()

    return ls_period, acf_period, pdm_period, period_err, starname


def make_age_plots_for_one_star(df, period, period_err, cmd_gyro=True,
                                corner_plot=True, age_plot=True):

    starname = str(int(df.kepid)).zfill(9)
    print("starname = ", starname)

    print("Load samples_and_results")
    gfilename = "samples/{0}_gyro.h5".format(starname)
    ifilename = "samples/{0}_iso.h5".format(starname)
    gflatsamples, _, _, _ = load_samples(gfilename, burnin=100)
    gresults = read_samples(gflatsamples)
    iflatsamples, _, _, _ = load_samples(ifilename, burnin=100)
    iresults = read_samples(iflatsamples)

    if cmd_gyro:
        print("Make CMD plot")
        age_gyr = gresults.age_med_gyr.values
        fig1 = cc.cmd_gyro_plots(df, period, age_gyr)
        fig1.savefig("plots/{}_cmd_plot".format(starname))
        plt.close()

    inits = cc.get_inits(df.bp_dered, df.rp_dered, period, df.koi_smass,
                         df.cks_smet, df.parallax)
    print("inits = ", inits)

    if corner_plot:
        labels = ["EEP", "log10(Age [yr])", "[Fe/H]", "ln(Distance)", "Av",
                    "ln(probability)"]
        inits.append("None")
        fig2 = corner.corner(gflatsamples, labels=labels, truths=inits);
        fig2.savefig("plots/{}_gyro_corner".format(starname))
        plt.close()
        fig3 = corner.corner(iflatsamples, labels=labels, truths=inits);
        fig3.savefig("plots/{}_iso_corner".format(starname))
        plt.close()

    print("Calculate sigma")
    sig = sigma(gresults.EEP_med.values,
                np.log10(gresults.age_med_gyr.values*1e9),
                gresults.feh_med.values, df.bp_dered-df.rp_dered)
    print("Sigma = ", sig)

    if age_plot:
        fig4 = cc.make_sample_comparison_plot(gflatsamples, iflatsamples,
                                              inits, sig)
        fig4.savefig("plots/{}_comparison_plot".format(starname))
        plt.close()
    return gresults

def get_successful_df():
    """
    Use glob to get a list of h5 filenames and return a dataframe containing
    only stars with samples.
    """

    # Load file with info for all CKS stars
    ro = pd.read_csv("cks_gaia_mazeh.csv")

    # Load filenames of successful posteriors
    gyro_filenames = glob.glob("samples/*gyro.h5")
    iso_filenames = glob.glob("samples/*iso.h5")

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
    df.to_csv("success.csv")
    return df


if __name__ == "__main__":

    # df = get_successful_df()
    df = pd.read_csv("success.csv")

    timespan = 300
    ls_period, acf_period, pdm_period, period_err, kepid = \
        make_period_plots_for_one_star(df.iloc[0], timespan)
    results = make_age_plots_for_one_star(df.iloc[0], df.Prot.values[0],
                                          df.e_Prot.values[0],
                                          corner_plot=False)
    results["ls_period"] = ls_period
    results["acf_period"] = acf_period
    results["pdm_period"] = pdm_period
    results["period_err"] = period_err
    results["kepid"] = kepid

    for i in range(1, len(df)):
        ls_period, acf_period, pdm_period, period_err, kepid = \
            make_period_plots_for_one_star(df.iloc[i], timespan)
        new_results = make_age_plots_for_one_star(df.iloc[i],
                                                  df.Prot.values[i],
                                                  df.e_Prot.values[i],
                                                  corner_plot=False)
        new_results["ls_period"] = ls_period
        new_results["acf_period"] = acf_period
        new_results["pdm_period"] = pdm_period
        new_results["period_err"] = period_err
        new_results["kepid"] = kepid
        results = pd.concat((results, new_results))

    results.to_csv("results.csv")
