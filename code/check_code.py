import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import re
import lightkurve as lk
import exoarch

import starspot as ss
import starspot.rotation_tools as rt
from stardate.lhf import gk_age_model
from stardate import load_samples, read_samples
from isochrones import get_ichrone

import kepler_data as kd
import kplr
client = kplr.API()


def get_index_from_filename(df, filename):
    kepid = re.findall(r'\d+', filename)[0]
    m = df.kepid == int(kepid)
    return int(np.arange(len(df))[m]), kepid


def load_lc(starname, timespan=200):
    # # lcfs = lk.search_lightcurvefile(starname).download_all()
    # lc1 = lk.search_lightcurvefile(starname, quarter=2)
    # lc2 = lk.search_lightcurvefile(starname, quarter=3)
    # # lc = lcf.PDCSAP_FLUX
    # # lc = lcfs.PDCSAP_FLUX.stitch()
    # lc_collection = lk.LightCurveCollection([lc1, lc2])
    # lc = lc_collection.stitch()

    star = client.star(starname)
    star.get_light_curves(fetch=True, short_cadence=False)
    LC_DIR = "/Users/rangus/.kplr/data/lightcurves/{}".format(starname)
    x, y, yerr = kd.load_and_join(LC_DIR)
    m = x < x[0] + timespan
    lc = lk.LightCurve(time=x[m], flux=y[m], flux_err=yerr[m])

    # Remove NaNs and sigma clip to remove flares.
    # no_nan_lc = lc.remove_nans()
    clipped_lc = lc.remove_outliers(sigma=4)
    return clipped_lc.time, clipped_lc.flux, clipped_lc.flux_err


def get_koi_properties(kid):
    kois = exoarch.KOICatalog().df
    k = kois.kepid == kid  # "K00092.01"
    dur = kois.koi_duration.values[k]
    porb = kois.koi_period.values[k]
    t0 = kois.koi_time0bk.values[k]
    nplanets = kois.koi_count.values[k]
    return dur, porb, t0, nplanets


def make_rotation_plots(time, flux, flux_err, interval=0.02043365,
                        period_grid=np.linspace(1, 50, 2000)):
    rotate = ss.RotationModel(time, flux, flux_err)
    ls_period = rotate.ls_rotation()
    acf_period = rotate.acf_rotation(interval)
    # pdm_period, period_err = rotate.pdm_rotation(rotate.lags, pdm_nbins=10)
    pdm_period, period_err = rotate.pdm_rotation(period_grid, pdm_nbins=10)
    fig = rotate.big_plot()
    return ls_period, acf_period, pdm_period, period_err, fig


def transit_mask_plot(time, flux, flux_err, kepid):
    dur, porb, t0, nplanets = get_koi_properties(kepid)
    mask = rt.transit_mask(time, t0[0], dur[0]*2, porb[0])

    fig = plt.figure(figsize=(16, 3))
    plt.plot(time, flux, "C1", lw=.5)
    plt.xlabel("Time - 2454833 [BKJD days]")
    plt.ylabel("Flux");
    plt.plot(time[mask], flux[mask], "C0", lw=.5)
    plt.xlabel("Time - 2454833 [BKJD days]")
    plt.ylabel("Normalized Flux");
    plt.tight_layout()
    return dur, porb, t0, nplanets, fig, mask


def cmd_gyro_plots(ro, period, age_gyr):
    mc = pd.read_csv("mcquillan_gaia.csv")

    # Calculate a simple gyro age.
    # log_age = gk_age_model(np.log10(ro.Prot), ro.bp_dered-ro.rp_dered)
    # age_gyr = (10**log_age)*1e-9

    fig = plt.figure(figsize=(16, 16), dpi=200)
    ax1 = fig.add_subplot(211)

    # Remove spurious very old stars!
    m = mc.ages_gyr.values < 15
    cb = ax1.scatter(mc.Teff[m], mc.Prot[m], c=mc.ages_gyr.values[m],
                     alpha=.1, vmin=0, vmax=5, edgecolor="none", zorder=0,
                     cmap="plasma",label="$\mathrm{McQuillan~+~(2014)}$")
    ax1.plot(ro.Teff, ro.period, "ko", ms=20, zorder=4)
    ax1.scatter([ro.Teff], [ro.period], c=[float(age_gyr)],
                s=200, vmin=0, vmax=5, edgecolor="w", cmap="plasma", zorder=5)
    ax1.scatter([5778], [26], c=[4.567],
                s=200, vmin=0, vmax=5, edgecolor="k", lw=2, zorder=2,
                cmap="plasma", label="$\mathrm{Sun}$")
    ax1.plot(5778, 26, "k.", zorder=3)
    color_bar = plt.colorbar(cb, ax=ax1, label="$\mathrm{Age~[Gyr]}$")
    color_bar.set_alpha(1)
    color_bar.draw_all()
    ax1.set_xlim(6500, 3000)
    ax1.set_yscale("log")
    ax1.set_ylabel("$\mathrm{Rotation~period~[days]}$")
    ax1.set_xlabel("$\mathrm{T_{eff}}$")

    # Add Praesepe
    cl = pd.read_csv("clean_clusters.csv")
    ax1.scatter(cl.teff, cl.period, c=cl.age_gyr,
                edgecolor="w", vmin=0, vmax=5, s=50, cmap="plasma", lw=.5,
                label="$\mathrm{Cluster~stars}$", zorder=1)
    # ax1.legend(fontsize=20)

    ax2 = fig.add_subplot(212)
    cb = ax2.scatter(mc.phot_bp_mean_mag - mc.phot_rp_mean_mag, mc.abs_G,
                     c=mc.Prot, s=50, alpha=.1, vmin=0, vmax=40,
                     edgecolor="none", zorder=0,
                     label="$\mathrm{McQuillan~+~(2014)}$");
    ax2.plot(ro.bp_dered - ro.rp_dered, ro.abs_G, "ko", ms=20, zorder=4)
    ax2.scatter([ro.bp_dered - ro.rp_dered], [ro.abs_G], c=[period],
                s=250, vmin=0, vmax=40, edgecolor="w", zorder=5);

    # Sun's Gaia mags from https://arxiv.org/pdf/1806.01953.pdf
    ax2.scatter([.82], [4.67], c=[26], s=300, edgecolor="k", zorder=2,
                vmin=0, vmax=40, label="$\mathrm{Sun}$")
    ax2.plot(.82, 4.67, "k.", zorder=3)
    ax2.scatter(cl.phot_bp_mean_mag - cl.phot_rp_mean_mag,
                m_to_M(cl.phot_g_mean_mag, 1./cl.parallax*1e3),
                c=cl.period, s = 100, vmin=0, vmax=40, edgecolor="w", lw=.5,
                label="$\mathrm{Cluster~stars}$", zorder=1)
    color_bar = plt.colorbar(cb, ax=ax2,
                             label="$\mathrm{Rotation~period~[days]}$")
    color_bar.set_alpha(1)
    color_bar.draw_all()
    ax2.set_ylim(7, 2)
    ax2.set_xlim(.6, 1.5)
    ax2.set_xlabel("$G_{BP} - G_{RP}$")
    ax2.set_ylabel("$\mathrm{Absolute~G~magnitude}$");
    # ax2.legend(fontsize=20)
    plt.tight_layout()
    return fig


def m_to_M(m, D):
    """
    Convert apparent magnitude to absolute magnitude.
    """
    return m - 5*np.log10(abs(D)) + 5


def get_inits(bp, rp, period, mass, feh, parallax_mas):

    # Get gyro age
    log10_age_yrs = gk_age_model(np.log10(period), bp-rp)
    gyro_age = (10**log10_age_yrs)*1e-9

    # Get initial EEP:
    mist = get_ichrone('mist')
    try:
        eep = mist.get_eep(mass, log10_age_yrs, feh, accurate=True)
    except:
        eep = 355

    distance_pc = 1./(parallax_mas*1e-3)
    inits = [eep, log10_age_yrs, feh, np.log(distance_pc), .1]
    return inits


def make_sample_comparison_plot(gyro_samples, iso_samples, inits, sigma):
    gs = gridspec.GridSpec(2, 4)

    fig = plt.figure(figsize=(16, 10), dpi=200)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist((10**gyro_samples[:, 1])*1e-9, 50, density=True, alpha=.5,
             color="k", label="$\mathrm{Gyro}$");
    ax1.hist((10**iso_samples[:, 1])*1e-9, 50, density=True, alpha=.5,
             label="$\mathrm{Iso}$");
    ax1.set_xlabel("$\mathrm{Age~[Gyr]}$")
    ax1.axvline((10**inits[1])*1e-9, color="C1", ls="--", zorder=10,
                label="$\mathrm{Initial~value}$")
    ax1.axvline((10**np.median(gyro_samples[:, 1])*1e-9), color="k", ls="--")
    ax1.axvline((10**np.median(iso_samples[:, 1])*1e-9), color="C0", ls="--")
    ax1.set_xlim(0, 14)
    ax1.legend(fontsize=15);
    plt.setp(ax1.get_yticklabels(), visible=False);
    plt.title("Sigma = {0}".format(sigma))

    def add_ax(i, init, gsamps, isamps, label):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(gsamps, 50, density=True, alpha=.5, color="k",
                label="$\mathrm{Gyro}$");
        ax.hist(isamps, 50, density=True, alpha=.5, label="$\mathrm{Iso}$");
        ax.set_xlabel(label)
        ax.axvline(init, color="C1", ls="--", zorder=10)
        ax.axvline(np.median(gsamps), color="k", ls="--")
        ax.axvline(np.median(isamps), color="C0", ls="--")
        plt.setp(ax.get_yticklabels(), visible=False);

    inds = [0, 2, 3, 4]
    labels = ["$\mathrm{EEP}$", "$\mathrm{[Fe/H]}$",
              "$\mathrm{ln(Distance~[pc])}$", "$\mathrm{A_v}$"]
    for i in range(4):
        add_ax(i, inits[inds[i]], gyro_samples[:, inds[i]],
               iso_samples[:, inds[i]], labels[i])

    plt.subplots_adjust(wspace=.02, hspace=.25)
    return fig
