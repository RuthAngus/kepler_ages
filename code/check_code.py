import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import lightkurve as lk
import exoarch
import starspot as ss
import starspot.rotation_tools as rt


def get_index_from_filename(df, filename):
    kepid = re.findall(r'\d+', filename)[0]
    m = df.kepid.values == int(kepid)
    return int(np.arange(len(df))[m]), kepid


def load_lc(starname, quarter):
    lcf = lk.search_lightcurvefile(starname, quarter="2").download()
    lc = lcf.PDCSAP_FLUX

    # Remove NaNs and sigma clip to remove flares.
    no_nan_lc = lc.remove_nans()
    clipped_lc = no_nan_lc.remove_outliers(sigma=4)
    return clipped_lc.time, clipped_lc.flux/np.median(clipped_lc.flux) - 1, \
        clipped_lc.flux_err


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

    fig1 = plt.figure(figsize=(10, 8))

    ax1 = fig1.add_subplot(211)
    ax1.plot(1./rotate.freq, rotate.power, "k")
    ax1.axvline(ls_period)
    ax1.axvline(ls_period/2., ls="--")
    ax1.axvline(ls_period*2., ls="--")
    ax1.set_ylabel("$\mathrm{Power}$")
    ax1.set_xlim(0, 40)
    plt.title("LS = {0:.2f} days, ACF = {0:.2f} days".format(ls_period,
                                                             acf_period))
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = fig1.add_subplot(212, sharex=ax1)
    ax2.plot(rotate.lags, rotate.acf, "k")
    ax2.axvline(acf_period)
    ax2.axvline(acf_period/2., ls="--")
    ax2.axvline(acf_period*2., ls="--")
    ax2.set_xlabel("$\mathrm{Time~[days]}$")
    ax2.set_ylabel("$\mathrm{Autocorrelation}$")
    plt.subplots_adjust(hspace=.0)

    pdm_period, period_err = rotate.pdm_rotation(period_grid, pdm_nbins=10)
    fig2 = rotate.pdm_plot();
    plt.title("PDM = {0:.2f} +/- {1:.2f} days".format(pdm_period, period_err))

    return ls_period, acf_period, pdm_period, period_err, fig1, fig2


def transit_mask_plot(time, flux, flux_err, kepid):
    dur, porb, t0, nplanets = get_koi_properties(kepid)

    mask = rt.transit_mask(time, t0, dur*2, porb)

    fig = plt.figure(figsize=(16, 3))
    plt.plot(time, flux, "C1", lw=.5)
    plt.xlabel("Time - 2454833 [BKJD days]")
    plt.ylabel("Flux");
    plt.plot(time[mask], flux[mask], "C0", lw=.5)
    plt.xlabel("Time - 2454833 [BKJD days]")
    plt.ylabel("Normalized Flux");
    return dur, porb, t0, nplanets, fig, mask
