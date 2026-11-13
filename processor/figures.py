"""Figures of one recording: PSD per region with the aperiodic fit, topomaps, PDR left vs right."""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np

from processor import channels as C

REGION_ORDER = ("frontal", "central", "temporal", "parietal", "occipital", "midline")
RESEARCH = "Research output, to be reviewed by a clinician."


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def psd_by_region(path: Path, rid: str, freqs: np.ndarray, psd: np.ndarray, regions: list[str],
                  fits: dict, fit_range: tuple[float, float]) -> None:
    """Mean PSD of each region (solid) and its specparam aperiodic fit (dashed), log-log."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    m = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    cmap = plt.get_cmap("tab10")
    for i, reg in enumerate([r for r in REGION_ORDER if r in regions] + sorted(set(regions) - set(REGION_ORDER))):
        sel = [j for j, r in enumerate(regions) if r == reg]
        spec = psd[sel].mean(axis=0)
        ax.plot(freqs[m], spec[m], color=cmap(i % 10), lw=1.6, label=f"{reg} ({len(sel)})")
        fit = fits.get(reg)
        if fit is not None and fit.ok:
            ax.plot(fit.freqs, 10 ** fit.aperiodic_log, color=cmap(i % 10), lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [t for t in (1, 2, 4, 8, 13, 20, 30, 45) if fit_range[0] <= t <= fit_range[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (µV²/Hz)")
    ax.set_title(f"{rid}\nmean spectrum per region; dashed: aperiodic (1/f) fit", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _info(names: list[str], contacts: dict[str, tuple[str, ...]]):
    import mne

    pos = {}
    for n in names:
        p = C.position(contacts.get(n, ()))
        # one marker per place: a reversed bipolar pair (P7-T7 next to T7-P7) shares its midpoint
        if p is not None and not any(np.allclose(p, q, atol=1e-4) for q in pos.values()):
            pos[n] = p
    if len(pos) < 4:
        return None, []
    tpl = C.montage().get_positions()
    dig = mne.channels.make_dig_montage(ch_pos=pos, nasion=tpl["nasion"], lpa=tpl["lpa"], rpa=tpl["rpa"],
                                        coord_frame="head")
    keep = list(pos)
    info = mne.create_info(keep, 256.0, "eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(dig, on_missing="ignore")
    return info, keep


def topomaps(path: Path, rid: str, names: list[str], contacts: dict[str, tuple[str, ...]],
             panels: list[tuple[str, np.ndarray, str]], bipolar: bool) -> bool:
    """One topomap per (title, values per channel in `names`, colormap). False when positions are missing."""
    import mne

    info, keep = _info(names, contacts)
    if info is None:
        return False
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    axes = np.atleast_1d(axes)
    idx = [names.index(k) for k in keep]
    for ax, (title, values, cmap) in zip(axes, panels):
        v = np.asarray(values, float)[idx]
        ok = np.isfinite(v)
        if ok.sum() < 3:
            ax.set_axis_off()
            ax.set_title(f"{title}\n(not enough values)", fontsize=9)
            continue
        v = np.where(ok, v, np.nanmedian(v))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=cmap, names=keep, contours=0,
                                         sensors=True, vlim=(float(np.nanmin(v)), float(np.nanmax(v))))
        for t in ax.texts:
            t.set_fontsize(6)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    note = " (bipolar pairs at their midpoints)" if bipolar else ""
    fig.suptitle(f"{rid}{note}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def pdr(path: Path, rid: str, freqs: np.ndarray, sides: dict[str, tuple[str, np.ndarray, float | None, float | None]],
        pdr_range: tuple[float, float], basis: str) -> None:
    """Left vs right posterior spectra 2-20 Hz with the PDR peak of each side marked.
    sides: {'left'|'right': (channel, psd, peak_hz, amplitude_uv)}."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    m = (freqs >= 2) & (freqs <= 20)
    colors = {"left": "#1f77b4", "right": "#d62728"}
    for side, (ch, spec, hz, amp) in sides.items():
        label = f"{side} {ch}"
        if hz is not None and math.isfinite(hz):
            label += f": {hz:.2f} Hz" + (f", ~{amp:.0f} µV p-p" if amp is not None and math.isfinite(amp) else "")
            ax.axvline(hz, color=colors[side], lw=0.8, ls=":")
        else:
            label += ": no peak"
        ax.plot(freqs[m], spec[m], color=colors[side], lw=1.6, label=label)
    ax.axvspan(*pdr_range, color="0.9", zorder=0, label=f"PDR search {pdr_range[0]:g}-{pdr_range[1]:g} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (µV²/Hz)")
    ax.set_title(f"{rid}\nposterior dominant rhythm, left vs right ({basis})", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
