"""
sgs.py
======

Sequential Gaussian Simulation (SGS) of a scalar attribute in any
GeoPandas-compatible point dataset (GeoJSON, Shapefile, GeoPackage, ...).

Pipeline (classic geostatistical workflow):
  1. Load points, keep rows with a valid value in the target column.
  2. Get x/y coordinates in a metric (projected) frame -- either from
     user-specified columns, or from the geometry (auto-reprojected to
     an estimated local UTM zone if the source CRS is geographic).
  3. Normal-score (Gaussian) transform of the target values.
  4. Compute + fit an experimental variogram of the transformed values.
  5. Sequential Gaussian Simulation on a regular grid: visit grid nodes
     in random order, krige (simple kriging, moving neighborhood) using
     original data + all previously simulated nodes, draw from the local
     conditional Gaussian distribution, back-transform, repeat for
     N realisations. Realisations are independent of each other and can
     run in parallel across processes.
  6. Summarize realisations (E-type mean, std = uncertainty) and plot.

Requirements
------------
numpy, scipy, pandas, geopandas, matplotlib

Usage
-----
    # single run over the whole file
    python sgs.py all_data.geojson -c "CO2_FLUX [mol/m2/day]" \
        -x "UTM LONGITUDE" -y "UTM LATITUDE" \
        -o sgs_results -n 20 -j -1

    # one independent SGS run per distinct value of a grouping column
    # (e.g. the original per-year use case)
    python sgs.py all_data.geojson -c "CO2_FLUX [mol/m2/day]" \
        -x "UTM LONGITUDE" -y "UTM LATITUDE" \
        --group-by year -o results -n 20 -j -1 -v exponential

Key flags
---------
    -c/--data-col         column to interpolate (required)
    -x/-y                 coordinate columns (optional; default: derive from
                            geometry, auto-projecting to a local UTM CRS if 
                            the data is in lon/lat degrees)
    --group-by            optional column; run one independent SGS per group
                            (e.g. "year") instead of one run over everything
    -j/--n-jobs           realisations to run in parallel: 1=sequential
                            (default), -1=all CPU cores, N = N cores
    -n/--n-realisations   total number of realisations
    -g/--grid-res         grid resolution (in UTM units/[m])
    -m/--max-neighbors    max number of neighbouring values to use in kdir
    -l/--n-lags           number of lags
    -v/--variogram-model  form of the variogram model
    -f/--variogram-fit    path to fitted variogram parameters (default=None)
    -s/--seed             random seed value
    -p/--prefix           prefix to add before data and figures (see below,
                          default="")

Outputs (in --outdir, prefixed by data column / group value):
    <prefix>_<data-col>_variogram.png      experimental vs fitted variogram
    <prefix>_<data-col>_data.png           sample locations colored by value
    <prefix>_<data-col>_realisation_1.png  one example simulated field
    <prefix>_<data-col>_etype_mean.png     mean of all realisations
    <prefix>_<data-col>_etype_std.png      std-dev of all realisations
                                           (uncertainty)
    <prefix>_<data-col>_grids.npz          raw arrays (X, Y, realisations,
                                           mean, std)

Reproducibility note: results depend only on --seed and --n-realisations,
not on --n-jobs. Each realisation gets its own child seed (via
numpy.random.SeedSequence.spawn), so running with -j 1 or -j 8 produces
the exact same set of realisations, just faster.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import curve_fit
from scipy.stats import rankdata, norm
from numpy.typing import ArrayLike
from typing import List, Dict, Tuple
from dataclasses import dataclass, field, asdict

import argparse
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Normal-score transform
# --------------------------------------------------------------------------

class NormalScoreTransform:
    """Rank-based Gaussian anamorphosis with linear-interpolation back
    transform. Fit on the hard data; used to move data <-> Gaussian space."""

    def __init__(self, values):
        values = np.asarray(values, dtype=float)
        order  = np.argsort(values)
        self.sorted_values = values[order]
        n     = len(values)
        ranks = rankdata(values, method="average")
        u     = (ranks - 0.5) / n
        self.sorted_scores = np.sort(norm.ppf(u))
        u_sorted = (np.arange(1, n + 1) - 0.5) / n
        self.scores_for_sorted_values = norm.ppf(u_sorted)

    def forward(self, values):
        return np.interp(values, self.sorted_values,
                         self.scores_for_sorted_values)

    def backward(self, scores):
        return np.interp(
            scores, self.scores_for_sorted_values, self.sorted_values,
            left=self.sorted_values[0], right=self.sorted_values[-1])


# --------------------------------------------------------------------------
# Variogram: experimental + model fitting
# --------------------------------------------------------------------------

@dataclass
class VariogramParams:
    """
    Parameters for variogram models
    """
    model  : str | None  = None
    nugget : float = 0
    sill   : float = 1  # nugget + sill should equal 1
    range_ : float = 1

    def to_dict(self):
        return dict(model=self.model, nugget=self.nugget, sill=self.sill,
                    range_=self.range_)

    def to_list(self):
        return list(self.to_dict().values())


def experimental_variogram(x: ArrayLike, y: ArrayLike, values: ArrayLike, 
                           n_lags: int=12, max_dist: float=None) -> Tuple:
    """
    Returns the lags, semivariance and number of values in each lag bin from
    an isotropic experimental variogram.

    Parameters
    ----------
    x        : array-like
        Horizontal positions
    y        : array-like
        Vertical positions
    values   : array-like
        Values of the observable
    n_lags   : int
        Number of lags (bins)
    max_dist : float
        Maximum distance for calculation of lags

    Returns
    -------
    lags   : np.ndarray
    gamma  : np.ndarray
    counts : np.ndarray
    """
    n = len(x)
    xi, xj = np.meshgrid(x, x, indexing="ij")
    yi, yj = np.meshgrid(y, y, indexing="ij")
    vi, vj = np.meshgrid(values, values, indexing="ij")
    iu, ju = np.triu_indices(n, k=1)
    dist = np.sqrt((xi[iu, ju] - xj[iu, ju])**2
                   + (yi[iu, ju] - yj[iu, ju])**2)
    sqdiff = 0.5 * (vi[iu, ju] - vj[iu, ju])**2

    if max_dist is None:
        max_dist = np.percentile(dist, 70)

    bins = np.linspace(0, max_dist, n_lags + 1)
    bin_idx = np.digitize(dist, bins) - 1

    lag_centers, gamma, counts = [], [], []
    for b in range(n_lags):
        mask = bin_idx == b
        # Do not add lag etc. if too few counts
        if mask.sum() < 3:
            continue
        lag_centers.append(dist[mask].mean())
        gamma.append(sqdiff[mask].mean())
        counts.append(mask.sum())
    return np.array(lag_centers), np.array(gamma), np.array(counts)


def spherical(h: ArrayLike, nugget: float, sill: float, rng: float) -> ArrayLike:
    h = np.asarray(h, dtype=float)
    out = nugget + sill * (1.5 * (h / rng) - 0.5 * (h / rng) ** 3)
    out = np.where(h >= rng, nugget + sill, out)
    out = np.where(h <= 0, 0.0, out)
    return out


def exponential(h: ArrayLike, nugget: float, sill: float, rng: float) -> ArrayLike:
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1 - np.exp(-3 * h / rng))


def gaussian_model(h: ArrayLike, nugget: float, sill: float, 
                   rng: float) -> ArrayLike:
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1 - np.exp(-3 * (h / rng) ** 2))


MODELS = {"spherical":   spherical,
          "exponential": exponential,
          "gaussian":    gaussian_model}


def fit_variogram(lags: ArrayLike, gamma: ArrayLike, 
                  model : str="spherical") -> VariogramParams:
    """
    Fits a variogram model to lag and semivariance data, returning the
    VariogramParams dataclass
    """
    func    = MODELS[model]
    sill0   = gamma.max()
    rng0    = lags.max() * 0.5
    nugget0 = max(gamma.min(), 0.0)
    p0      = [nugget0, sill0, rng0]
    bounds  = ([0,                  1e-8,             1e-3],
               [sill0 * 1.5 + 1e-8, sill0 * 2 + 1e-8, lags.max() * 3])
    try:
        popt, _ = curve_fit(func, lags, gamma, p0=p0,
                            bounds=bounds, maxfev=10000)
    except RuntimeError:
        popt = p0
    return VariogramParams(model, popt[0], popt[1], popt[2])


def man_variogram(model: str, nugget: float, sill: float,
                  range_: float) -> VariogramParams:
    '''Return a variogram dict with manually entered parameters'''
    return VariogramParams(model, nugget, sill, range_)


class VariogramCovariance(VariogramParams):
    """Picklable covariance function C(h) = total_sill - gamma(h).
    Built from a VariogramParams instance so it can be sent to worker 
    processes for parallel SGS."""

    def __init__(self, params: VariogramParams):
        super().__init__(**asdict(params))
        if self.model not in MODELS:
            raise ValueError(f"Unknown variogram model {self.model!r}; "
                             f"expected one of {list(MODELS)}")
    @property
    def total_sill(self):
        return self.nugget + self.sill

    def __call__(self, h):
        func = MODELS[self.model]
        h    = np.asarray(h, dtype=float)
        gam  = np.where(h <= 1e-12, 0.0,
                        func(h, self.nugget, self.sill, self.range_))
        return self.total_sill - gam


def covariance_function(params):
    """Return (C, total_sill) given a fitted variogram model dict."""
    cov = VariogramCovariance(params)
    return cov, cov.total_sill


# --------------------------------------------------------------------------
# Simple kriging (moving neighborhood) — building block for SGS
# --------------------------------------------------------------------------

def simple_krige(x0, y0, xs, ys, vals, cov_func, total_sill,
                 mean=0.0, eps=1e-8):
    n = len(xs)
    if n == 0:
        return mean, total_sill

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = np.sqrt(dx ** 2 + dy ** 2)
    K = cov_func(dist_matrix)
    K[np.diag_indices(n)] += eps

    d0 = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
    k0 = cov_func(d0)

    try:
        weights = np.linalg.solve(K, k0)
    except np.linalg.LinAlgError:
        weights, *_ = np.linalg.lstsq(K, k0, rcond=None)

    est = mean + np.sum(weights * (vals - mean))
    var = total_sill - np.sum(weights * k0)
    var = max(var, 0.0)
    return est, var


# --------------------------------------------------------------------------
# Sequential Gaussian Simulation
# --------------------------------------------------------------------------

def sgs_realisation(grid_x, grid_y, data_x, data_y, data_g,
                    cov_func, total_sill,
                    max_neighbors=24, search_radius=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_grid = len(grid_x)
    path = rng.permutation(n_grid)

    cond_x = list(data_x)
    cond_y = list(data_y)
    cond_v = list(data_g)

    sim_values = np.full(n_grid, np.nan)

    for idx in path:
        x0, y0 = grid_x[idx], grid_y[idx]
        cx = np.asarray(cond_x)
        cy = np.asarray(cond_y)
        cv = np.asarray(cond_v)

        d = np.sqrt((cx - x0) ** 2 + (cy - y0) ** 2)
        if search_radius is not None:
            keep = d <= search_radius
            d, cx, cy, cv = d[keep], cx[keep], cy[keep], cv[keep]

        if len(cx) > max_neighbors:
            nearest = np.argpartition(d, max_neighbors)[:max_neighbors]
            cx, cy, cv = cx[nearest], cy[nearest], cv[nearest]

        est, var = simple_krige(x0, y0, cx, cy, cv, cov_func,
                                total_sill, mean=0.0)
        draw = est + np.sqrt(var) * rng.standard_normal()

        sim_values[idx] = draw
        cond_x.append(x0)
        cond_y.append(y0)
        cond_v.append(draw)

    return sim_values


def _run_one_realisation(payload):
    """Top-level (picklable) worker: unpack args and run one realisation.
    Used both for the sequential loop and for ProcessPoolExecutor workers."""
    (grid_x, grid_y, data_x, data_y, data_g, cov_func, total_sill,
     max_neighbors, search_radius, seed_seq) = payload
    rng = np.random.default_rng(seed_seq)
    return sgs_realisation(grid_x, grid_y, data_x, data_y, data_g,
                            cov_func, total_sill,
                            max_neighbors=max_neighbors,
                            search_radius=search_radius, rng=rng)


def resolve_n_jobs(n_jobs):
    """1 -> sequential. -1 -> all cores. N>1 -> N worker processes."""
    cpu = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, cpu + n_jobs + 1)
    return min(n_jobs, cpu)


def run_sgs(df, x=None, y=None, z=None,
            n_realisations=20, grid_res=None,
            n_grid_target=60, max_neighbors=24,
            search_radius=None, n_lags=11,
            variogram_model="spherical",
            variogram_fit=True,
            seed=None, n_jobs=1, bounds=None):
    """Full SGS pipeline on a (Geo)DataFrame. Returns a dict with grid
    arrays, all realisations, E-type mean/std, and diagnostic info.

    x, y : column names holding metric coordinates. If omitted, coordinates
        are taken from df.geometry (auto-reprojected to an estimated local
        UTM CRS if the current CRS is geographic, i.e. in degrees).
    z : column name of the value to interpolate (required).
    n_jobs : realisations to compute in parallel (1 = sequential, -1 = all
        cores, N = N processes). Does not change the result, only speed.
    """
    if z is None:
        raise ValueError("z (the column to interpolate) must be given")

    xs, ys, x_label, y_label = _get_xy(df, x, y)
    v = df[z].to_numpy(dtype=float)

    nst = NormalScoreTransform(v)
    g = nst.forward(v)

    lags, gamma, counts = experimental_variogram(xs, ys, g, n_lags=n_lags)
    vparams = fit_variogram(lags, gamma, model=variogram_model)
    cov_func, total_sill = covariance_function(vparams)

    pad = 0.05 * max(xs.max() - xs.min(), 
                     ys.max() - ys.min(), 
                     1.0)

    # round all values to nearest .25
    pad  = np.round(pad * 4) / 4
    xmin = np.floor(xs.min() * 4) / 4 - pad
    ymin = np.floor(ys.min() * 4) / 4 - pad
    xmax = np.ceil(xs.max() * 4)  / 4 + pad
    ymax = np.ceil(ys.max() * 4)  / 4 + pad

    #ys.min() - pad, ys.max() + pad
    if grid_res is None:
        grid_res = max(xmax - xmin, ymax - ymin) / n_grid_target
    gx     = np.arange(xmin, xmax, grid_res)
    gy     = np.arange(ymin, ymax, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    grid_x = GX.ravel()
    grid_y = GY.ravel()

    # Spawning child seeds from a single SeedSequence guarantees the same
    # set of realisations regardless of n_jobs (sequential or parallel).
    child_seeds = np.random.SeedSequence(seed).spawn(n_realisations)
    tasks = [
        (grid_x, grid_y, xs, ys, g, cov_func, total_sill,
         max_neighbors, search_radius, cs)
        for cs in child_seeds
    ]

    workers = resolve_n_jobs(n_jobs)
    if workers <= 1:
        sim_g_list = [_run_one_realisation(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            sim_g_list = list(ex.map(_run_one_realisation, tasks))

    realisations = np.array([nst.backward(sg) for sg in sim_g_list])

    etype_mean = realisations.mean(axis=0).reshape(GX.shape)
    etype_std = realisations.std(axis=0).reshape(GX.shape)

    return {
        "GX": GX, "GY": GY,
        "realisations": realisations.reshape(n_realisations, *GX.shape),
        "etype_mean": etype_mean, "etype_std": etype_std,
        "variogram_params": vparams,
        "exp_variogram": (lags, gamma, counts),
        "data_x": xs, "data_y": ys, "data_v": v,
        "x_label": x_label, "y_label": y_label, "z_label": z,
    }


# --------------------------------------------------------------------------
# Coordinate handling — works whether the file has explicit x/y columns
# or only a geometry column (possibly in lon/lat degrees)
# --------------------------------------------------------------------------

def _get_xy(gdf, x, y):
    """Return (xs, ys, x_label, y_label) as plain metric coordinates.

    If x and y are given and present as columns, use them directly.
    Otherwise fall back to df.geometry, reprojecting to an estimated
    local UTM CRS first if the current CRS is geographic (degrees)."""
    if (x is not None) and (y is not None) \
       and (x in gdf.columns) and (y in gdf.columns):
        return (gdf[x].to_numpy(dtype=float),
                gdf[y].to_numpy(dtype=float),
                x, y)

    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.geometry is None:
        raise ValueError(
            "No usable coordinates: x/y columns not found and no geometry "
            "column is present. Pass --x/--y matching actual column names."
        )

    label = ""
    if gdf.crs is not None and gdf.crs.is_geographic:
        utm_crs = gdf.estimate_utm_crs()
        gdf     = gdf.to_crs(utm_crs)
        label   = f"{utm_crs.name} "
    return (gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(),
            f"{label}X (m)", f"{label}Y (m)")


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

def plot_variogram(lags, gamma, params, outpath):
    func   = MODELS[params.model]
    h      = np.linspace(0, lags.max() * 1.1, 200)
    fitted = func(h, params.nugget, params.sill, params.range_)
    plt.figure(figsize=(6, 4))
    plt.scatter(lags, gamma, label="experimental")
    plt.plot(h, fitted, "r-", label=f"{params.model} fit")
    plt.xlabel("lag distance")
    plt.ylabel("semivariance")
    plt.title(f"Variogram (nugget={params.nugget:.3g}, sill={params.sill:.3g}, "
              f"range={params.range_:.1f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_data(x, y, v, outpath, title, x_label, y_label, z_label):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=v, cmap="viridis", s=25,
                     edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, label=z_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_field(GX, GY, field, x, y, outpath, title,
               x_label, y_label, z_label,
               cmap="viridis", log=False):
    plt.figure(figsize=(6, 5))
    plot_vals = np.log10(np.clip(field, 1e-6, None)) if log else field
    im        = plt.pcolormesh(GX, GY, plot_vals, shading="auto", cmap=cmap)
    plt.scatter(x, y, edgecolor="k", facecolor='none', 
                s=8, alpha=0.5, lw=.8, label="data")
    plt.colorbar(im, label=("log10 " if log else "") + z_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axis("equal")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _slug(text):
    """Turn an arbitrary column/group value into a safe filename fragment."""
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text).strip("_")
    return text or "value"


def save_outputs(results, outdir, prefix, n_realisations):
    vp = results["variogram_params"]
    print(f"  Fitted variogram ({vp.model}): nugget={vp.nugget:.4g}, "
          f"sill={vp.sill:.4g}, range={vp.range_:.1f}")

    x_label, y_label, z_label = results["x_label"], results["y_label"], \
        results["z_label"]
    lags, gamma, _ = results["exp_variogram"]

    plot_variogram(lags, gamma, vp, outdir / f"{prefix}_variogram.png")
    plot_data(results["data_x"], results["data_y"], results["data_v"],
              outdir / f"{prefix}_data.png", f"{prefix} — sample data",
              x_label, y_label, z_label)
    plot_field(results["GX"], results["GY"], results["realisations"][0],
               results["data_x"], results["data_y"],
               outdir / f"{prefix}_realisation_1.png",
               f"{prefix} — SGS realisation no 1", x_label, y_label, z_label)
    plot_field(results["GX"], results["GY"], results["etype_mean"],
               results["data_x"], results["data_y"],
               outdir / f"{prefix}_etype_mean.png",
               f"{prefix} — E-type mean ({n_realisations} realisations)",
               x_label, y_label, z_label)
    plot_field(results["GX"], results["GY"], results["etype_std"],
               results["data_x"], results["data_y"],
               outdir / f"{prefix}_etype_std.png",
               f"{prefix} — E-type std (uncertainty)",
               x_label, y_label, z_label, cmap="magma")

    np.savez(
        outdir / f"{prefix}_grids.npz",
        X=results["GX"], Y=results["GY"],
        realisations=results["realisations"],
        etype_mean=results["etype_mean"],
        etype_std=results["etype_std"],
        data_x=results["data_x"],
        data_y=results["data_y"],
        data_v=results["data_v"],
    )

    print(f"  Saved outputs to {outdir}/{prefix}_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputfile",
                    help="path to any geopandas-readable input file "
                         "(GeoJSON, Shapefile, GeoPackage, ...)")
    ap.add_argument("-o", "--outdir",
                    default="results", help="output directory")
    ap.add_argument("-c", "--data-col", type=str, required=True,
                    help="name of the column to interpolate")
    ap.add_argument("--group-by", type=str, default=None,
                    help="optional column; run one independent SGS per "
                         "distinct value (e.g. 'year') instead of a single "
                         "run over the whole dataset")
    ap.add_argument("-n", "--n-realisations", type=int, default=20)
    ap.add_argument("-b", "--bounds", type=str, default=None,
                    help="location of file containing interpolation bounds")
    ap.add_argument("-p", "--prefix", type=str, default="",
                    help="prefix for figure and saved date file names")
    ap.add_argument("-g", "--grid-res", type=float, default=None,
                     help=("grid cell size in the data's coordinate units "
                           "(default: auto, ~60 cells across)"))
    ap.add_argument("-m", "--max-neighbors", type=int, default=24,
                     help="max conditioning points used per kriging system")
    ap.add_argument("-r", "--search-radius", type=float, default=None,
                    help="optional max distance for conditioning neighbors")
    ap.add_argument("-l", "--n-lags", type=int, default=12)
    ap.add_argument("-v", "--variogram-model", choices=list(MODELS),
                    default="spherical")
    ap.add_argument("-f", "--variogram-fit", default=None,
                    help="path to fitted variogram parameters")
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("-j", "--n-jobs", type=int, default=1,
                    help="realisations to run in parallel: 1 = sequential "
                         "(default), -1 = all CPU cores, N = N processes")
    ap.add_argument("-x", default=None, type=str,
                    help="horizontal coordinate column (default: derive "
                         "from geometry)")
    ap.add_argument("-y", default=None, type=str,
                    help="vertical coordinate column (default: derive "
                         "from geometry)")
    ap.add_argument("--min-points", type=int, default=20,
                    help="skip a group if it has fewer valid points than this")

    args = ap.parse_args()

    print(f"\nInput file: {args.inputfile}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bounds = args.bounds
    if bounds is not None:
        bounds = Path(args.bounds)
        if bounds.is_file():
            bounds = gpd.read_file(bounds).to_crs().geometry.bounds
    replacements = {'^': '', ']': '', '[': '', ' ': '_', 
                    '/': '_', '2': '2', '_': '_', '%': '%'}
    postfix = args.data_col
    prefix  = args.prefix + '_' + ''.join([replacements.get(c, c)
                                           for c in postfix])
    print(prefix)

    inputfile = Path(args.inputfile)
    if inputfile.suffix == '.csv':
        gdf = pd.read_csv(inputfile)  ## Plain pandas faster for csv files
    else:
        gdf = gpd.read_file(inputfile)
    if args.data_col not in gdf.columns:
        numeric_cols = [
            c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
        raise SystemExit(
            f"Column {args.data_col!r} not found. Numeric columns available: "
            f"{numeric_cols}"
        )
    gdf = gdf.dropna(subset=[args.data_col]).reset_index(drop=True)
    print(f"Loaded {len(gdf)} valid rows for column {args.data_col!r}")

    common = dict(
        x=args.x, y=args.y, z=args.data_col,
        n_realisations=args.n_realisations,
        grid_res=args.grid_res,
        max_neighbors=args.max_neighbors,
        search_radius=args.search_radius,
        n_lags=args.n_lags,
        variogram_model=args.variogram_model,
        variogram_fit=args.variogram_fit,
        seed=args.seed,
        n_jobs=args.n_jobs,
        bounds=bounds,
    )

    if args.group_by is None:
        results = run_sgs(gdf, **common)
        if prefix is None:
            prefix = _slug(args.data_col)
        save_outputs(results, outdir, prefix, args.n_realisations)
    else:
        if args.group_by not in gdf.columns:
            raise SystemExit(f"--group-by column {args.group_by!r} not found")
        for group_val, df_group in gdf.groupby(args.group_by):
            df_group = df_group.reset_index(drop=True)
            n = len(df_group)
            print(f"\n=== {args.group_by} = {group_val}: {n} points ===")
            if n < args.min_points:
                print((f"  Skipping: fewer than " 
                       f"--min-points ({args.min_points})."))
                continue
            results = run_sgs(df_group, **common)
            if prefix is None:
                prefix = f"{_slug(args.data_col)}_{_slug(group_val)}"
            save_outputs(results, outdir, prefix, args.n_realisations)

    print("\nDone.")

