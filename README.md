# euclid-mocks: Euclid Q1-like mock images of TNG50 galaxies

This package turns IllustrisTNG subhaloes into Euclid Q1-like multi-band cutouts. It produces VIS, NIR-Y, NIR-J and NIR-H
science images plus detection maps, all on a single pixel grid. Each cutout includes the Q1 dither pattern, detector
noise (VIS photon noise and the NISP MACC up-the-ramp readout), real (or mock) MER background, and the Euclid PSF.
flux calibration values should be checked and possibly amended for changes in the DR1 release.
Applicable tiles should always be queried with astroquery for their datalabs path to check if they are valid files before using them.
This code was inspired by the lightcones created in https://github.com/gsnyder206/mock-surveys/releases/tag/v1.0.0 and relies on Values from multiple publications, such as Kubik+2016, Jahnke+2024, Scaramella+2022, the VIS and NIS PF papers from McCraken+2025 & Polenta+2025, Nelson+2019a and Plank+2015.
CHANGE THE `euclid_mer_dir` in `config/default.yaml` to the path holding the Euclid data in your system.

## Install

```bash
conda env create -f environment.yml      # creates "euclid-mocks", installs this package in editable mode
conda activate euclid-mocks
```

drizzle 2.x comes from pip because it is not on conda-forge. The pipeline uses its `resample`/`calc_pixmap` API only.

## Quick start (offline, no credentials)

```bash
python scripts/make_mock_tile.py         # writes data/mock_tile/ (≈21 MB, deterministic, ~10 s)
jupyter lab notebooks/mock_imaging_overview.ipynb
```

The notebook builds the mock tile if it is missing. It has four parts:
- **A** (offline): pushes a synthetic interacting galaxy pair through the full instrument chain and writes
  `output/ScienceReady/synthetic_sub-0_snap-99_...fits`.
- **B**: one real TNG galaxy (needs credentials).
- **C**: two snapshots × two galaxies, then reads and plots the results with fitsbolt (needs credentials).
- **D**: builds background-position catalogues from a list of tile IDs.

## Real TNG galaxies

1. Copy `config/tng_credentials.example.yaml` to `config/tng_credentials.yaml` and enter your
   [TNG API key](https://www.tng-project.org/users/profile/). That file is gitignored.
2. Run one galaxy, or the full batch:

```bash
python scripts/run_single.py --sub-id 123456 --snapshot 67
python scripts/run_mock_imaging.py                       # all snapshots in cfg.run.snapshots
python scripts/run_mock_imaging.py --snapshots 53 52     # a subset
```

The batch builds the merger catalogue of each snapshot from the Eisert et al. (2023) merger histories, samples major,
minor and non-merging galaxies, and processes them in a process pool. It writes one CSV per snapshot with `status` ∈
{created, existing, rejected, failed}. Finished galaxies are detected and skipped on a rerun.

### Render timeout fallback

`render.method: sphMap` (the default) renders every particle of the subhalo's parent FoF group, so merging
companions stay visible. For galaxies in very massive groups (e.g. satellites of the most massive TNG50 cluster) the
render server's gateway gives up first and answers HTTP 504. With

```yaml
render:
  timeout_fallback:
    enabled: true            # set false to record such galaxies as failed instead
    method: sphMap_subhalo
```

such a galaxy is re-rendered in **all four bands** with `sphMap_subhalo`, which uses only the subhalo's own
particles. Companions that are separate subhaloes are then missing. Every galaxy is flagged in the summary CSV column
`render_method` and in the header card `M_METH`, so fallback mocks can be filtered out or analysed separately. When
the fallback is enabled, `sphMap` is tried once (no retry), because these timeouts are deterministic.
Every retry, and the switch to the fallback, waits `tng.retry_wait_s` (default 5 s). A timed-out render can keep the
server busy for a while, and immediate follow-up requests then tend to fail with 403 or 504. The `error` column of the
summary CSV lists the outcome of every attempt, e.g.
`... [attempt 1/2: HTTP 403 Forbidden; attempt 2/2: HTTP 504 Gateway Time-out]`.

## Configuration

Everything tunable lives in [config/default.yaml](config/default.yaml) and is read as a DotMap (`cfg.render.n_pixels`).
Relative paths resolve against the YAML file's directory. To use the real Q1 data, point these at it:

```yaml
paths:
  euclid_mer_dir: /path/to/Q1_R1/MER                 # {tile}/VIS|NISP/EUC_MER_{BGSUB-MOSAIC,GRID-PSF}-*.fits
  empty_region_dir: /path/to/EmptyRegionsCatalog     # TILE_{tile}_empty_regions.csv
```

### Cutout size

There is a size multiplier(passed as `size` with
`sizeType=rHalfMassStars`). It now lives in the config:

```yaml
render:
  size_in_half_mass_radii: 22.5   # render side = 22.5 x stellar half-mass radius
  n_pixels: 500                   # render resolution, independent of the final pixel scale
```

**Why so large.** 22.5 r½ means about ±11 half-mass radii around the galaxy. That is enough to include tidal
features and nearby companions of systems up to ±2 Gyr from a merger, not just the stellar body, and to leave sky
around the source. Earlier runs used smaller fields.

**How the render size becomes the cutout size** (`source.render_size_arcmin`, `bands.observe_vis`):
1. The side is 22.5 × `halfmassrad_stars` (ckpc/h), converted to proper kpc (÷ h, ÷ (1+z)) and then to arcmin with
   the TNG cosmology. It is rounded to 0.001′ and requested from the API with `sizeType=arcmin`, so all four bands
   cover exactly the same sky.
2. After the VIS exposures and stacking, the image is cropped to the square bounding box of its non-zero pixels
   (2 px brim removed, `detector.vis_brim_trim_px`) and cut by one more pixel per side. That defines the VIS grid,
   and every other band and the DET maps are put on it. The final side is therefore at most about render side / 0.1″
   VIS pixels, and smaller when the faint outskirts receive no photons.
3. The background stage needs an empty-region position whose distance to the tile border exceeds 0.52 × the cutout
   side (`background.border_size_factor`). That is no problem for 19200² px Q1 tiles, but the 1024² px mock tile
   supports cutouts up to about 980 px.

Typical render sides (arcmin / VIS pixels):

| z | r½ = 1 ckpc/h | r½ = 3 ckpc/h | r½ = 8 ckpc/h |
|---|---|---|---|
| 0.2 | 0.135′ / 81 px | 0.406′ / 244 px | 1.084′ / 650 px |
| 0.5 | 0.059′ / 35 px | 0.176′ / 106 px | 0.470′ / 282 px |
| 1.0 | 0.034′ / 20 px | 0.101′ / 61 px | 0.269′ / 161 px |
| 1.5 | 0.025′ / 15 px | 0.076′ / 46 px | 0.204′ / 122 px |

Changing `size_in_half_mass_radii` changes the cutout size of every galaxy, and with it which galaxies pass the
minimum-size cut below.

### Minimum on-sky size: small galaxies are rejected

```yaml
selection:
  min_size_arcmin: 0.024   # 1.5 x 0.016 arcmin = 1.44" = ~14 VIS pixels for the whole render
```

A galaxy whose **render side** (22.5 r½ on the sky, not the galaxy itself) is below 0.024′ is rejected before any
render is requested (`status = rejected`, `error = "render too small: ..."`). Equivalently, its stellar half-mass
radius must be at least about 0.064″ on the sky (0.64 VIS pixels). In physical terms, the smallest r½ that is kept:

| z | 0.2 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|
| min r½ [ckpc/h] | 0.18 | 0.41 | 0.71 | 0.94 |

This cut therefore removes compact galaxies preferentially at **high redshift**, which biases the high-z sample
towards larger galaxies. Keep this in mind when comparing merger fractions across redshift. The other selection cuts
are listed in `selection` in the config: at least 10 star particles, M★ ≥ 3 × 10⁶ M☉, and a brightest pixel brighter
than 28 mag/arcsec². The merger-catalogue stage additionally requires `halfmassrad_stars > 0.3` ckpc/h and
log M★ ≥ 9.5.

### Background positions for your own tiles

The background pool is one CSV of empty sky positions per MER tile. Build them from the tiles' segmentation maps
(`paths.mer_segmap_dir/{tile}/EUC_MER_FINAL-SEGMAP*.fits`):

```python
from pathlib import Path
from euclid_mocks.config import load_config
from euclid_mocks.empty_regions import build_empty_region_catalogs

cfg = load_config()
positions = build_empty_region_catalogs(cfg, [102018211, 102159776], cfg.empty_regions, Path("EmptyRegionsCatalog"))
```

Then set `paths.empty_region_dir` to that folder. Notebook Part D shows this on the mock tile.

> **DR1 deep fields:** a tile folder there can contain several images of the same kind (different stacks or
> versions), so the MER glob patterns can match more than one file. The package uses the alphabetically first match
> and logs a warning listing all candidates. Check these warnings, and make sure the mosaic, PSF grid and segmap
> belong together.

Fixed instrument and simulation numbers (gains, exposure times, zeropoints, IS factors, dither pattern, TNG cosmology)
are in [euclid_mocks/constants.py](euclid_mocks/constants.py).

## Output

`output/ScienceReady/{merger_type}_sub-{id}_snap-{snap}_view-{axes}_BKGPSF_{tile}_{index}.fits` contains:

| Extensions | Content |
|---|---|
| `VIS`, `NIR-Y`, `NIR-J`, `NIR-H` | PSF-convolved mock plus background, in Jy/pixel (`BUNIT=Jy`) |
| `VISDET`, `NIR-YDET`, ... | pre-PSF mock divided by the roughly estimated avarage background noise |

The summary CSV per snapshot has one row per sampled galaxy: masses, merger times, `hdu_path`, `snr` ("[vis,y,j,h]"),
`render_method`, `status` and `error`. Notebook Part C reads and plots these files with
[fitsbolt](https://pypi.org/project/fitsbolt/) (`load_and_process_images` with an asinh normalisation, NIR-H/NIR-Y/VIS
as RGB).

All 8 extensions share the shape and WCS of `VIS`, and this is checked before writing. Header cards: `M_*` record how
the mock was made (render size, view, merger class and times), `BKG_*` record the background patch, and `M_DETLVL`
holds the central 8×8 signal-to-noise.

## Layout

```
config/          default.yaml, tng_credentials.example.yaml
euclid_mocks/    tng_api → source → resample → detector → bands → background → science_ready → pipeline
                 merger_catalog, empty_regions, mock_tile
scripts/         make_mock_tile.py, run_single.py, run_mock_imaging.py
notebooks/       mock_imaging_overview.ipynb
tests/           test_end_to_end.py (geometry of all extensions on a mock tile)
```

## Tests

```bash
pytest          # about 20 s; builds a 512² mock tile in a temp dir
```

## Notes

- photutils warns `alpha + beta > 1.0` for VIS. These are the original window parameters (0.8 / 0.75), kept on purpose.
- Uncovered pixels in a real MER background stay 0 in the science images.
- After 5 failed background-quality redraws, the last background is used and a warning is logged.