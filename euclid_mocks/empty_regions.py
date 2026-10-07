"""Find source-free sky patches in a MER segmentation map.

Port of ``process_segmap`` from ``CosmicCuddlesV1.ipynb``, which produced the
Q1 ``EmptyRegionsCatalog``. The resulting CSVs are what
:func:`euclid_mocks.background.choose_background` draws positions from.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from dotmap import DotMap
from loguru import logger
from scipy.ndimage import binary_closing, distance_transform_edt
from scipy.ndimage import label as ndi_label
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

from euclid_mocks.background import first_match


def find_empty_regions(
    segmap: np.ndarray,
    wcs: WCS,
    tile: int,
    min_empty_pixels_area: int,
    min_object_distance: float,
    min_border_distance: float,
    peak_min_distance: int,
    keep_segment_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return roughly circular empty patches of a segmentation map.

    Empty pixels (segmap 0) are distance-transformed; local maxima of the
    distance seed a watershed that splits the empty area into round patches.
    Each patch is kept if it is large enough, far enough from the tile border
    and far enough from the nearest object, and its centroid becomes one
    catalogue row.

    Args:
        segmap (np.ndarray): Segmentation map (0 = sky); NaN counts as an object.
        wcs (WCS): WCS of the segmap.
        tile (int): Tile ID written to ``Folder_name``.
        min_empty_pixels_area (int): Smallest patch area in pixels.
        min_object_distance (float): Smallest centroid-to-object distance in pixels.
        min_border_distance (float): Smallest centroid-to-border distance in pixels.
        peak_min_distance (int): Smallest separation of patch seeds in pixels.
        keep_segment_ids (np.ndarray | None): If given, only these segments count as
            objects (e.g. after removing small or spurious catalogue sources).

    Returns:
        pd.DataFrame: Columns ``Folder_name, pos_x, pos_y, pos_ra, pos_dec,
        min_object_distance, border_distance``.
    """
    segmap = np.nan_to_num(np.array(segmap, dtype=np.float64), nan=1)
    if keep_segment_ids is not None:
        segmap[~np.isin(segmap, keep_segment_ids)] = 0
    height, width = segmap.shape

    empty = segmap == 0
    distance = distance_transform_edt(empty)
    seeds = peak_local_max(
        distance,
        footprint=np.ones((5, 5)),
        min_distance=peak_min_distance,
        exclude_border=int(min_border_distance),
        labels=empty.astype(np.int32),
    )
    seed_mask = np.zeros(distance.shape, dtype=bool)
    seed_mask[tuple(seeds.T)] = True
    markers, _ = ndi_label(seed_mask)
    patches = watershed(-distance, markers, mask=empty)

    # Closing merges object pixels into blobs so the distance is to an object, not to a noise speck.
    object_pixels = np.array(np.nonzero(binary_closing(patches == 0, structure=np.ones((5, 5))))).T
    tree = cKDTree(object_pixels)

    rows = []
    for region in regionprops(patches):
        if region.area < min_empty_pixels_area:
            continue
        cy, cx = region.centroid
        border_distance = min(cx, width - 1 - cx, cy, height - 1 - cy)
        if border_distance < min_border_distance:
            continue
        object_distance, _ = tree.query([cy, cx], k=1)
        if object_distance < min_object_distance:
            continue
        ra, dec = wcs.all_pix2world(cx, cy, 0)
        rows.append(
            {
                "Folder_name": tile,
                "pos_x": cx,
                "pos_y": cy,
                "pos_ra": float(ra),
                "pos_dec": float(dec),
                "min_object_distance": int(object_distance),
                "border_distance": int(border_distance),
            }
        )
    logger.info(f"tile {tile}: {len(rows)} empty regions from {len(seeds)} seeds")
    return pd.DataFrame(rows)


def write_empty_region_catalog(df: pd.DataFrame, out_dir: Path, tile: int) -> Path:
    """Write an empty-region catalogue in the layout the background stage expects.

    The file name must have the tile ID as its second ``_``-separated token,
    and the pandas index is written as the unnamed first column.

    Args:
        df (pd.DataFrame): Output of :func:`find_empty_regions`.
        out_dir (Path): Catalogue folder (``cfg.paths.empty_region_dir``).
        tile (int): Tile ID.

    Returns:
        Path: Written CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"TILE_{tile}_empty_regions.csv"
    df.to_csv(path, index=True)
    return path


def build_empty_region_catalogs(
    cfg: DotMap, tile_ids: list[int], params: DotMap, out_dir: Path
) -> pd.DataFrame:
    """Create the empty-region (background position) catalogues of a list of MER tiles.

    For each tile the segmentation map ``{mer_segmap_dir}/{tile}/EUC_MER_FINAL-SEGMAP*.fits``
    is searched for empty patches with :func:`find_empty_regions` and written
    to ``out_dir/TILE_{tile}_empty_regions.csv``. Point ``paths.empty_region_dir``
    at ``out_dir`` to use the result as the background pool of the pipeline.

    Args:
        cfg (DotMap): Run configuration (uses ``paths.mer_segmap_dir``).
        tile_ids (list[int]): MER tile IDs.
        params (DotMap): Search parameters, e.g. ``cfg.empty_regions``.
        out_dir (Path): Folder for the CSV catalogues.

    Returns:
        pd.DataFrame: All positions of all tiles.
    """
    catalogs = []
    for tile in tile_ids:
        segmap_path = first_match(f"{cfg.paths.mer_segmap_dir}/{tile}/EUC_MER_FINAL-SEGMAP*.fits")
        with fits.open(segmap_path) as hdul:
            segmap, wcs = hdul[0].data, WCS(hdul[0].header)
        regions = find_empty_regions(
            segmap,
            wcs,
            tile,
            min_empty_pixels_area=params.min_empty_pixels_area,
            min_object_distance=params.min_object_distance_px,
            min_border_distance=params.min_border_distance_px,
            peak_min_distance=params.peak_min_distance_px,
        )
        write_empty_region_catalog(regions, Path(out_dir), tile)
        catalogs.append(regions)
    return pd.concat(catalogs, ignore_index=True)
