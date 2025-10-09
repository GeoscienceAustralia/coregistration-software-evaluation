import glob
import os
import asyncio
import json
import rasterio.coords
import requests
import rasterio
from rasterio.plot import show
from zipfile import ZipFile
import pandas as pd
import shutil
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject
from rasterio.fill import fillnodata
import numpy as np
import imageio
import cv2 as cv
from pyproj import Proj, Transformer
from rasterio.coords import BoundingBox
from typing import Union
from itertools import product as itrprod
import re
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.exposure import equalize_hist, rescale_intensity
import itertools
from pykml import parser
from collections import namedtuple
import utm as utm_convrter
import boto3
from pystac_client import Client
from osgeo import ogr
from shapely import ops, from_wkt, to_geojson, Polygon, box
from pathlib import Path
from typing import Literal
from datetime import datetime
from datetime import timedelta
import time
import asyncio
from subprocess import run
import shlex
from cv2 import Sobel, Laplacian, Canny
import PIL
import rioxarray as rxr
import xarray as xr
import pystac
from typing import Callable, Any
import warnings
from sklearn.cluster import DBSCAN
import multiprocess as mp
from geojson import loads as gloads
from dask import optimize
import copy


dbscan = DBSCAN(min_samples=2, eps=0.01)

try:
    from arosics import COREG_LOCAL
except ImportError:
    print("AROSICS not installed. `arosics` function will not work.")

PIL.Image.MAX_IMAGE_PIXELS = None


def get_sentinel_filenames(
    polygon_list: list[str],
    years: list[str],
    products: list[str] = ["GRD", "SLC"],
    is_poly_bbox: bool = True,
    satellite: str = "S1",
    instrument: str = "C-SAR",
    identifier: str = "",
    output_file_path: str = "",
    includes: list[str] = [],
    return_query_only: bool = False,
):
    """
    Get scene names for S1 via direct requet to SARA server
    """
    s_names = []
    if os.path.isfile(output_file_path):
        with open(output_file_path, "r") as f:
            for l in f:
                s_names.append(l.strip())
    else:
        start = f"{years[0]}-01-01"
        end = f"{years[1]}-12-31"
        query = f"https://copernicus.nci.org.au/sara.server/1.0/api/collections/{satellite}/search.json?_pretty=1&startDate={start}&completionDate={end}&instrument={instrument}&maxRecords=500"

        if satellite == "S1":
            query += "&sensor=IW"
        else:
            products = []

        if identifier != "":
            query += f"&identifier={identifier}"

        if len(polygon_list) == 0:
            polygon_list = [""]
        if len(products) == 0:
            products = [""]

        for poly, prod in list(itrprod(polygon_list, products)):
            if poly == "":
                poly_query = ""
            else:
                poly_query = f'&{"box" if is_poly_bbox else "geometry"}={poly}'

            if prod == "":
                prod_query = ""
            else:
                prod_query = f"&productType={prod}"

            no_page_query = query + poly_query + prod_query

            page = 1
            query_resp = ["start"]
            while query_resp != []:
                to_query = no_page_query + f"&page={page}"
                if return_query_only:
                    s_names.append(to_query)
                else:
                    print(f"Querying: {to_query}", end="\r")
                    response = json.loads(requests.get(to_query).content)
                    if "features" not in response:
                        query_resp = []
                        continue
                    query_resp = [
                        r["properties"]["title"] for r in response["features"]
                    ]
                    s_names.extend(query_resp)
                if return_query_only:
                    break
                page += 1

        if (return_query_only) and (len(includes) != 0):
            print("Filtering the outputs does not work in return query only mode.")
        else:
            filtered_list = []
            for p in includes:
                filtered_list.extend(
                    list(filter(lambda el: len(re.findall(p, el)) != 0, s_names))
                )
            s_names = list(set(filtered_list))

        if output_file_path != "":
            with open(output_file_path, "w") as f:
                for n in s_names:
                    f.write(f"{n}\n")
    return s_names


def save_file_list(file_list: dict, save_path: str) -> None:
    """
    Saves the retrieved data.
    """
    with open(save_path, "w") as f:
        for k, v in file_list.items():
            for filename in v:
                f.write(f"{k},{filename}\n")
    return None


async def find_all_files_for_case(query_case: tuple, sat_data_dir: str) -> bool:
    """
    Finds all files for a selected case of product/year/month
    """
    case_path = os.path.join(
        sat_data_dir, query_case[0], query_case[1], f"{query_case[1]}-{query_case[2]}"
    )
    print(f"Retrieving files for {case_path}", end="\r")
    return glob.glob(case_path + "/*/*.zip")


async def find_aoi_files(aoi: str, all_files: list[str]) -> list[str]:
    """
    Filters all files and finds files for the area of interest.
    """
    print(f"filtering files for {aoi}", end="\r")
    return list(filter(lambda p: len(re.findall(aoi, p)) != 0, all_files))


def flatten(l: list[list]) -> list:
    """
    Flattens the list
    """
    return [x for xs in l for x in xs]


# Not sure if async runs well in notebook
async def find_files_for_aios_async(
    query_cases: list[tuple],
    sat_data_dir: str,
    aoi_list: list[str],
) -> dict:
    """
    Asyncronously finds the files for an AOI list given as list of identifiers based on a combination of produt/year/month from NCI Copernicus databse.
    Set `is_s1` to True for Sentinel-1.
    """
    all_files_async = [find_all_files_for_case(c, sat_data_dir) for c in query_cases]
    all_files = await asyncio.gather(*all_files_async)
    all_files = flatten(all_files)
    print("")

    aoi_files_async = [find_aoi_files(aoi, all_files) for aoi in aoi_list]
    aoi_files = await asyncio.gather(*aoi_files_async)
    print("")
    return dict(map(lambda k, v: (k, v), aoi_list, aoi_files))


# syncronous function for all cases and AOIs at the same time. Could take long
def find_files_for_aios(
    query_cases: list[tuple],
    sat_data_dir: str,
    aoi_list: list[str],
) -> dict:
    """
    Finds the files for an AOI list given as list of identifiers based on a combination of produt/year/month from NCI Copernicus databse.
    Set `is_s1` to True for Sentinel-1.
    """
    all_files = []
    aoi_files = []
    for c in query_cases:
        case_path = os.path.join(sat_data_dir, c[0], c[1], f"{c[1]}-{c[2]}")
        print("\r", f"Retrieving files for {case_path}", end="")
        all_files.extend(glob.glob(case_path + "/*/*.zip"))

    print("")
    aoi_files = {}
    for aoi in aoi_list:
        print("\r", f"filtering files for {aoi}", end="")
        aoi_files[aoi] = list(filter(lambda p: aoi in p, all_files))

    print("")
    return aoi_files


def find_files_for_s1_aois(nci_files_dict: dict, s1_file_names: list[str]) -> dict:
    """
    Finds the files found from SARA server inside the AOI files retrieved from NCI
    """
    files_dict = {}
    for k, v in nci_files_dict.items():
        nci_files = [os.path.splitext(os.path.basename(f))[0] for f in v]
        found_idx = [nci_files.index(f) for f in s1_file_names if f in nci_files]
        found = [v[idx] for idx in found_idx]
        files_dict[k] = found
    return files_dict


def find_polarisation_files_s1(dir: str) -> list[str]:
    """
    Finds Sentinel 1 data from the provided path.
    """
    return glob.glob(os.path.join(dir, "measurement", "*"))


def load_s1_scenes(
    zip_file_path: str,
    zip_file_id: str,
    subdir_name: str = "",
    remove_input: bool = True,
) -> tuple:
    """
    Loads S1 scenes from provided path and id
        * `zip_file_path` and `zip_file_ids` are the path and id of the data.
        * `subdir_name` will be added to the output directory path if provided
    """

    with ZipFile(zip_file_path) as f:
        f.extractall(f"./data/inputs/{zip_file_id}/{subdir_name}")
    data_dir = os.listdir(f"./data/inputs/{zip_file_id}/{subdir_name}")[0]
    dir = os.path.join(f"./data/inputs/{zip_file_id}/{subdir_name}", data_dir)
    band_files = find_polarisation_files_s1(dir)
    scenes = [rasterio.open(band_file) for band_file in band_files]

    if remove_input:
        shutil.rmtree(f"./data/inputs/{zip_file_id}/{subdir_name}", ignore_errors=True)

    return scenes


def transform_s1_data(
    scenes,
    scale_factor=0.03,
) -> tuple:
    """
    Downsamples a list of scenes and returns their new data and affine transformations.
    """
    new_data = []
    new_transforms = []
    names = []
    for scene in scenes:
        new_datum, new_transform = downsample_dataset(scene.name, scale_factor)
        new_data.append(new_datum)
        new_transforms.append(new_transform)
        names.append(os.path.splitext(os.path.basename(scene.name))[0])

    return new_data, new_transforms, names


def enhance_color_s1(data, is_slc: bool = True):
    """
    Enhances the generated data for better visualisation
    """
    if is_slc:
        amplitude = np.linalg.norm(data, axis=0)
        amplitude = 10 * np.log10(amplitude + 2.0e-10)
    else:
        amplitude = data / 256
    amplitude = amplitude.astype("float64")
    amplitude *= 255 / amplitude.max()
    return amplitude.astype("uint8")


def plot_scenes_s1(data, data_names, data_transforms, is_slc: bool = True):
    """
    Plots the data given the names of the scenes and their affine transformations
    """
    _, axes = plt.subplots(1, len(data), figsize=(10 * len(data), 10 * len(data)))
    if type(axes) != np.ndarray:
        axes = [axes]
    for i, d in enumerate(data):
        ax = axes[i]
        show(
            enhance_color_s1(d, is_slc),
            ax=ax,
            title=f"{data_names[i]}",
            transform=data_transforms[i],
        )
        ax.set_title(f"{data_names[i]}")
        ax.title.set_size(10)


def get_scenes_dict(
    data_df: pd.DataFrame, product: list[str] = [], is_s1: bool = True
) -> dict:
    scenes_dict = {}
    id_list = data_df.ID.unique()
    for id in id_list:
        filtered_df = data_df[data_df.ID == id].reset_index(drop=True)
        if product != []:
            for p in product:
                filtered_df = filtered_df[
                    filtered_df.Path.apply(lambda x: p in x)
                ].reset_index(drop=True)

        grouper = filtered_df.Path.apply(
            lambda r: os.path.split(r)[1].split("_")[5 if is_s1 else 2][0:6]
        )
        secene_list = [
            list(filtered_df.groupby(grouper))[i][1].Path.iloc[0]
            for i in range(0, len(grouper.unique()))
        ]
        scenes_dict[id] = secene_list
    return scenes_dict


def scale_transform(
    warp_matrix: np.ndarray,
    resolution_ratio_y: float,
    resolution_ratio_x: float,
    adjust_to_centre: bool = False,
) -> np.ndarray:
    """Rescales the warp matrix according to the provided resolution ratios.

    Parameters
    ----------
    warp_matrix : np.ndarray
        Warp matrix to be rescaled.
    resolution_ratio_y : float
        Resolution ratio for the y dimension.
    resolution_ratio_x : float
        Resolution ratio for the x dimension.
    adjust_to_centre : bool, optional
        Adjusts the warp matrix to the centre of the new pixel size if True.

    Returns
    -------
    np.ndarray
        New warp matrix with adjusted resolutions.
    """
    scaled_warp = warp_matrix.copy()
    scaled_warp[:, 0] /= resolution_ratio_x
    scaled_warp[:, 1] /= resolution_ratio_y
    if not adjust_to_centre:
        return scaled_warp
    scaled_warp[0, 2] += (1 - resolution_ratio_x) * abs(scaled_warp[0, 0]) / 2
    scaled_warp[1, 2] -= (1 - resolution_ratio_y) * abs(scaled_warp[1, 1]) / 2
    return scaled_warp


def readjust_origin_for_new_pixel_size(
    transform: rasterio.Affine,
    scale_factor_y: float,
    scale_factor_x: float,
    adjust_to_centre: bool = False,
) -> rasterio.Affine:
    """
    Readjusts the origin of a scene after resampling.

    Parameters
    ----------
    transform : rasterio.Affine
        The original affine transformation of the scene.
    scale_factor_y : float
        The scale factor for the y dimension.
    scale_factor_x : float
        The scale factor for the x dimension.
    adjust_to_centre : bool, optional
        If True, the origin will be adjusted to the centre of the new pixel size.
        If False, the origin will remain at the top-left corner. Defaults to False.

    Returns
    -------
    rasterio.Affine
        The new affine transformation with the adjusted origin.
    """
    if not adjust_to_centre:
        return transform
    new_orig_x = transform.c + ((1 - scale_factor_x) * abs(transform.a) / 2)
    new_orig_y = transform.f - ((1 - scale_factor_y) * abs(transform.e) / 2)
    return rasterio.Affine(
        transform.a, transform.b, new_orig_x, transform.d, transform.e, new_orig_y
    )


def resample_xarray_dataset(
    dataset: xr.Dataset | str,
    scale_factor: Union[float, list[float]] = 1.0,
    output_path: str = "",
) -> xr.Dataset:
    """
    Resamples an xarray dataset to a new resolution.
    Parameters
    ----------
    dataset : xr.Dataset | str
        The input dataset to be resampled.
    scale_factor : float | list[float], optional
        The scaling factor(s) for the resampling, by default 1.0 (no resampling).
    output_path : str, optional
        The path to save the resampled dataset, by default an empty string (no saving).
    """
    if isinstance(dataset, str):
        dataset = rxr.open_rasterio(dataset)

    if isinstance(scale_factor, float):
        scale_factor = [scale_factor, scale_factor]
    elif isinstance(scale_factor, list) and len(scale_factor) == 1:
        scale_factor = [scale_factor[0], scale_factor[0]]

    new_width = int(dataset.rio.width * scale_factor[1])
    new_height = int(dataset.rio.height * scale_factor[0])

    resampled_dataset = dataset.rio.reproject(
        dataset.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.bilinear,
    )

    if output_path:
        resampled_dataset.rio.to_raster(output_path)

    return resampled_dataset


def downsample_dataset(
    dataset_path: str,
    scale_factor: Union[float, list[float]] = 1.0,
    output_file: str = "",
    enhance_function=None,
    force_shape: tuple = (),  # (height, width)
    readjust_origin: bool = False,
    round_resolution: bool = False,
    masked_data: bool = False,
) -> tuple:
    """
    Downsamples the output data and returns the new downsampled data and its new affine transformation according to `scale_factor`
    The output shape could also be forced using `forced_shape` parameter.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset to be downsampled.
    scale_factor : float or list[float], optional
        Scale factor for downsampling. If a single float is provided, it will be applied to both dimensions (height, width).
        If a list of two floats is provided, they will be applied to the height and width respectively.
        Defaults to 1.0 (no downsampling).
    output_file : str, optional
        Path to save the downsampled dataset. If not provided, the data will not be saved to a file.
    enhance_function : callable, optional
        A function to enhance the data after downsampling. It should accept a numpy array and return a modified numpy array.
        Defaults to None (no enhancement).
    force_shape : tuple, optional
        A tuple specifying the desired output shape (height, width). If provided, the output data will be forced to this shape.
        If empty, the output shape will be calculated based on the scale factor.
    readjust_origin : bool, optional
        If True, the origin of the affine transformation will be readjusted after downsampling.
        Defaults to False.
    round_resolution : bool, optional
        If True, the pixel size in the affine transformation will be rounded to the nearest integer.
        Defaults to False.
    masked_data : bool, optional
        If True, reads the data as a masked array to handle nodata values. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing:
        - data: numpy array of the downsampled data.
        - transform: rasterio.Affine object representing the new affine transformation.
    """
    with rasterio.open(dataset_path) as dataset:
        # resample data to target shape
        if type(scale_factor) == float:
            scale_factor = [scale_factor] * 2
        if len(force_shape) != 0:
            output_shape = force_shape
        else:
            output_shape = (
                int(dataset.height * scale_factor[0]),
                int(dataset.width * scale_factor[1]),
            )
        data = dataset.read(
            out_shape=(
                dataset.count,
                *output_shape,
            ),
            resampling=Resampling.bilinear,
            masked=masked_data,
        )

        if enhance_function is not None:
            data = enhance_function(data)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )

        if round_resolution:
            transform = rasterio.Affine(
                np.round(transform.a).tolist(),
                transform.b,
                transform.c,
                transform.d,
                np.round(transform.e).tolist(),
                transform.f,
            )

        if len(force_shape) != 0:
            scale_factor = [
                force_shape[0] / scale_factor[0],
                force_shape[1] / scale_factor[1],
            ]

        if readjust_origin:
            transform = readjust_origin_for_new_pixel_size(transform, *scale_factor)

        profile = dataset.profile
        profile.update(
            transform=transform,
            width=data.shape[2],
            height=data.shape[1],
            dtype=data.dtype,
        )

    if output_file != "":
        with rasterio.open(output_file, "w", **profile) as ds:
            for i in range(0, profile["count"]):
                ds.write(data[i], i + 1)

    return data, transform


def enhance_color_matching(data, uint16: bool = False):
    """
    Increases the brightness of the output data
    """
    if uint16:
        data = data / 256
    data = data.astype("float64")
    data *= 255 / data.max()
    return data.astype("uint8")


def plot_matching(
    datasets, alpha: float = 0.65, plot_title: str = "", save_fig_path: str = ""
):
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    show(datasets[0][0], ax=axes[0, 0], title="Reference scene")
    show(datasets[0][0], ax=axes[1, 0], title="Reference scene")
    show(datasets[0][0], ax=axes[2, 0], title="Reference scene")

    show(datasets[1][0], ax=axes[0, 1], title="Target scene")
    show(datasets[2][0], ax=axes[1, 1], title="Global matching")
    show(datasets[3][0], ax=axes[2, 1], title="Local matching")

    show(
        ((datasets[0][0] * alpha) + (datasets[1][0] * (1 - alpha))).astype("uint8"),
        ax=axes[0, 2],
        title="Reference + Target ",
    )
    show(
        ((datasets[0][0] * alpha) + (datasets[2][0] * (1 - alpha))).astype("uint8"),
        ax=axes[1, 2],
        title="Reference + Global matching",
    )
    show(
        ((datasets[0][0] * alpha) + (datasets[3][0] * (1 - alpha))).astype("uint8"),
        ax=axes[2, 2],
        title="Reference + Local matching",
    )
    for ax in axes.ravel():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if plot_title != "":
        plt.suptitle(plot_title)

    if save_fig_path != "":
        plt.savefig(save_fig_path)

    plt.subplots_adjust(top=0.9)


def reproject_tif(
    src_path: Path | str,
    dst_path: Path | str,
    dst_crs: str,
    resampling: Resampling = Resampling.bilinear,
    verbose: bool = True,
) -> None:
    """Reprojects a raster file to a new coordinate reference system (CRS) and saves it to a new file.

    Parameters
    ----------
    src_path : : Path | str
        Path to the source raster file to be reprojected.
    dst_path : : Path | str
        Destination path where the reprojected raster file will be saved.
    dst_crs : : str
        Destination coordinate reference system in a format recognized by rasterio (e.g., "EPSG:4326").
    resampling : Resampling, optional
        Resampling method to use during reprojection. Defaults to Resampling.bilinear.
    verbose : bool, optional
        If True, print detailed information about the reprojection process. Defaults to True.
    """

    with rasterio.open(src_path) as src:
        if verbose:
            print(f"reprojecting from {src.crs} to {dst_crs}")
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        if verbose:
            print(f"saving - {dst_path}")
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )
    return None


flip_img = lambda img: np.moveaxis(img, 0, -1)


def adjust_resolutions(
    dataset_paths: list[str],
    output_paths: list[str],
    resampling_resolution: str = "lower",
) -> tuple:
    """
    Adjusts the resolutions for two or more datasets with different ones. Rounding errors might cause a slightly different output resolutions.

    Parameters
    ----------
    dataset_paths : list[str]
        List of paths to the datasets to be adjusted.
    output_paths : list[str]
        List of paths where the adjusted datasets will be saved.
    resampling_resolution : str, optional
        The resolution to which the datasets should be resampled if their resolutions are different.
        Can be either "lower" or "higher", by default "lower".
    """
    ps_x = []
    ps_y = []

    for ds in dataset_paths:
        raster = rasterio.open(ds)
        raster_px_size = abs(raster.profile["transform"].a)
        raster_py_size = abs(raster.profile["transform"].e)
        ps_x.append(raster_px_size)
        ps_y.append(raster_py_size)

    if resampling_resolution == "lower":
        ref_res_x = max(ps_x)
        ref_res_y = max(ps_y)
    else:
        ref_res_x = min(ps_x)
        ref_res_y = min(ps_y)

    scale_factors = []
    for sx, sy in zip(ps_x, ps_y):
        scale_factors.append(
            [
                sy / ref_res_y,
                sx / ref_res_x,
            ]
        )

    transforms = []
    for i, ds in enumerate(dataset_paths):
        transforms.append(downsample_dataset(ds, scale_factors[i], output_paths[i])[1])

    return [[abs(t.a), abs(t.e), t] for t in transforms], scale_factors


def find_overlap(
    dataset_1: str,
    dataset_2: str,
    return_images: bool = False,
    return_pixels: bool = False,
    resampling_resolution: str = "lower",
) -> tuple:
    """
    Crude overlap finder for two overlapping scenes. (finds the bounding box around the overlapping area.
    A better way is to straighten the images and then find the overlap and then revert the transform.)

    Parameters
    ----------
    dataset_1 : str
        Path to the first dataset.
    dataset_2 : str
        Path to the second dataset.
    return_images : bool, optional
        If True, returns the images of the overlapping area, by default False.
    return_pixels : bool, optional
        If True, returns the pixel coordinates of the overlapping area, by default False.
    resampling_resolution : str, optional
        The resolution to which the datasets should be resampled if their resolutions are different.
        Can be either "lower" or "higher", by default "lower".

    Returns
    -------
    tuple
        A tuple containing:
        - The bounding box of the overlapping area as a rasterio.coords.BoundingBox object.
        - If `return_images` is True, a tuple containing:
            - The mosaiced image as a numpy array.
            - The overlapping area of the mosaiced image.
            - The overlapping area of the first dataset.
            - The overlapping area of the second dataset.
        - Scale factors for each dataset if their resolutions were adjusted.
    """
    raster_1 = rasterio.open(dataset_1)
    raster_2 = rasterio.open(dataset_2)

    bounds_1 = raster_1.bounds
    bounds_2 = raster_2.bounds

    if return_images:
        return_pixels = True

    scale_factors = [[1.0, 1.0]] * 2
    if return_pixels:
        raster_1_px_size = abs(raster_1.profile["transform"].a)
        raster_1_py_size = abs(raster_1.profile["transform"].e)

        raster_2_px_size = abs(raster_2.profile["transform"].a)
        raster_2_py_size = abs(raster_2.profile["transform"].e)

        if (raster_1_px_size != raster_2_px_size) or (
            raster_1_py_size != raster_2_py_size
        ):
            print(
                f"WARNING: Ground resolutions are different for the provided images. Setting them to the {resampling_resolution} resolution.\n"
            )

            os.makedirs("temp", exist_ok=True)
            outputs = adjust_resolutions(
                [dataset_1, dataset_2],
                ["temp/scaled_raster_1.tif", "temp/scaled_raster_2.tif"],
                resampling_resolution,
            )
            dataset_1 = "temp/scaled_raster_1.tif"
            dataset_2 = "temp/scaled_raster_2.tif"

            (
                (raster_1_px_size, raster_1_py_size, _),
                (
                    raster_2_px_size,
                    raster_2_py_size,
                    _,
                ),
            ), scale_factors = outputs

        min_left = min(bounds_1.left, bounds_2.left)
        max_top = max(bounds_1.top, bounds_2.top)

        res_x = [raster_1_px_size, raster_2_px_size]
        res_y = [raster_1_py_size, raster_2_py_size]
        selected_res_x = max(res_x) if resampling_resolution == "lower" else min(res_x)
        selected_res_y = max(res_y) if resampling_resolution == "lower" else min(res_y)

        bounds_1 = rasterio.coords.BoundingBox(
            int((bounds_1.left - min_left) / selected_res_x),
            int((max_top - bounds_1.bottom) / selected_res_y),
            int((bounds_1.right - min_left) / selected_res_x),
            int((max_top - bounds_1.top) / selected_res_y),
        )

        bounds_2 = rasterio.coords.BoundingBox(
            int((bounds_2.left - min_left) / selected_res_x),
            int((max_top - bounds_2.bottom) / selected_res_y),
            int((bounds_2.right - min_left) / selected_res_x),
            int((max_top - bounds_2.top) / selected_res_y),
        )

    overlap_left = max(bounds_1.left, bounds_2.left)
    overlap_bottom = (
        min(bounds_1.bottom, bounds_2.bottom)
        if return_pixels
        else max(bounds_1.bottom, bounds_2.bottom)
    )
    overlap_right = min(bounds_1.right, bounds_2.right)
    overlap_top = (
        max(bounds_1.top, bounds_2.top)
        if return_pixels
        else min(bounds_1.top, bounds_2.top)
    )

    x_condition = (overlap_right - overlap_left) > 0
    y_condition = (
        (overlap_bottom - overlap_top) > 0
        if return_pixels
        else (overlap_top - overlap_bottom) > 0
    )

    assert x_condition and y_condition, "The provided scenes do not overlap."

    overlap_in_mosaic = rasterio.coords.BoundingBox(
        overlap_left, overlap_bottom, overlap_right, overlap_top
    )

    mosaic = None
    mosaic_overlap = None
    raster_overlap_1 = None
    raster_overlap_2 = None

    if return_images:
        mosaic, warps, _ = make_mosaic([dataset_1, dataset_2], return_warps=True)
        mosaic_overlap = mosaic[
            overlap_in_mosaic.top : overlap_in_mosaic.bottom,
            overlap_in_mosaic.left : overlap_in_mosaic.right,
        ]
        raster_overlap_1 = warps[0][
            overlap_in_mosaic.top : overlap_in_mosaic.bottom,
            overlap_in_mosaic.left : overlap_in_mosaic.right,
        ]
        raster_overlap_2 = warps[1][
            overlap_in_mosaic.top : overlap_in_mosaic.bottom,
            overlap_in_mosaic.left : overlap_in_mosaic.right,
        ]

    shutil.rmtree("temp", ignore_errors=True)

    return (
        overlap_in_mosaic,
        (
            mosaic,
            mosaic_overlap,
            raster_overlap_1,
            raster_overlap_2,
        ),
        scale_factors,
    )


def make_mosaic(
    dataset_paths: list[str],
    offset_x: int = 0,
    offset_y: int = 0,
    return_warps: bool = False,
    resolution_adjustment: bool = False,
    resampling_resolution: str = "lower",
    mosaic_output_path: str = "",
    return_profile_only: bool = False,
    cv_flags: int = cv.INTER_NEAREST,
    output_type: str = "uint8",
    universal_masking: bool = False,
    nodata: int | float | None = None,
    cluster_masks: bool = False,
    force_crs: Literal["auto"] | str | None = "auto",
    no_affine: bool = False,
) -> tuple:
    """
    Creates a mosaic of overlapping scenes. Offsets will be added to the size of the final mosaic if specified.
    NOTE: dataset ground resolutions should be the same. Use `resolution_adjustment` flag to fix the unequal resolutions.

    Parameters
    ----------
    dataset_paths : list[str]
        List of paths to the datasets to be mosaiced.
    offset_x : int, optional
        Offset to be added to the width of the mosaic, by default 0.
    offset_y : int, optional
        Offset to be added to the height of the mosaic, by default 0.
    return_warps : bool, optional
        If True, returns the warped images of the mosaiced datasets, by default False.
    resolution_adjustment : bool, optional
        If True, adjusts the resolutions of the datasets to match the selected `resampling_resolution`,
        by default False.
    resampling_resolution : str, optional
        The resolution to which the datasets will be adjusted if `resolution_adjustment` is True.
        Can be either "lower" or "higher", by default "lower".
    mosaic_output_path : str, optional
        If provided, the mosaic will be saved to this path. If not provided, the mosaic will not be saved.
    return_profile_only : bool, optional
        If True, returns only the real world profiles of the mosaic instead of the new transforms in pixels.
        Defaults to False.
    cv_flags : int, optional
        OpenCV flags for the warping process. Defaults to `cv.INTER_NEAREST`.
    output_type : str, optional
        The data type of the output mosaic. Defaults to "uint8". Other options could be "uint16", "float32", etc.
    universal_masking : bool, optional
        If True, applies a universal masking to the mosaic to ensure that all pixels are masked correctly
        across all datasets. This is useful for datasets with different nodata values.
        Defaults to False.
    nodata : int | float | None, optional
        The nodata value to be used for masking.
    cluster_masks : bool, optional
        If True, clusters the masks of the datasets to create a universal mask.
        Defaults to False.
    force_crs : Literal["auto"] | dict | None, optional
        If "auto", the function will try to determine the CRS from the datasets.
        If None, the CRS will not be forced. Defaults to "auto".
        If a str, the function will use the provided CRS information. The provided CRS should be in the form of an EPSG code (e.g., "EPSG:4326").
    no_affine: bool, optional
        Does not perform affine transformation and assumes the images are already geo-referenced.

    Returns
    -------
    tuple
        A tuple containing:
        - The mosaiced image as a numpy array.
        - A list of warped images if `return_warps` is True, otherwise an empty list.
        - A list of new transforms for each dataset or the real world profiles if `return_profile_only` is True.
    """

    if resolution_adjustment:
        os.makedirs("temp/res_adjustment", exist_ok=True)
        new_dataset_paths = [
            os.path.join("temp/res_adjustment", f"scaled_raster_{i}.tif")
            for i in range(len(dataset_paths))
        ]
        adjust_resolutions(
            dataset_paths,
            new_dataset_paths,
            resampling_resolution,
        )
        dataset_paths = new_dataset_paths

    ps_x = []
    ps_y = []
    rasters = []
    transforms = []
    boundss = []
    original_boundss = []
    crs_numbers = []
    first_crs = rasterio.open(dataset_paths[0]).crs
    if first_crs is None:
        force_crs = None
    else:
        if type(force_crs) == str and force_crs != "auto":
            first_crs_data = force_crs
        else:
            first_crs_data = f"EPSG:{first_crs.to_epsg()}"

    for p in dataset_paths:
        with rasterio.open(p, "r") as raster:
            raster_crs = raster.crs
            if raster_crs is not None:
                crs_numbers.append(raster.crs.to_epsg())

    if type(force_crs) == str and force_crs != "auto":
        print(f"Using provided CRS information: {force_crs}")
    elif len(crs_numbers) != 0 and np.any(np.diff(crs_numbers) != 0):
        if force_crs is None:
            warnings.warn(
                "Datasets have different CRS. The mosaicing process might fail with discrepancies in CRS information."
            )
        else:
            warnings.warn(
                f"Datasets have different CRS. The first CRS: {first_crs}, will be used for the mosaicing process."
            )
    else:
        force_crs = None

    for p in dataset_paths:
        raster_path = p
        with rasterio.open(raster_path, "r") as raster:
            raster_crs = raster.crs
        if raster_crs is None:
            warnings.warn(f"Dataset {p} does not have a CRS.")
        else:
            if force_crs == "auto" or type(force_crs) == str:
                os.makedirs("temp/reproject", exist_ok=True)
                new_p = f"temp/reproject/{os.path.basename(p)}"
                reproject_tif(p, new_p, first_crs_data, verbose=False)
                raster_path = new_p
            elif force_crs is not None:
                raise ValueError(
                    "force_crs should be either 'auto' or None. Please check the documentation."
                )

        raster = rasterio.open(raster_path)
        raster_crs = raster.crs
        raster_utm_bounds = (
            utm_bounds(raster.bounds, raster_crs.data)
            if raster_crs is not None
            else raster.bounds
        )  # I need to get rid of utm_bounds func at some point. technically doing nothing unless bounds are negative.
        boundss.append(raster_utm_bounds)
        original_boundss.append(raster_utm_bounds)
        transform = raster.transform
        ps_x.append(abs(transform.a))
        ps_y.append(abs(transform.e))
        rasters.append(raster)
        transforms.append(transform)

    selected_res_x = max(ps_x) if resampling_resolution == "lower" else min(ps_x)
    selected_res_y = max(ps_y) if resampling_resolution == "lower" else min(ps_y)
    ps_x_condition = all(round(ps) == round(ps_x[0]) for ps in ps_x)
    ps_y_condition = all(round(ps) == round(ps_y[0]) for ps in ps_y)
    if not (ps_x_condition and ps_y_condition):
        print(
            """Ground resolutions are different for datasets. The mosaicing process might fail if adding large datasets.
              Please use `resolution_adjustment` flag first if you encounter memory related issues."""
        )

    lefts = []
    rights = []
    tops = []
    bottoms = []
    original_tops = []
    original_lefts = []
    for i, bounds in enumerate(boundss):
        lefts.append(bounds.left)
        rights.append(bounds.right)
        tops.append(bounds.top)
        bottoms.append(bounds.bottom)
        original_tops.append(original_boundss[i].top)
        original_lefts.append(original_boundss[i].left)

    min_left = min(lefts)
    min_bottom = min(bottoms)
    max_right = max(rights)
    max_top = max(tops)

    max_original_top = max(original_tops)
    min_original_left = min(original_lefts)

    new_shape = (
        int((max_top - min_bottom) / selected_res_y) + offset_y,
        int((max_right - min_left) / selected_res_x) + offset_x,
    )

    new_transforms = []
    for i, t in enumerate(transforms):
        orig_x = boundss[i].left
        orig_y = boundss[i].top
        new_transforms.append(
            np.array(
                [
                    [
                        t.a / selected_res_x,
                        abs(t.b / t.e),
                        (orig_x - min_left) / selected_res_x,
                    ],
                    [
                        t.d / t.a,
                        abs(t.e / selected_res_y),
                        (max_top - orig_y) / selected_res_y,
                    ],
                ]
            )
        )

    if no_affine:
        new_shape = rasters[0].shape
    mosaic = np.zeros((*new_shape, 3)).astype(output_type)
    warps = []
    masks = []
    for i, rs in enumerate(rasters):
        if universal_masking:
            data = flip_img(rs.read(masked=True))
            mask = data.mask
            img = data.data
        else:
            img = flip_img(rs.read())
        if no_affine:
            imgw = img.copy()
        else:
            imgw = cv.warpAffine(
                img,
                new_transforms[i],
                (new_shape[1], new_shape[0]),
                flags=cv_flags,
            )
        if universal_masking:
            if no_affine:
                maskw = mask.copy().astype(np.uint8)
            else:
                maskw = cv.warpAffine(
                    mask.astype(np.uint8),
                    new_transforms[i],
                    (new_shape[1], new_shape[0]),
                    borderValue=tuple([1] * mask.shape[2]),
                    flags=cv_flags,
                )
            maskw = np.where(maskw == 1, 1, 0).astype(bool)
            masks.append(maskw)

        if len(imgw.shape) == 2:
            idx = np.where(imgw != 0)
            for i in range(0, 3):
                mosaic[idx[0], idx[1], i] = imgw[idx[0], idx[1]]
        else:
            if imgw.shape[2] == 1:
                warnings.warn(
                    "WARNING: Single channel image detected. Converting to 3 channels."
                )
                imgw = cv.merge([imgw] * 3)
            elif imgw.shape[2] == 2:
                warnings.warn(
                    "WARNING: Two channel image detected. Converting to 3 channels."
                )
                imgw = cv.merge(
                    [
                        imgw[:, :, 0],
                        imgw[:, :, 1],
                        imgw[:, :, 0],
                    ]
                )

            idx = np.where(cv.cvtColor(imgw, cv.COLOR_BGR2GRAY) != 0)
            mosaic[idx[0], idx[1], :] = imgw[idx[0], idx[1], :]

        if return_warps:
            warp = np.zeros_like(imgw).astype(output_type)
            if len(imgw.shape) == 2:
                warp[idx[0], idx[1]] = imgw[idx[0], idx[1]]
            else:
                warp[idx[0], idx[1], :] = imgw[idx[0], idx[1], :]
            warps.append(warp)

    if universal_masking:
        if cluster_masks:
            cls_list = np.array(
                [
                    np.mean(np.argwhere(np.all(ar, axis=2) == 0), axis=0) / ar.shape[:2]
                    for ar in masks
                ]
            )
            nan_inds = np.where(np.isnan(cls_list[:, 0]))[0]
            nan_arr = np.arange(1, len(nan_inds) + 1) * -1
            cls_list[nan_inds] = np.stack([nan_arr, nan_arr], axis=1)
            cls = dbscan.fit(np.array(cls_list))
            labels = cls.labels_
            unique_labels = list(filter(lambda x: x != -1, np.unique(labels)))
            print(f"Number of clusters in masks: {len(unique_labels)}")
            masks_dict = dict(
                [(j, i) for i, j in zip(labels, range(len(masks)))]
            )  # Create a dictionary of masks with their labels and indices
            masks_per_cluster = []
            for label in unique_labels:  # Iterate over unique labels
                cluster_masks_idx = [
                    i for i in (range(len(labels))) if labels[i] == label
                ]
                cluster_masks = [masks[i] for i in cluster_masks_idx]
                masks_per_cluster.append(np.logical_or.reduce(cluster_masks))

            for i, warp in enumerate(warps):
                if masks_dict[i] != -1:
                    universal_mask = masks_per_cluster[masks_dict[i]]
                    if np.all(universal_mask):
                        warnings.warn(
                            "WARNING: All pixels in the warps are masked. Warped outputs will be empty."
                        )
                    warp[universal_mask] = 0
        else:
            universal_mask = np.logical_or.reduce(masks)
            if np.all(universal_mask):
                warnings.warn(
                    "WARNING: All pixels in the warps are masked. Warped outputs will be empty."
                )
            # if len(universal_mask.shape) == 3:
            #     universal_mask = np.all(universal_mask, axis=2)
            for warp in warps:
                warp[universal_mask] = 0

    if os.path.exists("temp/reproject"):
        shutil.rmtree("temp/reproject", ignore_errors=True)

    mosaic_profile = rasterio.open(dataset_paths[0]).profile
    if resolution_adjustment:
        shutil.rmtree("temp/res_adjustment", ignore_errors=True)

    mosaic_profile["height"] = new_shape[0]
    mosaic_profile["width"] = new_shape[1]
    mosaic_profile["count"] = 3
    mosaic_profile["transform"] = rasterio.Affine(
        selected_res_x, 0.0, min_left, 0.0, -selected_res_y, max_top
    )
    original_mosaic_profile = mosaic_profile.copy()
    original_mosaic_profile["transform"] = rasterio.Affine(
        selected_res_x, 0.0, min_original_left, 0.0, -selected_res_y, max_original_top
    )
    original_mosaic_profile["dtype"] = output_type
    if first_crs is not None:
        mosaic_profile["crs"] = first_crs
        original_mosaic_profile["crs"] = first_crs

    if mosaic_output_path != "":
        print("Writing mosaic file.")
        with rasterio.open(mosaic_output_path, "w", **mosaic_profile) as ds:
            for i in range(0, 3):
                ds.write(mosaic[:, :, i], i + 1)

    if nodata is not None:
        mosaic_profile["nodata"] = nodata
        original_mosaic_profile["nodata"] = nodata

    return (
        mosaic,
        ((warps, masks) if universal_masking else warps),
        (
            (mosaic_profile, original_mosaic_profile)
            if return_profile_only
            else new_transforms
        ),
    )


def simple_mosaic(img_list):
    """
    A simple mosaic of two image by overlapping the non-zero parts in a consecutive order.
    Assumes all images are the same size and ignores any transformations.
    """
    mosaic = np.zeros_like(img_list[0]).astype("uint8")
    for img in img_list:
        if len(img.shape) == 2:
            idx = np.where(img != 0)
            mosaic[idx[0], idx[1]] = img[idx[0], idx[1]]
        else:
            idx = np.where(cv.cvtColor(img, cv.COLOR_BGR2GRAY) != 0)
            mosaic[idx[0], idx[1], :] = img[idx[0], idx[1], :]
    return mosaic


def make_difference_gif(
    images_list: list[str],
    output_path: str,
    titles_list: list[str] = [],
    scale_factor: float = -1.0,
    mosaic_scenes: bool = False,
    mosaic_offsets_x: int = 0,
    mosaic_offsets_y: int = 0,
    fps: int = 3,
    font_scale: float = 1.5,
    thickness: int = 3,
    color: tuple = (255, 0, 0),
    origin: tuple = (5, 50),
) -> None:
    """Makes a GIF from a list of images with titles and optional scaling.

    Parameters
    ----------
    images_list : list[str]
        List of image file paths to be included in the GIF.
    output_path : str
        Output path for the generated GIF.
    titles_list : list[str], optional
        List of titles for each image, by default []
    scale_factor : float, optional
        Scale factor for downsampling the images, by default -1.0
    mosaic_scenes : bool, optional
        Flag to indicate if the images should be mosaiced, by default False
    mosaic_offsets_x : int, optional
        Offset mosaic size in the x direction for the mosaic, by default 0
    mosaic_offsets_y : int, optional
        Offset mosaic size in the y direction for the mosaic, by default 0
    fps : int, optional
        Frames per second for the GIF, by default 3
    font_scale : float, optional
        Size of the font for titles, by default 1.5
    thickness : int, optional
        Thickness of the text for titles, by default 3
    color : tuple, optional
        Color of the text for titles, by default (255, 0, 0)
    origin : tuple, optional
        Origin point for the text placement, by default (5, 50)
    """
    os.makedirs("temp", exist_ok=True)
    temp_paths = [os.path.join("temp", os.path.basename(f)) for f in images_list]

    if scale_factor != -1.0:
        for i, p in enumerate(temp_paths):
            downsample_dataset(images_list[i], scale_factor, p)
    else:
        temp_paths = images_list

    if len(titles_list) > 0:
        assert len(titles_list) == len(
            images_list
        ), "Length of provided list of titles does not match the number of images."
    else:
        titles_list = [os.path.splitext(os.path.basename(f))[0] for f in images_list]

    images = []
    font = cv.FONT_HERSHEY_SIMPLEX
    if mosaic_scenes:
        _, warps, _ = make_mosaic(
            images_list,
            mosaic_offsets_x,
            mosaic_offsets_y,
            return_warps=True,
        )
        for i, warp in enumerate(warps):
            cv.putText(
                warp,
                titles_list[i],
                origin,
                font,
                font_scale,
                color,
                thickness,
                cv.LINE_AA,
            )
            images.append(warp)
    else:
        temp_images = []
        transforms = []
        for i, p in enumerate(temp_paths):
            img_raster = rasterio.open(p)
            transforms.append(img_raster.profile["transform"])
            img = img_raster.read()
            img = flip_img(img).copy().astype("uint8")
            if (len(img.shape) == 3) and (img.shape[2] == 1):
                img = img[:, :, 0]
            temp_images.append(img)

        for i, img in enumerate(temp_images):
            cv.putText(
                img,
                titles_list[i],
                origin,
                font,
                font_scale,
                color,
                thickness,
                cv.LINE_AA,
            )
            images.append(img)

    imageio.mimwrite(output_path, images, loop=0, fps=fps)
    shutil.rmtree("temp", ignore_errors=True)
    return None


def shift_targets_to_origin(
    tgt_imgs: list[np.ndarray],
    ref_transform: rasterio.Affine,
    tgt_transforms: list[rasterio.Affine],
) -> list[np.ndarray]:
    """Shifts the target images to the origin of the reference image based on their affine transformations.

    Parameters
    ----------
    tgt_imgs : list[np.ndarray]
        Target images to be shifted.
    ref_transform : rasterio.Affine
        Reference image affine transformation.
    tgt_transforms : list[rasterio.Affine]
        Target images affine transformations.

    Returns
    -------
    list[np.ndarray]
        List of shifted target images.
    """
    ref_x = abs(ref_transform.c / ref_transform.a)
    ref_y = abs(ref_transform.f / ref_transform.e)
    shifted_tgts = []
    for i, img in enumerate(tgt_imgs):
        tgt_x = abs(tgt_transforms[i].c / tgt_transforms[i].a)
        tgt_y = abs(tgt_transforms[i].f / tgt_transforms[i].e)
        shift_x = tgt_x - ref_x
        shift_y = ref_y - tgt_y
        shifted_tgts.append(
            warp_affine_dataset(
                img, translation_x=shift_x, translation_y=shift_y
            ).astype("uint8")
        )
    return shifted_tgts


def find_band_files_s2(dir: str, selected_res_index: int = -1) -> list[str]:
    """
    Retrieving band files from the data path `dir`, If there are multiple resolutions of the data, `selected_res_index` must be specified as a positive integer.
    """
    if selected_res_index == -1:
        band_files = [
            file
            for file in glob.glob(os.path.join(dir, "GRANULE", "*", "IMG_DATA", "*"))
        ]
    else:
        band_dirs = [
            file
            for file in glob.glob(os.path.join(dir, "GRANULE", "*", "IMG_DATA", "*"))
        ]
        selected_res = band_dirs[selected_res_index]
        band_files = glob.glob(f"{selected_res}/*")
    return band_files


def load_s2_bands(
    zip_file_path: str,
    zip_file_id: str,
    s2_other_bands_list: list[str],
    subdir_name: str = "",
    remove_input: bool = True,
    selected_res_index: int = -1,
) -> tuple:
    """
    Loads bands from the data.
    * `zip_file_path` and `zip_file_ids` are the path and id of the data.
    * The function also loads the other bands specified in the  other bands list
    * `subdir_name` will be added to the output directory path if provided
    * If there are multiple resolutions of the data, `selected_res_index` must be specified as a positive integer
    """
    with ZipFile(zip_file_path) as f:
        f.extractall(f"./data/inputs/{zip_file_id}/{subdir_name}")
    data_dir = os.listdir(f"./data/inputs/{zip_file_id}/{subdir_name}")[0]
    dir = os.path.join(f"./data/inputs/{zip_file_id}/{subdir_name}", data_dir)
    band_files = find_band_files_s2(dir, selected_res_index)
    other_band_files = []
    for suffix in s2_other_bands_list:
        other_band = list(filter(lambda b: suffix in b, band_files))
        if other_band != []:
            other_band_files.append(other_band[0])

    band_files = [bf for bf in band_files if bf not in other_band_files]
    bands = [rasterio.open(band_file) for band_file in band_files]

    other_bands = []
    for bf in other_band_files:
        other_bands.append(rasterio.open(bf))

    if remove_input:
        shutil.rmtree(f"./data/inputs/{zip_file_id}/{subdir_name}", ignore_errors=True)

    return bands, other_band


def write_true_color_s2(
    zip_id: str,
    true_bands,
    band_profile,
    scale_factor=0.03,
    subdir_name="",
) -> tuple:
    """
    writes a true color image using true color bands (not TCI). The updated profile for true bands shold be provided.
    also downsamples the output data and returns the new downsampled data and its new affine transformation according to `scale_factor`
    """
    output_dir = f"./data/outputs/{zip_id}/{subdir_name}/"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(f"{output_dir}/out.tif", "w", **band_profile) as dest_file:
        for i, b in enumerate(true_bands):
            dest_file.write(b.read(1), i + 1)

    new_data, new_transform = downsample_dataset(f"{output_dir}/out.tif", scale_factor)

    return new_data, new_transform


def enhance_color_s2(data, uint16: bool = False):
    """
    Increases the brightness of the output data
    """
    if uint16:
        data = data / 256
    data = data.astype("float64")
    data *= 255 / data.max()
    return data.astype("uint8")


def extract_true_bands(bands):
    "Extracts true color bands and retuns them together with an updated profile for the bands"
    true_bands = [
        b
        for b in bands
        if os.path.basename(b.name).split("_")[2].replace(".jp2", "").strip()
        in ["B02", "B03", "B04"]
    ]
    sortperm = np.argsort([b.name for b in true_bands])
    true_bands = [true_bands[i] for i in sortperm]
    band_profile = true_bands[0].profile
    band_profile.update({"count": len(true_bands)})
    return true_bands, band_profile


def plot_scene_s2(data, data_transform):
    """
    Plots the true color image and blue band of the scene given the names of the scenes and their affine transformation
    """
    _, (axt, axb) = plt.subplots(1, 2, figsize=(10, 20))
    show(
        enhance_color_s2(data[1]),
        ax=axb,
        title="Single color band (B)",
        cmap="Blues",
        transform=data_transform,
    )
    show(
        enhance_color_s2(data),
        ax=axt,
        title="True colour bands image",
        transform=data_transform,
    )


def resize_bbox(bbox, scale_factor=1.0) -> BoundingBox:
    """Resizes the bounding box by a given scale factor.

    Parameters
    ----------
    bbox : _type_
        BoundingBox object representing the bounding box to resize.
    scale_factor : float, optional
        scale factor to apply to the bounding box dimensions, by default 1.0.

    Returns
    -------
    BoundingBox
        Resized bounding box with the adjusted dimensions.
    """
    x_dim = bbox.right - bbox.left
    y_dim = bbox.top - bbox.bottom

    dx = ((scale_factor - 1) * x_dim) / 2
    dy = ((scale_factor - 1) * y_dim) / 2

    return BoundingBox(bbox.left - dx, bbox.bottom - dy, bbox.right + dx, bbox.top + dy)


UTM = namedtuple("UTM", ["x", "y"])
LLA = namedtuple("LLA", ["lat", "lon"])

UTM3 = namedtuple("UTM3", ["x", "y", "z"])
LLA3 = namedtuple("LLA3", ["lat", "lon", "alt"])

UTM3to2 = lambda utm3: UTM(utm3.x, utm3.y)
LLA3to2 = lambda lla3: LLA(lla3.lat, lla3.lon)

UTM2to3 = lambda utm: UTM3(utm.x, utm.y, 0.0)
LLA2to3 = lambda lla: LLA3(lla.lat, lla.lon, 0.0)


def UTMtoLLA(utm: Union[UTM, UTM3], crs: dict):
    """
    `UMT(x, y)`

    crs example:
    ```
    crs = {
        'proj': 'utm',
        'zone': 42,
        'south': True,
        'datum': 'WGS84',
        'units': 'm',
        'no_defs': True
    }
    ```
    """
    proj = Proj(**crs)
    lla = proj(utm.x, utm.y, inverse=True)
    if type(utm) == UTM3:
        return LLA3(lla[1], lla[0], utm.z)
    else:
        return LLA(lla[1], lla[0])


def LLAtoUTM(lla: Union[LLA, LLA3], crs: Union[dict, None]):
    """
    `LLA(lat, lon)`

    crs example:
    ```
    crs = {
        'proj': 'utm',
        'zone': 42,
        'south': True,
        'datum': 'WGS84',
        'units': 'm',
        'no_defs': True
    }
    ```
    """
    proj = Proj(**crs)
    if crs["proj"] != "utm":
        e, n, _, _ = utm_convrter.from_latlon(lla.lat, lla.lon)
        output = (float(e), float(n))
    else:
        utm = proj(lla.lon, lla.lat)
        output = (utm[0], utm[1])

    if type(lla) == LLA3:
        return UTM3(*output, lla.alt)
    else:
        return UTM(*output)


def utm_bounds(
    bounds: BoundingBox,
    crs: dict,
    skip_stereo: bool = True,
    forced_zone: int | None = None,
) -> BoundingBox:
    """Returns the bounding box in UTM coordinates if the CRS is UTM or Stereographic.

    Parameters
    ----------
    bounds : BoundingBox
        Bounding box to convert.
    crs : dict
        CRS data containing projection information.
    skip_stereo : bool, optional
        Skip conversion for stereographic projection, by default True.
    forced_zone : int | None, optional
        If specified, forces the conversion to the given UTM zone, by default None.

    Returns
    -------
    BoundingBox
        Converted bounding box in UTM coordinates or the original bounding box if no conversion is needed.
    """
    has_negative = any(b < 0 for b in bounds)
    if crs == {}:
        print("No CRS data found. Returning original.")
        return bounds
    if (crs["proj"] == "stere") and (skip_stereo):
        return bounds

    lla_bl = UTMtoLLA(UTM(bounds.left, bounds.bottom), crs)
    lla_tr = UTMtoLLA(UTM(bounds.right, bounds.top), crs)
    lla_br = UTMtoLLA(UTM(bounds.right, bounds.bottom), crs)
    lla_tl = UTMtoLLA(UTM(bounds.left, bounds.top), crs)
    if (crs["proj"] == "utm") and ("south" not in crs) and (has_negative):
        crs = {"south": True} | crs
    if (crs["proj"] == "stere") or (
        (crs["proj"] == "utm") and (forced_zone is not None)
    ):
        if forced_zone is not None:
            crs["zone"] = forced_zone
        utm_bl = LLAtoUTM(lla_bl, crs)
        utm_tr = LLAtoUTM(lla_tr, crs)
        utm_br = LLAtoUTM(lla_br, crs)
        utm_tl = LLAtoUTM(lla_tl, crs)
        xs = [utm.x for utm in [utm_bl, utm_tr, utm_br, utm_tl]]
        ys = [utm.y for utm in [utm_bl, utm_tr, utm_br, utm_tl]]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))
    elif crs["proj"] == "utm":
        return bounds


def find_scene_bounding_box_lla(scene: str, scale_factor=1.0) -> str:
    """Finds the bounding box of the scene in LLA coordinates.

    Parameters
    ----------
    scene : str
        Path to the scene file.
    scale_factor : float, optional
        Scale factor to resize the bounding box, by default 1.0

    Returns
    -------
    str
        Bounding box in the format "west,south,east,north".
    """
    raster = rasterio.open(scene)
    raster_bounds = raster.bounds

    raster_bounds = resize_bbox(raster_bounds, scale_factor)

    raster_crs = raster.crs

    raster_proj = Proj(**raster_crs.data)

    west, south = raster_proj(raster_bounds.left, raster_bounds.bottom, inverse=True)
    east, north = raster_proj(raster_bounds.right, raster_bounds.top, inverse=True)

    bbox = f"{west},{south},{east},{north}"

    return bbox


def warp_affine_dataset(
    dataset: Union[str, np.ndarray],
    output_path: str = "",
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    rotation_angle: float = 0.0,
    scale: float = 1.0,
    write_new_transform: bool = False,  # if writing to an output file, changes the transform of the profile instead of shifting image pixels.
) -> np.ndarray:
    """
    Transforms the dataset accroding to given translation, rotation and scale params and writes it to the `output_path` file.

    Parameters
    ----------
    dataset : Union[str, np.ndarray]
        The input dataset to be transformed. Can be a file path or a numpy array.
    output_path : str, optional
        The path to save the transformed dataset. If empty, the transformed image will not be saved
    translation_x : float, optional
        The translation in the x direction, by default 0.0
    translation_y : float, optional
        The translation in the y direction, by default 0.0
    rotation_angle : float, optional
        The rotation angle in degrees, by default 0.0
    scale : float, optional
        The scale factor for the image, by default 1.0
    write_new_transform : bool, optional
        If True, the new affine transformation will be written to the output file's profile.
        If False, the image pixels will be shifted according to the translation parameters, by default False

    Returns
    -------
    np.ndarray
        The transformed image as a numpy array.
        If `output_path` is provided, the transformed image will also be saved to that path
    """
    if type(dataset) == str:
        raster = rasterio.open(dataset)
        img = flip_img(raster.read()).copy()
    else:
        img = dataset
    img_centre = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_mat = cv.getRotationMatrix2D(img_centre, rotation_angle, scale)
    translation_mat = np.array([[1.0, 0.0, translation_x], [0.0, 1.0, translation_y]])
    affine_transform = np.matmul(
        rotation_mat, np.vstack([translation_mat, np.array([0, 0, 1])])
    )
    warped_img = cv.warpAffine(img, affine_transform, (img.shape[1], img.shape[0]))

    if (type(dataset) == str) and (output_path != ""):
        profile = raster.profile
        if write_new_transform:
            profile["transform"] = rasterio.Affine(*affine_transform.ravel())
            with rasterio.open(output_path, "w", **profile) as ds:
                for i in range(0, profile["count"]):
                    ds.write(img[:, :, i], i + 1)
        else:
            with rasterio.open(output_path, "w", **profile) as ds:
                if profile["count"] == 1:
                    ds.write(warped_img, 1)
                else:
                    for i in range(0, profile["count"]):
                        ds.write(warped_img[:, :, i], i + 1)

    return warped_img


def find_cells(
    img: np.ndarray, points: np.ndarray, window_size: tuple, invert_points: bool = False
) -> list[np.ndarray]:
    """Finds grid cells around the given points in the image.
    Parameters
    ----------
    img : np.ndarray
        The input image.
    points : np.ndarray
        The points around which to find the grid cells.
    window_size : tuple
        The size of the grid cells.
    invert_points : bool, optional
        If True, inverts the points to (y, x) format, by default False
    Returns
    -------
    list
        List of grid cells as numpy arrays.
    """
    if invert_points:
        points = np.column_stack([points[:, 1], points[:, 0]])
    h = img.shape[0]
    w = img.shape[1]
    cells = []
    sx = window_size[0] // 2
    sy = window_size[1] // 2
    for p in points:
        l_h = p[0] - sx
        u_h = p[0] + sx
        l_w = p[1] - sy
        u_w = p[1] + sy
        h_range = (np.clip(l_h, 0, l_h), np.clip(u_h, u_h, h))
        w_range = (np.clip(l_w, 0, l_w), np.clip(u_w, u_w, w))
        cells.append(img[h_range[0] : h_range[1], w_range[0] : w_range[1]])
    return cells


def find_corrs_shifts(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    ref_points: np.ndarray,
    tgt_points: np.ndarray,
    corr_win_size: tuple = (25, 25),
    signal_power_thresh: float = 0.9,
    drop_unbound: bool = True,
    invert_points: bool = True,
    valid_num_points=10,
) -> tuple:
    """Filters the reference and target points based on phase correlation of the grid cells around the points.

    Parameters
    ----------
    ref_img : np.ndarray
        Reference image corresponding to the reference points.
    tgt_img : np.ndarray
        Target image corresponding to the target points.
    ref_points : np.ndarray
        Reference points to be filtered.
    tgt_points : np.ndarray
        Target points to be filtered.
    corr_win_size : tuple, optional
        Window size for calculating phase correlation aroung a feature point, by default (25, 25)
    signal_power_thresh : float, optional
        Phase correlation signal threshold for filtering points, by default 0.9
    drop_unbound : bool, optional
        Drop out og bound points, by default True
    invert_points : bool, optional
        invert points to (y, x) format, by default True
    valid_num_points : int, optional
        Valid number of points to be found, if fewer points are found, the function will return the original points, by default 10

    Returns
    -------
    tuple
        Filtered reference and target points as numpy arrays.
        If no valid features are found, returns (ref_points, tgt_points).
    """

    ref_points_temp = ref_points.copy().astype("int")
    tgt_points_temp = tgt_points.copy().astype("int")

    ref_cells = find_cells(
        ref_img,
        ref_points_temp,
        corr_win_size,
        invert_points,
    )
    tgt_cells = find_cells(
        tgt_img,
        tgt_points_temp,
        corr_win_size,
        invert_points,
    )

    if drop_unbound:
        final_ref_cells = []
        final_tgt_cells = []
        for ref, tgt in zip(ref_cells, tgt_cells):
            if ref.shape == tgt.shape:
                final_ref_cells.append(ref)
                final_tgt_cells.append(tgt)
    else:
        final_ref_cells = ref_cells
        final_tgt_cells = tgt_cells

    corrs = []
    warning = False
    for ref, tgt in zip(final_ref_cells, final_tgt_cells):
        try:
            corrs.append(cv.phaseCorrelate(np.float32(tgt), np.float32(ref), None)[1])
        except Exception as e:
            warning = True

    if warning:
        print(
            "WARNING: Phase correlation filtering failed for some of the gird cells. This might be due to the size of the cells or the image data type."
        )

    valid_idx = np.where(np.array(corrs) > signal_power_thresh)

    if len(valid_idx[0]) < valid_num_points:
        has_s = " was" if len(valid_idx[0]) == 1 else "s were"
        print(
            f"WARNING: {len(valid_idx[0])} point{has_s} found with the given correlation threshold (fewer than accepted {valid_num_points}), turning off phase correlation filter..."
        )
        return ref_points, tgt_points

    return ref_points[valid_idx], tgt_points[valid_idx]


def filter_features(
    ref_points: np.ndarray,
    tgt_points: np.ndarray,
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    bounding_shape: tuple,
    dists: np.ndarray,
    dist_thresh: Union[None, int, float] = None,
    lower_of_dist_thresh: Union[None, int, float] = None,
    target_info: Union[None, tuple] = None,
) -> tuple:
    """Filters the reference and target points based on distance thresholds and image validity.

    Parameters
    ----------
    ref_points : np.ndarray
        Array of reference points to be filtered.
    tgt_points : np.ndarray
        Array of target points to be filtered.
    ref_img : np.ndarray
        Reference image corresponding to the reference points.
    tgt_img : np.ndarray
        Target image corresponding to the target points.
    bounding_shape : tuple
        Shape of the bounding box for the images, used to validate points.
    dists : np.ndarray
        Array of distances corresponding to the reference and target points.
    dist_thresh : Union[None, int, float], optional
        Threshold for filtering points based on distance, by default None
    lower_of_dist_thresh : Union[None, int, float], optional
        Lower threshold for filtering points based on distance, by default None
    target_info : Union[None, tuple], optional
        Information about the target, used for logging if no valid features are found, by default None

    Returns
    -------
    tuple
        Filtered reference and target points as numpy arrays.
        If no valid features are found, returns (None, None).
    """

    if dist_thresh != None:
        if lower_of_dist_thresh != None:
            upper_idx = np.squeeze(dists) < dist_thresh
            lower_idx = np.squeeze(dists) > lower_of_dist_thresh
            filter_idx = np.where(np.logical_and(upper_idx, lower_idx))
        else:
            filter_idx = np.where(np.squeeze(dists) < dist_thresh)
        ref_good = np.squeeze(ref_points)[filter_idx]
        tgt_good = np.squeeze(tgt_points)[filter_idx]

        valid_idx = np.all(
            (
                tgt_good[:, 0] >= 0,
                tgt_good[:, 1] >= 0,
                tgt_good[:, 0] < bounding_shape[1],
                tgt_good[:, 1] < bounding_shape[0],
            ),
            axis=0,
        )

        ref_good = ref_good[valid_idx]
        tgt_good = tgt_good[valid_idx]

        points = ref_good.astype("int")
        invalid_idx_ref = np.where(ref_img[points[:, 1], points[:, 0]] == 0)
        points = tgt_good.astype("int")
        invalid_idx_tgt = np.where(tgt_img[points[:, 1], points[:, 0]] == 0)
        invalid_idx = set(
            np.hstack([invalid_idx_ref, invalid_idx_tgt]).ravel().tolist()
        )
        valid_idx = np.array(list(set(range(0, len(ref_good))) - invalid_idx))

        if len(valid_idx) == 0:
            info_str = ""
            if target_info is not None:
                info_str = f"For target {target_info[0]} ({target_info[1]}), "
            print(info_str + "Couldn't find valid features for target or reference.")
            return None, None

        tgt_good = np.expand_dims(tgt_good[valid_idx], axis=0)
        ref_good = np.expand_dims(ref_good[valid_idx], axis=0)
    else:
        tgt_good = np.expand_dims(tgt_points, axis=0)
        ref_good = np.expand_dims(ref_points, axis=0)

    return ref_good, tgt_good


def co_register(
    reference: Union[str, np.ndarray],
    targets=Union[
        str, np.ndarray, list[str], list[np.ndarray], list[Union[str, np.ndarray]]
    ],
    number_of_iterations: int = 30,
    termination_eps: float = 0.03,
    of_params: dict = dict(
        # params for ShiTomasi corner detection
        feature_params=dict(
            maxCorners=20000,
            qualityLevel=0.1,
            minDistance=10,
            blockSize=15,
        ),
        # Parameters for lucas kanade optical flow
        lk_params=dict(
            winSize=(25, 25),
            maxLevel=1,
        ),
    ),
    output_path: str = "",
    export_outputs: bool = True,
    fps: int = 3,
    of_dist_thresh: Union[None, int, float] = 2,  # pixels
    phase_corr_filter: bool = True,
    phase_corr_signal_thresh: float = 0.9,
    phase_corr_valid_num_points=10,
    rethrow_error: bool = False,
    resampling_resolution: str = "lower",
    laplacian_kernel_size: Union[None, int, list] = None,
    laplacian_for_targets_ids: list | None = None,
    lower_of_dist_thresh: Union[None, int, float] = None,
    band_number: Union[None, int] = None,
) -> tuple:
    """
    Co-registers the target images to the reference image using optical flow and phase correlation.

    Parameters
    ----------
    reference: Union[str, np.ndarray]
        Path to the reference image or a numpy array of the reference image.
    targets: Union[str, np.ndarray, list[str], list[np.ndarray], list[Union[str, np.ndarray]]]
        Path to the target image(s) or a numpy array of the target image(s). Can be a single image or a list of images.
    number_of_iterations: int, Optional
        Number of iterations for the optical flow algorithm, by default 30.
    termination_eps: float, Optional
        Termination epsilon for the optical flow algorithm, by default 0.03.
    of_params: dict, Optional
        Parameters for the optical flow algorithm, by default dict(
            feature_params=dict(maxCorners=20000, qualityLevel=0.1, minDistance=10, blockSize=15),
            lk_params=dict(winSize=(25, 25), maxLevel=1),
        ).
    output_path: str, Optional
        Path to save the output images, by default "" (no output).
    export_outputs: bool, Optional
        Whether to export the output images, by default True.
    fps: int, Optional
        Frames per second for the GIF, by default 3.
    of_dist_thresh: Union[None, int, float], Optional
        Distance threshold for the optical flow points, by default 2 (pixels).
    phase_corr_filter: bool, Optional
        Whether to apply phase correlation filtering, by default True.
    phase_corr_signal_thresh: float, Optional
        Signal threshold for the phase correlation filtering, by default 0.9.
    phase_corr_valid_num_points: int, Optional
        Minimum number of valid points for the phase correlation filtering, by default 10.
    rethrow_error: bool, Optional
        Whether to rethrow errors during the co-registration process, by default False.
    resampling_resolution: str, Optional
        Resolution for resampling the images, by default "lower". Can be "lower" or "higher".
    laplacian_kernel_size: Union[None, int, list], Optional
        Kernel size for the Laplacian filter, by default None (no filtering). If a list is provided, it will be applied to each target image.
    laplacian_for_targets_ids: list | None, Optional
        List of target image indices for which to apply the Laplacian filter, by default None (apply to all targets).
    lower_of_dist_thresh: Union[None, int, float], Optional
        Lower distance threshold for filtering the points, by default None (no lower threshold).
    band_number: Union[None, int], Optional
        Band number to use for the target images, by default None (use all bands). If specified, it will select the band from the target images.

    Returns
    -------
    tuple
        A tuple containing list of aligned target images, shifts applied to each target image, and IDs of the processed target images.

    Raises
    ------
    Exception
        If an error occurs during the co-registration process and `rethrow_error` is True,
        the error will be rethrown.
    """

    run_start = full_start = time.time()

    criteria = (
        cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    if (type(targets) == str) or (type(targets) == np.ndarray):
        targets = [targets]

    ref_raster = rasterio.open(reference)
    tgt_profiles = []
    for tgt in targets:
        tgt_raster = rasterio.open(tgt)
        tgt_profiles.append(tgt_raster.profile)

    ref_transform = ref_raster.profile["transform"]
    tgt_transforms = [profile["transform"] for profile in tgt_profiles]
    all_transforms = [ref_transform] + tgt_transforms
    ref_width = ref_raster.profile["width"]
    ref_height = ref_raster.profile["height"]
    tgt_widths = [profile["width"] for profile in tgt_profiles]
    tgt_heights = [profile["height"] for profile in tgt_profiles]
    all_widths = [ref_width] + tgt_widths
    all_heights = [ref_height] + tgt_heights

    transform_condition = len(set(all_transforms)) > 1
    shape_condition = (len(set(all_widths)) > 1) or (len(set(all_heights)) > 1)
    use_overlap = transform_condition or shape_condition
    if use_overlap:
        print("Using overlapping regions of the reference and target images.")
        tgt_imgs = []
        tgt_origs = []
        ref_imgs = []
        scale_factors = []
        for i, tgt in enumerate(targets):
            tgt_origs.append(flip_img(rasterio.open(tgt).read().copy().astype("uint8")))
            _, (_, _, ref_overlap, tgt_overlap), scale_facrtors_temp = find_overlap(
                reference, tgt, True, resampling_resolution=resampling_resolution
            )

            scale_factors.append(scale_facrtors_temp[1])

            if len(ref_overlap.shape) > 2:
                if ref_overlap.shape[2] == 1:
                    ref_overlap = ref_overlap[:, :, 0]
                else:
                    if band_number is not None:
                        ref_overlap = ref_overlap[:, :, band_number]
                    else:
                        ref_overlap = cv.cvtColor(ref_overlap, cv.COLOR_BGR2GRAY)

            if len(tgt_overlap.shape) > 2:
                if tgt_overlap.shape[2] == 1:
                    tgt_overlap = tgt_overlap[:, :, 0]
                else:
                    if band_number is not None:
                        tgt_overlap = tgt_overlap[:, :, band_number]
                    else:
                        tgt_overlap = cv.cvtColor(tgt_overlap, cv.COLOR_BGR2GRAY)

            ref_imgs.append(ref_overlap)
            tgt_imgs.append(tgt_overlap)
    else:
        scale_factors = [[1.0, 1.0]] * len(targets)
        if type(reference) == str:
            ref_img = flip_img(ref_raster.read().copy())
            if ref_img.shape[2] == 1:
                ref_img = ref_img[:, :, 0]
            else:
                if band_number is not None:
                    ref_img = ref_img[:, :, band_number]
                else:
                    ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
        else:
            if len(reference.shape) == 2:
                ref_img = reference
            else:
                if band_number is not None:
                    ref_img = ref_img[:, :, band_number]
                else:
                    ref_img = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
        ref_img = ref_img.astype("uint8")

        tgt_imgs = []
        tgt_origs = []
        for tgt in targets:
            if type(tgt) == str:
                tgt_raster = rasterio.open(tgt)
                img = flip_img(tgt_raster.read().copy())
                tgt_origs.append(img.astype("uint8"))
                if img.shape[2] == 1:
                    img = img[:, :, 0]
                else:
                    if band_number is not None:
                        img = img[:, :, band_number]
                    else:
                        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                tgt_imgs.append(img.astype("uint8"))
            else:
                if len(tgt.shape) == 2:
                    tgt_imgs.append(tgt.astype("uint8"))
                else:
                    tgt_imgs.append(cv.cvtColor(tgt, cv.COLOR_BGR2GRAY).astype("uint8"))
                tgt_origs.append(tgt.astype("uint8"))

    ref_imgs_temp = []
    if laplacian_kernel_size is not None:
        if laplacian_for_targets_ids is None:
            laplacian_for_targets_ids = list(range(len(tgt_imgs)))
        if use_overlap:
            for ref_img in ref_imgs:
                ref_imgs_temp.append(
                    cv.Laplacian(ref_img, cv.CV_8U, ksize=laplacian_kernel_size)
                )
            grey_refs = ref_imgs.copy()
            ref_imgs = ref_imgs_temp.copy()
            ref_imgs_temp = None
        else:
            grey_ref = ref_img.copy()
            ref_img_laplacian = cv.Laplacian(
                ref_img, cv.CV_8U, ksize=laplacian_kernel_size
            )

    tgt_imgs_temp = []
    if laplacian_kernel_size is not None:
        for i, tgt_img in enumerate(tgt_imgs):
            if i in laplacian_for_targets_ids:
                tgt_imgs_temp.append(
                    cv.Laplacian(tgt_img, cv.CV_8U, ksize=laplacian_kernel_size)
                )
            else:
                tgt_imgs_temp.append(tgt_img.copy())
        tgt_imgs = tgt_imgs_temp.copy()
        tgt_imgs_temp = None

    if export_outputs:
        if (type(reference) != str) or (any(type(el) != str for el in targets)):
            print(
                "To generate output GeoTiffs or GIF animation all inputs should be string paths to input scenes. Setting related flags to False."
            )
            export_outputs = False
            os.makedirs(output_path, exist_ok=True)

    tgt_aligned_list = []
    processed_tgt_images = []
    processed_output_images = []
    shifts = []
    process_ids = []

    if export_outputs:
        aligned_output_dir = os.path.join(output_path, "Aligned")
        os.makedirs(aligned_output_dir, exist_ok=True)
    for i, tgt_img in enumerate(tgt_imgs):
        if use_overlap:
            ref_img = ref_imgs[i]

        if laplacian_kernel_size is not None:
            if i not in laplacian_for_targets_ids:
                if use_overlap:
                    ref_img = grey_refs[i]
                else:
                    ref_img = grey_ref
            else:
                if use_overlap:
                    ref_img = ref_imgs[i]
                else:
                    ref_img = ref_img_laplacian
                print("Applying Laplacian filter...")

        try:
            p0 = cv.goodFeaturesToTrack(
                ref_img, mask=None, **of_params["feature_params"]
            )
            p1, st, _ = cv.calcOpticalFlowPyrLK(
                ref_img,
                tgt_img,
                p0,
                None,
                **of_params["lk_params"],
                criteria=criteria,
            )
            dist = np.linalg.norm(p1[st == 1] - p0[st == 1], axis=1)

            print(
                f"For target {i} ({os.path.basename(targets[i])}), found {len(p0[st == 1])} initial features."
            )
            print("Filtering features based on distance...")

            ref_good, tgt_good = filter_features(
                p0[st == 1],
                p1[st == 1],
                ref_img,
                tgt_img,
                tgt_img.shape,
                dist,
                of_dist_thresh,
                lower_of_dist_thresh,
                (i, os.path.basename(targets[i])),
            )

            if tgt_good is None:
                continue
            if tgt_good.shape[1] < 4:
                print(
                    f"""For target {i} ({os.path.basename(targets[i])}), couldn't find enough good features for target or reference. Num features: {tgt_good.shape[0]}\n"""
                )
                continue
            _, inliers = cv.estimateAffine2D(tgt_good, ref_good)

            print("Applying RANSAC filter....")
            ref_good_temp = ref_good.copy()[0, :, :]
            tgt_good_temp = tgt_good.copy()[0, :, :]
            ref_good_temp = ref_good_temp[inliers.ravel().astype(bool)]
            tgt_good_temp = tgt_good_temp[inliers.ravel().astype(bool)]

            if len(tgt_good_temp) == 0:
                print(
                    f"For target {i} ({os.path.basename(targets[i])}), no valid features found after RANSAC filtering.\n"
                )
                continue

            if phase_corr_filter:
                print(f"Applying phase correlation filter...")
                ref_good_temp, tgt_good_temp = find_corrs_shifts(
                    ref_img,
                    tgt_img,
                    ref_good_temp,
                    tgt_good_temp,
                    of_params["lk_params"]["winSize"],
                    phase_corr_signal_thresh,
                    valid_num_points=phase_corr_valid_num_points,
                )

            shift_x, shift_y = np.mean(ref_good_temp - tgt_good_temp, axis=0)
            num_features = ref_good_temp.shape[0]

            print(
                f"For target {i} ({os.path.basename(targets[i])}), Num features: {num_features}"
            )

            print(
                f"For target {i} ({os.path.basename(targets[i])}), shifts => x: {shift_x / scale_factors[i][1]}, y: {shift_y / scale_factors[i][0]} pixels.\n"
            )

            if shift_x == np.inf:
                print(
                    f"No valid shifts found for target {i} ({os.path.basename(targets[i])})\n"
                )
                continue

            shifts.append(
                (shift_x / scale_factors[i][1], shift_y / scale_factors[i][0])
            )
            process_ids.append(i)

            if export_outputs:
                out_path = os.path.join(
                    aligned_output_dir, os.path.basename(targets[i])
                )
                warp_affine_dataset(
                    targets[i],
                    out_path,
                    translation_x=shift_x / scale_factors[i][1],
                    translation_y=shift_y / scale_factors[i][0],
                )
                processed_tgt_images.append(targets[i])
                processed_output_images.append(out_path)

        except Exception as e:
            print(
                f"Algorithm did not converge for target {i} ({os.path.basename(targets[i])}) {'for the reason below:' if rethrow_error else ''}\n"
            )
            if rethrow_error:
                raise
            else:
                print(e)

    run_time = time.time() - run_start

    if export_outputs:
        generate_results_from_raw_inputs(
            reference,
            processed_output_images,
            processed_tgt_images,
            output_dir=output_path,
            shifts=shifts,
            run_time=run_time,
            target_ids=process_ids,
            gif_fps=fps,
        )

    full_time = time.time() - full_start
    print(f"Run time: {run_time} seconds")
    print(f"Total time: {full_time} seconds")

    return tgt_aligned_list, shifts, process_ids


def apply_gamma(
    data,
    gamma=0.5,
    stretch_hist: bool = False,
    adjust_hist: bool = False,
    min_max: bool = True,
    output_type: str = "uint8",
) -> np.ndarray:
    """Applies image enhancement using gamma correction and optional histogram adjustments.

    Parameters
    ----------
    data : np.ndarray
        Data to be enhanced, typically a numpy array representing an image.
    gamma : float, optional
        Gamma value for correction, by default 0.5
    stretch_hist : bool, optional
        Histogram stretching flag, by default False
    adjust_hist : bool, optional
        Equalize histogram flag, by default False
    min_max : bool, optional
        Min-max scaling flag, by default True
    output_type : str, optional
        Output data type, by default "uint8"

    Returns
    -------
    np.ndarray
        Enhanced image data as a numpy array.
    """
    data = np.power(data, gamma)
    if adjust_hist:
        data = equalize_hist(data)
    if stretch_hist:
        p2, p98 = np.percentile(data, (2, 98))
        data = rescale_intensity(data, in_range=(p2, p98), out_range="uint8")
    elif min_max:
        data *= 255 / data.max()
    return data.astype(output_type)


def query_stac_server(
    query: dict,
    server_url: str,
    use_pystac: bool = False,
    return_pystac_items: bool = False,
    max_cloud_cover: float | None = None,
    id_filter: str | None = None,
) -> list | pystac.ItemCollection:
    """
    Queries the stac-server (STAC) backend.
    This function handles pagination.
    query is a python dictionary to pass as json to the request.

    server_url example: https://landsatlook.usgs.gov/stac-server/search

    Parameters
    ----------
    query : dict
        Dictionary containing the query parameters for the STAC server.
    server_url : str
        URL of the STAC server to query.
    use_pystac : bool, optional
        If True, uses the pystac library to query the server, by default False.
    return_pystac_items : bool, optional
        If True, returns pystac items instead of raw features, by default False.
    max_cloud_cover : float, optional
        Maximum cloud cover percentage to filter items, by default None
    id_filter: str | None = None,
        Filters items without the provided pattern out.
    """

    if use_pystac:
        client = Client.open(server_url)
        search = client.search(**query)
        items = search.item_collection()

        items_list = items
        if max_cloud_cover is not None:
            print("Filtering items by cloud cover...")
            items_list = [
                item
                for item in items_list
                if item.properties["eo:cloud_cover"] < max_cloud_cover
            ]
        if id_filter is not None:
            print("Filtering items by id filter...")
            items_list = [item for item in items_list if id_filter in item.id]

        items = pystac.ItemCollection(items_list)
        if return_pystac_items:
            return items
        features = list(items)
        if len(features) == 0:
            print("No features found.")
            return []
    else:
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
            "Accept": "application/geo+json",
        }

        data = requests.post(server_url, headers=headers, json=query).json()
        error = data.get("message", "")
        if error:
            raise Exception(f"STAC-Server failed and returned: {error}")
        features = data["features"]

        context = data.get("context", {})
        if not context.get("matched"):
            if len(features) == 0:
                print("No features found.")
                return []

        if data["links"]:
            query["page"] += 1
            if context.get("limit"):
                query["limit"] = context["limit"]
            else:
                if len(features) < query["limit"]:
                    return features
                else:
                    query["limit"] = len(features)

            features = list(
                itertools.chain(
                    features, query_stac_server(query, server_url, use_pystac)
                )
            )

    return features


def find_scenes_dict(
    features: list,
    one_per_month: bool = True,
    start_end_years: list[int] = [],
    acceptance_list: list[str] = [],
    remove_duplicate_times: bool = True,
    duplicate_idx: int = 0,
    min_scenes_per_id: int | None = None,
    id_filter: str | None = None,
) -> tuple:
    """Generates a dictionary of scenes from the provided features.

    Parameters
    ----------
    features : list
        List of features to process, typically from a STAC query.
        Each feature should have an 'id', 'properties', and 'assets'.
    one_per_month : bool, optional
        Only keep one scene per month, by default True
    start_end_years : list[int], optional
        Start and end years to filter scenes, by default []
    acceptance_list : list[str], optional
        List of asset names to accept, by default []
    remove_duplicate_times : bool, optional
        Remove duplicate times based on the `duplicate_idx`, by default True
    duplicate_idx : int, optional
        Index of the duplicate time to keep, by default 0
    min_scenes_per_id : int | None, optional
        Minimum number of scenes required per ID, by default None
    id_filter: str | None = None,
        Filters items without the provided pattern out.

    Returns
    -------
    tuple
        Dictionary of scenes grouped by path/row and a list of all scenes.
    """
    scene_list = []
    scene_dict = dict()
    for feature in features:
        is_item = type(feature) != dict

        if is_item:
            feature = feature.to_dict()

        id = feature["id"]
        use_title = False
        if "title" in feature["properties"]:
            id = feature["properties"]["title"]
            use_title = True

        if (id_filter is not None) and (id_filter not in id):
            continue

        if "landsat:scene_id" in feature["properties"]:
            scene_id = feature["properties"]["landsat:scene_id"]
        else:
            scene_id = None

        assets = feature["assets"]

        if len(acceptance_list) > 0:
            acceptance_condition = all([s in assets for s in acceptance_list])
        else:
            acceptance_condition = True

        if acceptance_condition:
            scene_dict[id] = dict(scene_id=scene_id)
            for s in acceptance_list:
                url = assets[s]["href"]
                if "alternate" in assets[s]:
                    url_alternate = assets[s]["alternate"]["s3"]["href"]
                else:
                    url_alternate = url

                scene_dict[id][s] = url
                scene_dict[id][f"{s}_alternate"] = url_alternate
        else:
            warnings.warn(
                f"Acceptance list is not empty but feature {id} does not have all required assets: {acceptance_list}. Skipping feature..."
            )
            continue

    f0 = features[0].to_dict() if type(features[0]) != dict else features[0]
    if "landsat:scene_id" in f0["properties"]:
        path_rows = [k.split("_")[2] for k in scene_dict]
        time_ind = 3
    else:
        if type(features[0]) == dict:
            path_rows = ["_".join(k.split("_")[3:6]) for k in scene_dict]
            time_ind = 2
        else:
            if use_title:
                path_rows = ["_".join(k.split("_")[3:5])[0:15] for k in scene_dict]
                time_ind = 4
            else:
                path_rows = ["_".join(k.split("_")[1:3])[0:5] for k in scene_dict]
                time_ind = 2
    scene_dict_pr = {}
    for pr in path_rows:
        temp_dict = {}
        required_keys = [k for k in scene_dict if pr in k]
        for k in required_keys:
            temp_dict[k] = scene_dict[k]
        scene_dict_pr[pr] = temp_dict

    scene_dict_pr_time = {}
    for pr in scene_dict_pr:
        se = pd.Series(list(scene_dict_pr[pr].keys())).astype("str")
        g = [s.split("_")[time_ind][0:6] for s in list(scene_dict_pr[pr].keys())]
        if len(start_end_years) != 0:
            years = [
                int(s.split("_")[time_ind][0:4]) for s in list(scene_dict_pr[pr].keys())
            ]
            year_range = range(start_end_years[0], start_end_years[1] + 1)
            valid_idx = list(
                filter(lambda i: years[i] in year_range, range(len(years)))
            )
            g = [g[i] for i in range(len(g)) if i in valid_idx]
            se = se.iloc[valid_idx]
        groups = list(se.groupby(g))
        temp_dict_time = {}
        scene_list_temp = []
        for i, t in enumerate([el[0] for el in groups]):
            if type(t) == tuple:
                t = t[0]
            temp_list = []
            if one_per_month:
                temp_dict = scene_dict_pr[pr][groups[i][1].iloc[0]]
                temp_dict["scene_name"] = groups[i][1].iloc[0]
                temp_list.append(temp_dict)
            else:
                for k in list(groups[i][1]):
                    temp_dict = scene_dict_pr[pr][k]
                    temp_dict["scene_name"] = k
                    temp_list.append(temp_dict)
            if remove_duplicate_times:
                if duplicate_idx == 0:
                    times_idx = sorted(
                        np.unique(
                            [
                                re.findall(r"\d{8}", d["scene_name"])[0]
                                for d in temp_list
                            ],
                            return_index=True,
                        )[1].tolist()
                    )
                else:
                    times_list = [
                        re.findall(r"\d{8}", d["scene_name"])[0] for d in temp_list
                    ]
                    unique_times = np.unique(times_list)
                    unique_idx_list = [
                        [i for i in range(len(times_list)) if times_list[i] == ut]
                        for ut in unique_times
                    ]
                    times_idx = []
                    for i, idx in enumerate(unique_idx_list):
                        if len(idx) < duplicate_idx + 1:
                            temp_idx = len(idx) - 1
                        else:
                            temp_idx = duplicate_idx
                        times_idx.append(idx[temp_idx])
                temp_list = [temp_list[idx] for idx in times_idx]

            scene_list_temp.extend(temp_list)
            temp_dict_time[t] = temp_list

        if min_scenes_per_id is not None:
            num_scenes_in_pr = sum(
                [len(sc) for sc in [temp_dict_time[kk] for kk in temp_dict_time.keys()]]
            )
            if num_scenes_in_pr < min_scenes_per_id:
                # print(pr, "skipped for not enough scenes", num_scenes_in_pr)
                continue
        scene_dict_pr_time[pr] = temp_dict_time
        scene_list.extend(scene_list_temp)

    return scene_dict_pr_time, scene_list


def get_search_query(
    bbox: Union[list, BoundingBox],
    collections: list[str] | None = ["landsat-c2l2-sr", "landsat-c2l2-st"],
    collection_category: list[str] | None = ["T1", "T2", "RT"],
    platform: str | list | None = "LANDSAT_8",
    start_date: str = "2014-10-30T00:00:00",
    end_date: str = "2015-01-23T23:59:59",
    cloud_cover: int | None = 80,
    pystac_query: bool = False,
    extra_query: dict | None = None,
) -> dict:
    """Generates a search query for the STAC server based on the provided parameters.

    Parameters
    ----------
    bbox : Union[list, BoundingBox]
        Bounding box for the search area, either as a list of coordinates or a BoundingBox object.
    collections : list[str] | None, optional
        List of collections to search in, by default ["landsat-c2l2-sr", "landsat-c2l2-st"]
    collection_category : list[str] | None, optional
        Categories of collections to filter by, by default ["T1", "T2", "RT"]
    platform : str | list | None, optional
        Platform name or list of platform names to filter by, by default "LANDSAT_8"
    start_date : _type_, optional
        Start date for the search, by default "2014-10-30T00:00:00"
    end_date : _type_, optional
        End date for the search, by default "2015-01-23T23:59:59"
    cloud_cover : int | None, optional
        Cloud cover percentage to filter by, by default 80
    pystac_query : bool, optional
        If True, returns a query suitable for pystac, by default False
    extra_query : dict | None, optional
        Additional query parameters to include in the search, by default None

    Returns
    -------
    dict
        Dictionary representing the search query to be sent to the STAC server.
    """

    if type(bbox) != list:
        bbox = [bbox.left, bbox.bottom, bbox.right, bbox.top]
    query = {
        "bbox": bbox,
        "page": 1,
        "limit": 100,
    }
    if pystac_query:
        del query["limit"]
        del query["page"]
        platform = None
        collection_category = None
        cloud_cover = None

    query["query"] = {}
    if platform is not None:
        query["query"] = query["query"] | {
            "platform": {"in": platform if type(platform) is list else [platform]}
        }
    if collections is not None:
        query["collections"] = collections
    if collection_category is not None:
        query["query"] = query["query"] | {
            "landsat:collection_category": {"in": collection_category}
        }
    if (start_date != "") and (end_date != ""):
        query["datetime"] = f"{start_date}.000Z/{end_date}.999Z"
    if cloud_cover is not None:
        query["query"] = query["query"] | {"eo:cloud_cover": {"lte": cloud_cover}}

    if len(query["query"]) == 0:
        del query["query"]

    if extra_query is not None:
        query = query | extra_query

    return query


def make_composite_scene(
    dataset_paths: list[str] | str,
    output_path: str | None = None,
    gamma: float = 1.0,
    equalise_histogram: bool = False,
    stretch_contrast: bool = False,
    gray_scale: bool = False,
    averaging: bool = False,
    edge_detection: bool = False,
    edge_detection_mode: Literal["sobel", "laplacian", "canny"] = "canny",
    post_process_only: bool = False,
    reference_band_number: Literal[1, 2, 3] | None = None,
    preserve_depth: bool = False,
    min_max_scaling: bool = True,
    three_channel: bool = False,
    remove_nans: bool = False,
    fill_nodata: bool = False,
    fill_nodata_max_threshold: int = 10,
) -> np.ndarray:
    """Makes a composite image from the given dataset paths.

    Parameters
    ----------
    dataset_paths : list[str] | str
        List of paths to the dataset bands or a single path to a composite image.
        If a single path is provided, it is assumed to be a pre-composited image for post-processing.
    output_path : str | None, optional
        Path to save the output image, by default None
    gamma : float, optional
        Gamma correction value for the images, by default 1.0
    equalise_histogram : bool, optional
        Equalise histogram of the images, by default False
    stretch_contrast : bool, optional
        Intensity enhancement of the images, by default False
    gray_scale : bool, optional
        Use gray scale images, by default False
    averaging : bool, optional
        Use averaging for generating gray scale images instead of NTSC formula, by default False
    edge_detection : bool, optional
        Using edge detection in the processing, by default False
    edge_detection_mode : Literal[&quot;sobel&quot;, &quot;laplacian&quot;, &quot;canny&quot;], optional
        Edge detection mode, by default "canny"
    post_process_only : bool, optional
        Only post-process the image if set to True, by default False.
    reference_band_number : Literal[1, 2, 3] | None, optional
        Reference band number for the reprojecting and resampling the images to the reference image, by default None
    preserve_depth : bool, optional
        Preserve the depth of the image, by default False.
    min_max_scaling : bool, optional
        Apply min-max scaling to the image, by default True.
    three_channel : bool, optional
        If True, the output image will be a 3-channel image, by default False.
    remove_nans : bool, optional
        If True, NaN values in the image will be removed, by default False.
    fill_nodata : bool, optional
        If True, fills small nodata holes in the image, by default False.
    fill_nodata_max_threshold : int, optional
        Maximum size of nodata holes to fill, by default 10.

    Returns
    -------
    np.ndarray
        A composite image as a numpy array.

    Raises
    ------
    FileNotFoundError
        If the output file does not exist when `post_process_only` is True.
    """
    if post_process_only:
        if not os.path.isfile(dataset_paths):
            raise FileNotFoundError(f"Output file {dataset_paths} does not exist.")
        profile = rasterio.open(dataset_paths).profile
        img = rasterio.open(dataset_paths).read(masked=fill_nodata)
        if fill_nodata:
            img = fillnodata(img, max_search_distance=fill_nodata_max_threshold)
        img = flip_img(img)
    else:

        if reference_band_number is not None:
            profile = rasterio.open(dataset_paths[reference_band_number - 1]).profile
        else:
            profile = rasterio.open(dataset_paths[0]).profile

        if not gray_scale:
            profile["count"] = len(dataset_paths)

        band_imgs = []
        for b in dataset_paths:
            bp = rasterio.open(b).profile
            if bp["width"] != profile["width"] or bp["height"] != profile["height"]:
                print(
                    f"Band {b} has different dimensions than the reference profile. Downsampling to match the reference profile."
                )
                bs, _ = downsample_dataset(
                    b,
                    force_shape=(profile["height"], profile["width"]),
                    masked_data=fill_nodata,
                )
            else:
                bs = rasterio.open(b).read(masked=fill_nodata)

            if fill_nodata:
                bs = fillnodata(bs, max_search_distance=fill_nodata_max_threshold)

            bs = bs[0, :, :]
            band_imgs.append(bs)

        if remove_nans:
            band_imgs = [np.nan_to_num(b, nan=0) for b in band_imgs]

        if not preserve_depth:
            band_imgs = [np.clip(b / 256, 0, 255).astype("uint8") for b in band_imgs]
            profile["dtype"] = "uint8"

        if three_channel:
            profile["count"] = 3
            if len(band_imgs) == 1:
                band_imgs = [band_imgs[0]] * 3
                print(
                    "Only one band provided, duplicating it to create a 3-channel image."
                )
            elif len(band_imgs) == 2:
                band_imgs = [band_imgs[0], band_imgs[1], band_imgs[0]]
                print(
                    "Two bands provided, duplicating the first band to create a 3-channel image."
                )
            elif len(band_imgs) > 3:
                band_imgs = band_imgs[:3]
                print(
                    "More than three bands provided, using only the first three bands to create a 3-channel image."
                )

        img = cv.merge(band_imgs)

    if gray_scale:
        if averaging:
            img = np.mean(img, axis=2)
            if not preserve_depth:
                img = img.astype("uint8")
        else:
            if len(dataset_paths) != 3:
                raise ValueError(
                    "For gray scale images, the number of bands should be 3."
                )
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if preserve_depth:
        output_type = img.dtype
    else:
        output_type = "uint8"
    img = apply_gamma(
        img, gamma, stretch_contrast, equalise_histogram, min_max_scaling, output_type
    )
    if edge_detection:
        img = edge_detector(img, edge_detection_mode, morphology_kernels=None)

    if output_path is not None:
        with rasterio.open(output_path, "w", **profile) as ds:
            if gray_scale or img.ndim == 2:
                ds.write(img, 1)
            elif img.ndim == 3 and img.shape[2] == 1:
                ds.write(img[:, :, 0], 1)
            else:
                for i in range(profile["count"]):
                    ds.write(img[:, :, i], i + 1)

    return img


def tracking_image(
    ref_points: np.ndarray,
    tgt_points: np.ndarray,
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    line_width: int = 100,
    dot_radius: int = 100,
) -> tuple:
    """Generates an image showing the tracking of points from reference to target images.

    Parameters
    ----------
    ref_points : np.ndarray
        Coordinates of feature points in the reference image.
    tgt_points : np.ndarray
        Coordinates of feature points in the target image.
    ref_img : np.ndarray
        Reference image.
    tgt_img : np.ndarray
        Target image.
    line_width : int, optional
        Width of the lines connecting points, by default 100
    dot_radius : int, optional
        Radius of the dots marking the points, by default 100

    Returns
    -------
    tuple
        A tracking image with lines drawn between points,
        a copy of the reference image with points marked,
        and a copy of the target image with points marked.
    """

    ref_points = ref_points.astype("int")
    tgt_points = tgt_points.astype("int")

    color = np.random.randint(0, 255, (ref_points.shape[1], 3))

    ref_copy = ref_img.copy()
    for i, f in enumerate(np.squeeze(ref_points)):
        centre = (int(f[0]), int(f[1]))
        ref_copy = cv.circle(ref_copy, centre, dot_radius, color[i].tolist(), -1)

    tgt_copy = tgt_img.copy()
    for i, f in enumerate(np.squeeze(tgt_points)):
        centre = (int(f[0]), int(f[1]))
        tgt_copy = cv.circle(tgt_copy, centre, dot_radius, color[i].tolist(), -1)

    # draw the tracks
    mask = np.zeros_like(ref_img.copy())
    frame = tgt_img.copy()
    for i, (new, old) in enumerate(zip(np.squeeze(tgt_points), np.squeeze(ref_points))):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.arrowedLine(mask, (a, b), (c, d), color[i].tolist(), line_width)
    track_img = cv.add(frame, mask)
    return track_img, ref_copy, tgt_copy


def read_kml_polygon(
    kml_path: str,
    is_lat_lon: bool = True,
    is_kml_str: bool = False,
    read_mode: str = "poly",
) -> tuple:
    """Reads a KML file and extracts coordinates and bounding box.

    Parameters
    ----------
    kml_path : str
        Path to the KML file or KML string.
    is_lat_lon : bool, optional
        Is the coordinate system latitude/longitude? If False, it is UTM, by default True.
    is_kml_str : bool, optional
        Is the provided `kml_path` a string containing KML data? If True, it will create a temporary file, by default False.
    read_mode : str, optional
        Read mode for KML. Could be `poly` for polygons or `point` for points, by default "poly".

    Returns
    -------
    tuple
        Coordinates as a list of LLA3 or UTM3 objects and a BoundingBox object.
    Raises
    ------
    AssertionError
        If `read_mode` is not "poly" or "point".
    """

    assert read_mode in ["poly", "point"], "read_mode` could be `poly` or `point"
    if is_kml_str:
        os.makedirs("temp_kml_dir", exist_ok=True)
        with open("temp_kml_dir/temp_kml.kml", "w") as f:
            f.write(kml_path)
        kml_path = "temp_kml_dir/temp_kml.kml"
    with open(kml_path) as f:
        if read_mode == "poly":
            doc = parser.parse(f).getroot().Document.Placemark
        else:
            doc = parser.parse(f).getroot().Folder.Placemark
    shutil.rmtree("temp_kml_dir", ignore_errors=True)

    coords = []
    for pm in doc:
        if read_mode == "poly":
            coord = pm.Polygon.outerBoundaryIs.LinearRing.coordinates.text.strip()
        else:
            coord = pm.Point.coordinates.text.strip()
        coord_list = coord.split(" ")
        for c in coord_list:
            c_splits = c.split(",")
            if any(el == "" for el in c_splits):
                continue
            c_nums = [float(i) for i in c_splits]
            if len(c_nums) == 2:
                c_nums.append(0.0)
            if is_lat_lon:
                coords.append(LLA3(c_nums[1], c_nums[0], c_nums[2]))
            else:
                coords.append(UTM3(c_nums[0], c_nums[1], c_nums[2]))

    if is_lat_lon:
        lats = [p.lat for p in coords]
        lons = [p.lon for p in coords]
        bbox = BoundingBox(min(lons), min(lats), max(lons), max(lats))
    else:
        xs = [p.x for p in coords]
        ys = [p.y for p in coords]
        bbox = BoundingBox(min(xs), min(ys), max(xs), max(ys))

    return coords, bbox


def stream_scene(
    geotiff_file: str | Path,
    aws_session: rasterio.session.AWSSession | None = None,
    metadata_only: bool = False,
    scale_factor: float | list[float] | None = None,
    resolution: float | list[float] | None = None,
    reshape_method: Resampling = Resampling.bilinear,
    round_transform: bool = True,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Streams a GeoTIFF scene from cloud storage using rasterio.

    Parameters
    ----------
    geotiff_file : str | Path
        Cloud storage path to the GeoTIFF file, e.g. "s3://bucket/path/to/file.tif"
    aws_session : rasterio.session.AWSSession | None, optional
        AWS session for authentication if downloading from S3, by default None
    metadata_only : bool, optional
        Only retrieve metadata without reading the data, by default False
    scale_factor: float | list[float] | None, optional
        Desired output scale factor for the streamed data, (h, w) if list, by default None
    resolution: float | list[float] | None, optional
        Desired output resolution in raster units for the streamed data, (h, w) if list, overrides `scale_factor`, by default None
    reshape_method: Resampling, optional
        Resampling method used for reshaping the streamed data, by default Resampling.bilinear
    round_transform: bool, optional
        If True, round the transform parameters to the nearest integer, by default True
    """

    def get_data(gtif, sclf, res, meta_only, roundtrans):
        with rasterio.open(gtif) as geo_fp:
            prof = geo_fp.profile
            bnds = geo_fp.bounds
            gcrs = geo_fp.crs
            dtype = prof["dtype"]
            if res is not None:
                if type(res) == float:
                    res = [res] * 2
                sclf = [
                    abs(geo_fp.transform.e) / res[0],
                    abs(geo_fp.transform.a) / res[1],
                ]
            if sclf is not None:
                if type(sclf) == float:
                    sclf = [sclf] * 2
                stream_out_shape = (
                    int(prof["height"] * sclf[0]),
                    int(prof["width"] * sclf[1]),
                )
                transform = geo_fp.transform * geo_fp.transform.scale(
                    (geo_fp.width / stream_out_shape[1]),
                    (geo_fp.height / stream_out_shape[0]),
                )
                if roundtrans:
                    transform = rasterio.Affine(
                        np.round(transform.a).tolist(),
                        transform.b,
                        transform.c,
                        transform.d,
                        np.round(transform.e).tolist(),
                        transform.f,
                    )
                prof.update(
                    transform=transform,
                    width=stream_out_shape[1],
                    height=stream_out_shape[0],
                    dtype=dtype,
                )

            if meta_only:
                scn = None
            else:
                if sclf is None:
                    scn = geo_fp.read()
                else:
                    scn = geo_fp.read(
                        out_shape=(geo_fp.count, *stream_out_shape),
                        resampling=reshape_method,
                    )

        return scn, prof, bnds, gcrs

    if aws_session is not None:
        with rasterio.Env(aws_session):
            scene, profile, bounds, crs = get_data(
                geotiff_file, scale_factor, resolution, metadata_only, round_transform
            )
    else:
        scene, profile, bounds, crs = get_data(
            geotiff_file, scale_factor, resolution, metadata_only, round_transform
        )

    return scene, {"profile": profile, "bounds": bounds, "crs": crs}


def hillshade(
    array: np.ndarray,
    azimuth: float = 30.0,
    angle_altitude: float = 30.0,
    skip_negative: bool = True,
) -> np.ndarray:
    """Generates a hillshade from a 2D numpy array image.

    Parameters
    ----------
    array : np.ndarray
        Array to generate hillshade from. Should be a 2D array.
    azimuth : float, optional
        Azimuth angle in degrees, by default 30.0
    angle_altitude : float, optional
        Angle of altitude in degrees, by default 30.0
    skip_negative : bool, optional
        Skip negative values in the array, by default True.

    Returns
    -------
    np.ndarray
        Hillshade image as a 2D numpy array.
    Raises
    ------
    AssertionError
        If azimuth is greater than 360 or angle_altitude is greater than 90.
    """

    assert (
        azimuth <= 360.0
    ), "Azimuth angle should be lass than or equal to 360 degrees."
    assert (
        angle_altitude <= 90.0
    ), "Altitude angle should be lass than or equal to 90 degrees."

    if skip_negative:
        array[array < 0] = np.nan

    azimuth = 360.0 - azimuth
    azi_rad = azimuth * np.pi / 180.0  # azimuth in radians

    alt_rad = angle_altitude * np.pi / 180.0  # altitude in radians

    x, y = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)

    shaded = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos(
        (azi_rad - np.pi / 2.0) - aspect
    )

    return 255 * (shaded + 1) / 2


def download_sentinel_product(
    product: str,
    target: str = "",
    aws_access_key: str | None = None,
    aws_secret_key: str | None = None,
) -> None:
    """
    Downloads every file in bucket with provided product as prefix

    Raises FileNotFoundError if the product was not found

    Args:
        bucket: boto3 Resource bucket object
        product: Path to product
        target: Local catalog for downloaded files. Should end with an `/`. Default current directory.
    """
    # session = boto3.session.Session()
    s3 = boto3.resource(
        "s3",
        endpoint_url="https://eodata.dataspace.copernicus.eu",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name="default",
    )  # generated secrets
    bucket = s3.Bucket("eodata")
    files = bucket.objects.filter(Prefix=product)
    if not list(files):
        raise FileNotFoundError(f"Could not find any files for {product}")
    for file in files:
        os.makedirs(os.path.dirname(file.key), exist_ok=True)
        if not os.path.isdir(file.key):
            bucket.download_file(file.key, f"{target}{file.key}")
    return None


def kml_to_poly(
    kml_file: Path,
) -> Polygon | list[Polygon]:
    """Reads KML file and returns Shaply Polygon

    Parameters
    ----------
    kml_file : Path

    Returns
    -------
    Polygon
    """
    driver = ogr.GetDriverByName("KML")
    datasource = driver.Open(kml_file)
    layer = datasource.GetLayer()
    poly_list = []
    feat = layer.GetNextFeature()
    while feat is not None:
        geom = feat.geometry()
        poly = geom.ExportToIsoWkt()
        poly_list.append(ops.transform(lambda x, y, z=None: (x, y), from_wkt(poly)))
        feat = layer.GetNextFeature()
    return poly_list[0] if len(poly_list) == 1 else poly_list


def get_pair_dict(
    data_dict: dict,
    time_distance: str = Literal["closest", "farthest"],
    reference_month: str = "01",
    force_ref_id: str | None = None,
) -> list:
    """Finds the closest or farthest member of the given scene dictionary in time in it to a given reference scence found automatically in the data using the reference month.

    Parameters
    ----------
    data_dict : dict
        Scenes dictionary
    reference_id : str
        Id of the reference scenes in the scene dictionary
    time_distance : str, optional
        Distance option, by default Literal["closest", "farthest"]
    prefered_month: str, optional
        Prefered month in the scenes dict to retrieve data for the reference scene, by default 01
    force_ref_id : str | None, optional
        Force the reference id to be the provided string, by default None

    Returns
    -------
    list
        list of reference data and its closest/farthest target
    """

    data = copy.deepcopy(data_dict)

    if time_distance not in ["closest", "farthest"]:
        raise ValueError("time distance options are only closest or farthest")

    if force_ref_id is not None:
        print("Forcing reference id:", force_ref_id)
        ref_data = {}
        for date_key in list(data.keys()):
            scene_dicts = data[date_key]
            scene_ids = [s["scene_name"] for s in scene_dicts]
            if force_ref_id not in scene_ids:
                continue
            else:
                print("Using reference id:", force_ref_id)
                force_ref_idx = scene_ids.index(force_ref_id)
                ref_data = [data[date_key][force_ref_idx]]
                reference_date_obj = datetime.strptime(date_key, "%Y%m")
                del data[date_key][force_ref_idx]
                break
        assert (
            len(ref_data) == 1
        ), f"Reference data not found for force_ref_id: {force_ref_id}"

    scene_dates = sorted(list(data.keys()))

    if force_ref_id is None:
        print("Finding reference id automatically using month:", reference_month)
        available_months = [d[-2:] for d in list(data_dict.keys())]
        try:
            assert (
                reference_month in available_months
            ), f"Reference month {reference_month} not in available months: {available_months}"
        except AssertionError as e:
            print(e)
            print("Choosing first available month instead.")
            reference_month = available_months[0]
        reference_date_idx = [
            reference_month in date[-2:] for date in scene_dates
        ].index(True)
        reference_date = scene_dates[reference_date_idx]

        reference_date_obj = datetime.strptime(reference_date, "%Y%m")
        ref_data = data[reference_date]
        del data[reference_date]
        del scene_dates[reference_date_idx]

    scene_ym_objects = [datetime.strptime(date, "%Y%m") for date in scene_dates]

    still_looking = False
    if len(ref_data) > 1:
        scene_names = [d["scene_name"] for d in ref_data]
        scene_ymd_objects = [
            datetime.strptime(re.findall(r"\d{8}", sn)[0], "%Y%m%d")
            for sn in scene_names
        ]
        date_diffs = [abs(d - scene_ymd_objects[0]) for d in scene_ymd_objects[1:]]
        if ~np.all(date_diffs == timedelta(0)):
            if time_distance == "closest":
                idx = np.argmin(date_diffs) + 1
            elif len(data) == 0:
                idx = np.argmax(date_diffs) + 1
            else:
                still_looking = True
            if not still_looking:
                return [ref_data[0], ref_data[idx]]
        else:
            raise Exception("Duplicate times were found in the data.")

    if (len(ref_data) == 1) or (still_looking):
        if len(data) > 0:
            date_diffs = [abs(d - reference_date_obj) for d in scene_ym_objects]
            if ~np.all(date_diffs == timedelta(0)):
                if time_distance == "closest":
                    idx = np.argmin(date_diffs)
                else:
                    idx = np.argmax(date_diffs)
                target_data = data[list(data.keys())[idx]]
                if len(target_data) == 0:
                    raise Exception("Not enough scenes in the provided dataset")
                scene_names = [d["scene_name"] for d in target_data]
                scene_ymd_objects = [
                    datetime.strptime(re.findall(r"\d{8}", sn)[0], "%Y%m%d")
                    for sn in scene_names
                ]
                ref_ymd_object = datetime.strptime(
                    re.findall(r"\d{8}", ref_data[0]["scene_name"])[0], "%Y%m%d"
                )
                date_diffs = [abs(d - ref_ymd_object) for d in scene_ymd_objects]
                if time_distance == "closest":
                    idx = np.argmin(date_diffs)
                else:
                    idx = np.argmax(date_diffs)
                return [ref_data[0], target_data[idx]]
            else:
                raise Exception("Duplicate times were found in the data.")
        else:
            raise Exception("Not enough scenes in the provided dataset.")


def get_pair_dict_alternate(
    data_1: dict,
    data_2: dict,
    time_distance: str = Literal["closest", "farthest"],
    reference_month_1: str = "01",
    reference_month_2: str = "01",
    reference_dict: int = 1,
) -> list:
    """Alternate version of `get_pair_dict` that takes two dictionaries and returns a pair of scenes from each dictionary based on the time distance.

    Parameters
    ----------
    data_1 : dict
        Input scene dictionary 1
    data_2 : dict
        Input scene dictionary 2
    time_distance : str, optional
        Distance option, by default Literal["closest", "farthest"]
    reference_month_1 : str, optional
        Reference month for the first dictionary, by default "01"
    reference_month_2 : str, optional
        Reference month for the second dictionary, by default "01"
    reference_dict : int, optional
       Reference dictionary to use for the first scene, by default 1.

    Returns
    -------
    list
        List of two scenes, one from each dictionary, based on the time distance.

    Raises
    ------
    ValueError
        Reference dictionary should be either 1 or 2, where 1 is the first dict and 2 is the second dict.
    """
    available_months_1 = [d[-2:] for d in list(data_1.keys())]
    available_months_2 = [d[-2:] for d in list(data_2.keys())]
    try:
        assert (
            reference_month_1 in available_months_1
        ), f"Reference month {reference_month_1} not in available months: {available_months_1}"
    except AssertionError as e:
        print(e)
        print("Choosing first available month for reference 1 instead.")
        reference_month_1 = available_months_1[0]

    try:
        assert (
            reference_month_2 in available_months_2
        ), f"Reference month {reference_month_2} not in available months: {available_months_2}"
    except AssertionError as e:
        print(e)
        print("Choosing first available month for reference 2 instead.")
        reference_month_2 = available_months_2[0]

    pair_1 = get_pair_dict(
        data_1,
        time_distance=time_distance,
        reference_month=reference_month_1,
    )
    pair_2 = get_pair_dict(
        data_2,
        time_distance=time_distance,
        reference_month=reference_month_2,
    )
    if reference_dict == 1:
        return [pair_1[0], pair_2[1]]
    elif reference_dict == 2:
        return [pair_2[0], pair_1[1]]
    else:
        raise ValueError(
            "reference_dict should be either 1 or 2, where 1 is the first dict and 2 is the second dict."
        )


def combine_scene_dicts(scene_dicts=list[dict]) -> dict:
    """
    Combine multiple scene dictionaries into a single dictionary.
    """
    scene_dicts = scene_dicts.copy()
    combined = {}
    for i, d in enumerate(scene_dicts):
        keys = d.keys()
        for key in keys:
            if key not in combined:
                combined[key] = d[key]
            else:
                combined[key].extend(d[key])
    return combined


def generate_results_from_raw_inputs(
    ref_image: str,
    processed_output_images: list[str],
    tgt_images: list[str],
    output_dir: str,
    shifts: list[tuple],
    run_time: float,
    output_name: str = "output",
    target_ids: list | None = None,
    gif_fps: int = 3,
) -> None:
    """Generates results from raw inputs by creating GIFs and CSV files.

    Parameters
    ----------
    ref_image : str
        Reference image path.
    processed_output_images : list[str]
        List of processed output image paths.
    tgt_images : list[str]
        List od raw target image paths.
    output_dir : str
        Output directory where results will be saved.
    shifts : list[tuple]
        List of shifts applied to the images, each shift is a tuple of (x_shift, y_shift).
    run_time : float
        Runtime of the processing in seconds.
    output_name : str, optional
        Name of the output files, by default "output".
    target_ids : list | None, optional
        Ids of the processed target images, by default None.
    gif_fps : int, optional
        Frames per second for the output GIFs, by default 3.

    Returns
    -------
    None
    """

    if target_ids is not None:
        assert len(target_ids) == len(
            processed_output_images
        ), "target_ids should be the same length as processed_output_images"
    else:
        target_ids = list(range(len(processed_output_images)))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}.gif")
    if os.path.isfile(output_path):
        os.remove(output_path)

    tgt_aligned_list = []
    ref_imgs = []
    for tgt in processed_output_images:
        _, (_, _, ref_overlap, tgt_overlap), _ = find_overlap(ref_image, tgt, True)
        ref_imgs.append(ref_overlap)
        tgt_aligned_list.append(tgt_overlap)

    datasets_paths = [ref_image] + processed_output_images
    ssims_aligned = [
        np.round(ssim(ref_imgs[id], tgt_aligned_list[id], win_size=3), 3)
        for id in range(len(tgt_aligned_list))
    ]
    mse_aligned = [
        np.round(mse(ref_imgs[id], tgt_aligned_list[id]), 3)
        for id in range(len(tgt_aligned_list))
    ]
    target_titles = [f"target_{str(i)}" for i in target_ids]
    datasets_titles = ["Reference"] + [
        f"{target_title}, ssim:{ssim_score}, mse:{mse_score}"
        for target_title, ssim_score, mse_score in zip(
            target_titles, ssims_aligned, mse_aligned
        )
    ]
    make_difference_gif(
        datasets_paths,
        output_path,
        datasets_titles,
        mosaic_scenes=True,
        fps=gif_fps,
    )

    output_path = os.path.join(output_dir, f"{output_name}_raw.gif")
    if os.path.isfile(output_path):
        os.remove(output_path)

    tgt_raw_list = []
    ref_imgs = []
    for tgt in tgt_images:
        _, (_, _, ref_overlap, tgt_overlap), _ = find_overlap(ref_image, tgt, True)
        ref_imgs.append(ref_overlap)
        tgt_raw_list.append(tgt_overlap)

    datasets_paths = [ref_image] + tgt_images
    ssims_aligned_raw = [
        np.round(ssim(ref_imgs[id], tgt_raw_list[id], win_size=3), 3)
        for id in range(len(tgt_raw_list))
    ]
    mse_aligned_raw = [
        np.round(mse(ref_imgs[id], tgt_raw_list[id]), 3)
        for id in range(len(tgt_raw_list))
    ]
    datasets_titles = ["Reference"] + [
        f"{target_title}, ssim:{ssim_score}, mse:{mse_score}"
        for target_title, ssim_score, mse_score in zip(
            target_titles, ssims_aligned_raw, mse_aligned_raw
        )
    ]
    make_difference_gif(
        datasets_paths,
        output_path,
        datasets_titles,
        mosaic_scenes=True,
        fps=gif_fps,
    )

    output_path = os.path.join(output_dir, f"{output_name}.csv")
    if os.path.isfile(output_path):
        os.remove(output_path)
    out_ssim_df = pd.DataFrame(
        zip(
            target_titles,
            ssims_aligned_raw,
            mse_aligned_raw,
            ssims_aligned,
            mse_aligned,
            [np.round(run_time, 2).tolist()] * len(target_titles),
            [
                tuple([np.round(el.tolist(), 3).tolist() for el in shift])
                for shift in shifts
            ],
        ),
        columns=[
            "Title",
            "SSIM Raw",
            "MSE Raw",
            "SSIM Aligned",
            "MSE Aligned",
            "Run Time",
            "Shifts",
        ],
        index=None,
    )
    out_ssim_df.to_csv(output_path, encoding="utf-8")

    return None


def download(
    chunk: tuple,
    bucket: str,
    s3_client: boto3.client,
):
    """Downloads files from an S3 bucket.

    Parameters
    ----------
    chunk : tuple
        Chunk of URLs and local paths to download.
    bucket : str
        Bucket name to download files from.
    s3_client : boto3.client
        An S3 client to use for downloading files.
    """
    tasks = []
    for url, path in zip(*chunk):
        print(f"downloading {os.path.basename(path)}")
        item = "/".join(url.split("/")[3:])
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tasks.append(
                s3_client.download_file(
                    bucket, item, path, {"RequestPayer": "requester"}
                )
            )
        else:
            print(f"{os.path.basename(path)} already exists")
            continue
    return tasks


async def async_download(
    chunk: tuple,
    bucket: str,
    s3_client: boto3.client,
):
    """Asynchronously downloads files from S3 bucket.

    Parameters
    ----------
    chunk : tuple
        Chunk of URLs and local paths to download.
    bucket : str
        Bucket name to download files from.
    s3_client : boto3.client
        An S3 client to use for downloading files.
    """
    tasks = download(chunk, bucket, s3_client)
    await asyncio.gather(*tasks)


def download_files(
    bucket_name: str,
    s3_urls: list[str],
    local_paths: list[str],
    num_tasks: int = 8,
    is_async_download: bool = False,
):
    """Downloads files from S3 bucket to local paths.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to download files from.
    s3_urls : list[str]
        List of S3 URLs to download files from.
    local_paths : list[str]
        List of local paths to save the downloaded files.
    num_tasks : int, optional
        Number of tasks to run in parallel, by default 8.
    is_async_download : bool, optional
        Is the download asynchronous, by default False.
    """
    s3_client = boto3.client("s3")

    download_list_chunk = (
        [(s3_urls[i::num_tasks], local_paths[i::num_tasks]) for i in range(num_tasks)]
        if num_tasks != -1
        else [(s3_urls, local_paths)]
    )

    for i, ch in enumerate(download_list_chunk):
        if is_async_download:
            asyncio.run(async_download(ch, bucket_name, s3_client))
        else:
            download(ch, bucket_name, s3_client)
        print(f"Chunk {i + 1} downloaded")


def karios(
    ref_image: str,
    tgt_images: list[str],
    output_dir: str,
    karios_executable: str = "karios",
) -> tuple:
    """Runs Karios coregistration on the provided reference and target images.

    Parameters
    ----------
    ref_image : str
        Path to the reference image.
    tgt_images : list[str]
        List of target image paths to be coregistered with the reference image.
    output_dir : str
        Output directory where the aligned images will be saved.
    karios_executable : str
        Path to the Karios executable. Default is 'karios'.

    Returns
    -------
    tuple
        Dictionary of shifts and list of processed target IDs.
    """
    os.makedirs(output_dir, exist_ok=True)
    tgt_images_copy = tgt_images.copy()
    run_start = full_start = time.time()
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    ref_profile = rasterio.open(ref_image).profile
    tgt_profiles = [rasterio.open(t).profile for t in tgt_images_copy]
    for i, tgt_profile in enumerate(tgt_profiles):
        downsample = False
        if tgt_profile["height"] != ref_profile["height"]:
            print(
                f"Target image {tgt_images_copy[i]} has different height than reference image {ref_image}"
            )
            downsample = True
        if tgt_profile["width"] != ref_profile["width"]:
            print(
                f"Target image {tgt_images_copy[i]} has different width than reference image {ref_image}"
            )
            downsample = True
        if downsample:
            downsample_dataset(
                tgt_images_copy[i],
                force_shape=(ref_profile["height"], ref_profile["width"]),
                output_file=f"{output_dir}/temp/{os.path.basename(tgt_images_copy[i])}",
            )
            tgt_images_copy[i] = (
                f"{output_dir}/temp/{os.path.basename(tgt_images_copy[i])}"
            )

    log_file = f"{output_dir}/karios.log"
    if os.path.isfile(log_file):
        os.remove(log_file)
    for i, tgt_image in enumerate(tgt_images_copy):
        try:
            cmd = f"{karios_executable} process {tgt_image} {ref_image} --out {output_dir} --log-file-path {log_file} "
            print(f"Running {cmd}")
            run(shlex.split(cmd))
        except Exception as e:
            print(f"Error running karios for {tgt_image}: {e}")
            continue

    shutil.rmtree(temp_dir, ignore_errors=True)
    tgt_images_copy = tgt_images.copy()

    scene_names = []
    shifts = []
    target_ids = []
    for i, tgt_image in enumerate(tgt_images_copy):
        line_found = False
        with open(log_file, "r") as f:
            for line in f:
                if (os.path.basename(tgt_image) in line) and ("Process" in line):
                    line_found = True
                if line_found:
                    if "DX/DY(KLT) MEAN" in line:
                        splits = line.strip().split(" ")
                        if splits[-3] != "nan" or splits[-1] != "nan":
                            scene_names.append(tgt_image)
                            shifts.append([float(splits[-3]), float(splits[-1])])
                            target_ids.append(i)
                        break

    shifts_dict = {}
    for f, sh in zip(scene_names, shifts):
        shifts_dict[f] = sh

    os.makedirs(f"{output_dir}/Aligned", exist_ok=True)
    processed_output_images = []
    processed_tgt_images = []
    final_shifts = []
    for key in list(shifts_dict.keys()):
        output_path = os.path.join(f"{output_dir}/Aligned", os.path.basename(key))
        shift_x, shift_y = shifts_dict[key]
        warp_affine_dataset(
            key, output_path, translation_x=shift_x, translation_y=shift_y
        )
        processed_output_images.append(output_path)
        processed_tgt_images.append(key)
        final_shifts.append((np.float64(shift_x), np.float64(shift_y)))

    run_time = time.time() - run_start
    generate_results_from_raw_inputs(
        ref_image,
        processed_output_images,
        processed_tgt_images,
        output_dir=output_dir,
        shifts=final_shifts,
        run_time=run_time,
        target_ids=target_ids,
    )
    full_time = time.time() - full_start
    print(f"Run time: {run_time} seconds")
    print(f"Total time: {full_time} seconds")

    return shifts_dict, target_ids


def arosics(
    ref_image: str,
    tgt_images: list[str],
    output_dir: str,
    max_points: int = None,
    r_b4match: int = 1,
    s_b4match: int = 1,
    max_iter: int = 5,
    max_shift: int = 5,
    grid_res: int = 250,
    min_reliability: int = 30,
    tieP_filter_level: int = 3,
    rs_max_outlier: float = 10,
    rs_tolerance: float = 2.5,
    existing_ref_image: str | None = None,
    existing_tgt_images: list[str] | None = None,
) -> tuple:
    """Runs AROSICS coregistration on the provided reference and target images.

    Parameters
    ----------
    ref_image : str
        Reference image path.
    tgt_images : list[str]
        List of target image paths to be coregistered with the reference image.
    output_dir : str
        Output directory where the aligned images will be saved.
    max_points : int, optional
        MAx number of points to be used for coregistration, by default None
    r_b4match : int, optional
        Reference band number for matching, by default 1
    s_b4match : int, optional
        Target band number for matching, by default 1
    max_iter : int, optional
        Max number of iterations for coregistration, by default 5
    max_shift : int, optional
        Maximum allowed shift in pixels, by default 5
    grid_res : int, optional
        Local grid resolution in pixels, by default 250
    min_reliability : int, optional
        Minimum tie point reliability percentage, by default 30
    tieP_filter_level : int, optional
        Tie point filter level, by default 3
    rs_max_outlier : float, optional
        Maximum outlier ratio for robust statistics, by default 10
    rs_tolerance : float, optional
        Tolerance for robust statistics, by default 2.5
    existing_ref_image : str | None, optional
        Existing reference image to force reference bounding box, by default None
    existing_tgt_images : list[str] | None, optional
        Existing target images to force target bounding boxes, by default None

    Returns
    -------
    tuple
        List of shifts applied to each target image in the format [(shift_x, shift_y), ...] and
        List of target IDs corresponding to the shifts.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_start = full_start = time.time()
    tgt_images_copy = tgt_images.copy()
    local_outputs = [
        os.path.join(
            f"{output_dir}/Aligned",
            os.path.basename(tgt),
        )
        for tgt in tgt_images_copy
    ]
    os.makedirs(f"{output_dir}/Aligned", exist_ok=True)

    processed_output_images = []
    processed_tgt_images = []
    target_ids = []
    shifts = []
    print(f"Reference image: {ref_image}")
    for i, tgt_image in enumerate(tgt_images_copy):
        print(f"Coregistering {tgt_image}")
        coreg_local = COREG_LOCAL(
            im_ref=ref_image,
            im_tgt=tgt_image,
            grid_res=grid_res,
            max_points=max_points,
            path_out=local_outputs[i],
            fmt_out="GTIFF",
            nodata=(0.0, 0.0),
            r_b4match=r_b4match,
            s_b4match=s_b4match,
            align_grids=True,
            max_iter=max_iter,
            max_shift=max_shift,
            ignore_errors=True,
            min_reliability=min_reliability,
            tieP_filter_level=tieP_filter_level,
            rs_max_outlier=rs_max_outlier,
            rs_tolerance=rs_tolerance,
            footprint_poly_ref=(
                None
                if existing_ref_image is None
                else box(*rasterio.open(existing_ref_image).bounds).wkt
            ),
            footprint_poly_tgt=(
                None
                if existing_tgt_images is None
                else box(*rasterio.open(existing_tgt_images[i]).bounds).wkt
            ),
        )
        try:
            coreg_local.correct_shifts()
            if not coreg_local.success:
                print(f"Coregistration was not successfull for {tgt_image}.")
                if os.path.isfile(local_outputs[i]):
                    print(f"Removing the corresponding output: {local_outputs[i]}")
                    os.remove(local_outputs[i])
            else:
                if existing_tgt_images is not None:
                    tgt_image = existing_tgt_images[i]
                    warp_affine_dataset(
                        tgt_image,
                        local_outputs[i],
                        translation_x=coreg_local.coreg_info["mean_shifts_px"]["x"],
                        translation_y=coreg_local.coreg_info["mean_shifts_px"]["y"],
                    )

                processed_output_images.append(local_outputs[i])
                processed_tgt_images.append(tgt_image)
                target_ids.append(i)
                shifts.append(
                    (
                        coreg_local.coreg_info["mean_shifts_px"]["x"],
                        coreg_local.coreg_info["mean_shifts_px"]["y"],
                    )
                )
        except:
            print(f"Coregistration was not successfull for {tgt_image}.")
            if os.path.isfile(local_outputs[i]):
                print(f"Removing the corresponding output: {local_outputs[i]}")
                os.remove(local_outputs[i])

    if existing_ref_image is not None:
        ref_image = existing_ref_image
    run_time = time.time() - run_start
    generate_results_from_raw_inputs(
        ref_image,
        processed_output_images,
        processed_tgt_images,
        output_dir=output_dir,
        shifts=shifts,
        run_time=run_time,
        target_ids=target_ids,
    )
    full_time = time.time() - full_start
    print(f"Run time: {run_time} seconds")
    print(f"Total time: {full_time} seconds")

    return shifts, target_ids


def edge_detector(
    img,
    mode: Literal["sobel", "laplacian", "canny"] = "canny",
    laplacian_kernel_size: int = 5,
    morphology_kernels: list[tuple[int, int]] | None = [(3, 3), (15, 15)],
) -> np.ndarray:
    """
    Applies edge detection to the input image using the specified mode."""
    if mode == "laplacian":
        edges = Laplacian(img, cv.CV_8U, ksize=laplacian_kernel_size)
    elif mode == "canny":
        edges = Canny(img, 100, 200)
        if morphology_kernels is not None:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv.dilate(edges, kernel, iterations=2)
            edges = cv.erode(edges, kernel, iterations=3)
            kernel = np.ones((15, 15), np.uint8)
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=10)
            edges = cv.Canny(edges, 0, 255)
    elif mode == "sobel":
        grad_x = Sobel(img, cv.CV_64F, 1, 0)
        grad_y = Sobel(img, cv.CV_64F, 0, 1)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        edges = (grad * 255 / grad.max()).astype(np.uint8)
    else:
        raise ValueError("Edge detection mode must be 'sobel', 'laplacian' or 'canny'.")
    return edges


def process_existing_outputs(
    existing_files: list[str] | list[list[str]],
    output_dir: str,
    edge_detection: bool = False,
    edge_detection_mode: Literal["sobel", "laplacian", "canny"] = "sobel",
    gamma: float = 1.0,
    equalise_histogram: bool = False,
    stretch_contrast: bool = False,
    gray_scale: bool = False,
    averaging: bool = False,
    subdir: str = "true_colour",
    force_reprocess: bool = False,
    preserve_depth: bool = False,
    min_max_scaling: bool = True,
    three_channel: bool = False,
    remove_nans: bool = False,
    scale_factor: float = 0.2,
    filename_suffix: str = "PROC",
    num_cpu: int = 1,
    write_pairs: bool = True,
    reference_band_number: int | None = None,
    fill_nodata: bool = False,
    fill_nodata_max_threshold: int = 10,
) -> None:
    """Processes existing files into composite scenes and saves them to the specified output directory.

    Parameters
    ----------
    existing_files : list[str] | list[list[str]]
        List of existing file paths to be processed.
    output_dir : str
        Output directory where processed files will be saved.
    edge_detection : bool, optional
        Using edge detection in the processing, by default False
    edge_detection_mode : Literal[&quot;sobel&quot;, &quot;laplacian&quot;, &quot;canny&quot;], optional
        Edge detection mode, by default "canny"
    gamma : float, optional
        Gamma correction value for the images, by default 1.0
    equalise_histogram : bool, optional
        Equalise histogram of the images, by default False
    stretch_contrast : bool, optional
        Intensity enhancement of the images, by default False
    gray_scale : bool, optional
        Use gray scale images, by default False
    averaging : bool, optional
        Use averaging for generating gray scale images instead of NTSC formula, by default False
    subdir : str, optional
        Subdirectory name for processed images, by default "true_color"
    force_reprocess : bool, optional
        Force reprocessing of existing files, by default False
    preserve_depth : bool, optional
        Preserve the depth of the original images, by default False
    min_max_scaling : bool, optional
        Use min-max scaling for the images, by default True
    three_channel : bool, optional
        If True, the composite image will have three channels, by default False
    remove_nans : bool, optional
        If True, removes NaN values from the processed images, by default False
    scale_factor: float, optional
        Scale factor for downsampling the images, by default 0.2
    file_name_suffix: str, optional
        Suffix to append to the processed file names, by default "PROC"
    num_cpu: int, optional
        Number of CPU cores to use for processing, by default 1
    write_pairs: bool = True,
        Whether to write image pairs to the output directory, by default True
    reference_band_number : int | None, optional
        Reference band number for the reprojecting and resampling the images to the reference image, by default None
    """
    os.makedirs(output_dir, exist_ok=True)

    process_dir = f"{output_dir}/{subdir}"
    process_ds_dir = f"{output_dir}/{subdir}_ds"

    if force_reprocess:
        print("Force reprocessing is enabled, all existing files will be reprocessed.")
        shutil.rmtree(process_dir, ignore_errors=True)
        shutil.rmtree(process_ds_dir, ignore_errors=True)

    os.makedirs(process_dir, exist_ok=True)
    if scale_factor != 1.0:
        os.makedirs(process_ds_dir, exist_ok=True)

    proc_files = []
    proc_files_ds = []
    proc_file_to_process = []
    proc_file_ds_to_process = []
    existing_files_to_process = []
    proc_for_ds = []
    for file in existing_files:
        if type(file) is str:
            proc_file = os.path.join(process_dir, os.path.basename(file))
        else:
            ext = os.path.splitext(file[0])[1]
            originals_dir = os.path.basename(os.path.dirname(file[0]))
            original_file_name = os.path.basename(file[0])
            suffix_list = list(
                filter(
                    lambda x: x != "" and ext not in x,
                    original_file_name.replace(originals_dir, "").split("_"),
                )
            )
            if len(suffix_list) > 0:
                suffix_to_add = f"{suffix_list[0]}_{filename_suffix}"
            else:
                suffix_to_add = filename_suffix
            proc_file = f"{os.path.join(process_dir, os.path.basename(originals_dir))}_{suffix_to_add}{ext}"

        proc_file_ds = os.path.join(process_ds_dir, os.path.basename(proc_file))
        proc_files.append(proc_file)
        proc_files_ds.append(proc_file_ds)

        if os.path.isfile(proc_file):
            print(f"Scene {proc_file} already exists, skipping.")
        else:
            if num_cpu == 1:
                make_composite_scene(
                    file,
                    proc_file,
                    gamma,
                    equalise_histogram,
                    stretch_contrast,
                    gray_scale,
                    averaging,
                    edge_detection,
                    edge_detection_mode,
                    True if type(file) is str else False,
                    reference_band_number,
                    preserve_depth,
                    min_max_scaling,
                    three_channel,
                    remove_nans,
                    fill_nodata,
                    fill_nodata_max_threshold,
                )
            else:
                proc_file_to_process.append(proc_file)
                existing_files_to_process.append(file)

        if scale_factor != 1.0:
            if os.path.isfile(proc_file_ds):
                print(f"Scene {proc_file_ds} already exists, skipping.")
            else:
                if num_cpu == 1:
                    downsample_dataset(proc_file, scale_factor, proc_file_ds)
                else:
                    proc_for_ds.append(proc_file)
                    proc_file_ds_to_process.append(proc_file_ds)

    if num_cpu == -1:
        num_cpu = mp.cpu_count() - 2

    if num_cpu > 1:
        print(f"Using {num_cpu} CPU cores.")
        with mp.Pool(num_cpu) as pool:
            pool.starmap(
                make_composite_scene,
                [
                    (
                        file,
                        proc_file,
                        gamma,
                        equalise_histogram,
                        stretch_contrast,
                        gray_scale,
                        averaging,
                        edge_detection,
                        edge_detection_mode,
                        True if type(file) is str else False,
                        reference_band_number,
                        preserve_depth,
                        min_max_scaling,
                        three_channel,
                        remove_nans,
                        fill_nodata,
                        fill_nodata_max_threshold,
                    )
                    for file, proc_file in list(
                        zip(existing_files_to_process, proc_file_to_process)
                    )
                ],
            )
            if scale_factor != 1.0:
                pool.starmap(
                    downsample_dataset,
                    [
                        (
                            proc_file,
                            scale_factor,
                            proc_file_ds,
                        )
                        for proc_file, proc_file_ds in list(
                            zip(proc_for_ds, proc_file_ds_to_process)
                        )
                    ],
                )

    if write_pairs:
        cols = ["Reference", "Closest_target", "Farthest_target"]
        df = pd.DataFrame(
            {
                cols[i]: [
                    file,
                    proc_files_ds[i] if scale_factor != 1.0 else "",
                ]
                for i, file in enumerate(proc_files)
            },
            columns=cols,
        )
        df.to_csv(
            f"{output_dir}/pairs.csv",
            index=False,
        )
    return None


def download_and_process_series(
    data: list,
    bands: list[str],
    bands_suffixes: list[str],
    output_dir: str,
    process_dir: str | None = None,
    process_ds_dir: str | None = None,
    aws_session: rasterio.session.AWSSession | None = None,
    keep_original_band_scenes: bool = False,
    edge_detection: bool = False,
    edge_detection_mode: Literal["sobel", "laplacian", "canny"] = "canny",
    gamma: float = 1.0,
    equalise_histogram: bool = False,
    stretch_contrast: bool = False,
    gray_scale: bool = False,
    averaging: bool = False,
    reference_band_number: int | None = None,
    filename_suffix: str = "PROC",
    download_only: bool = False,
    composite_band_indexes: list[int] | None = None,
    scale_factor: float | list[float] = 0.2,
    scene_name_map: Callable | None = None,
    preserve_depth: bool = False,
    min_max_scaling: bool = True,
    extra_bands: list[str] | None = None,
    three_channel: bool = False,
    remove_nans: bool = False,
    force_reprocess: bool = False,
    stream_out_scale_factor: float | list[float] | None = None,
    stream_reshape_method: Resampling = Resampling.bilinear,
    stream_round_transform: bool = True,
    fill_nodata: bool = False,
    fill_nodata_max_threshold: int = 10,
) -> list:
    """Downloads and processes a series of scenes from AWS S3, creating composite images from the specified bands.

    Parameters
    ----------
    data : dict | list
        Dictionary or list containing scene data.
    bands : list[str]
        Bands to be used for processing, e.g. ["red", "green", "blue"] for true color.
    bands_suffixes : list[str]
        Suffixes for the bands, e.g. ["_B04", "_B03", "_B02"] for Sentinel-2.
    output_dir : str
        Directory where processed scenes will be saved.
    process_dir : str
        Directory where original band scenes will be saved.
    process_ds_dir : str
        Directory where downsampled scenes will be saved.
    aws_session : rasterio.session.AWSSession | None, optional
        AWS session for interacting with AWS, where required, by default None
    keep_original_band_scenes : bool, optional
        Keep original band scenes in the output directory, by default False
    edge_detection : bool, optional
        Using edge detection in the processing, by default False
    edge_detection_mode : Literal[&quot;sobel&quot;, &quot;laplacian&quot;, &quot;canny&quot;], optional
        Edge detection mode, by default "canny"
    gamma : float, optional
        Gamma correction value for the images, by default 1.0
    equalise_histogram : bool, optional
        Equalise histogram of the images, by default False
    stretch_contrast : bool, optional
        Intensity enhancement of the images, by default False
    gray_scale : bool, optional
        Use gray scale images, by default False
    averaging : bool, optional
        Use averaging for generating gray scale images instead of NTSC formula, by default False
    reference_band_number : int | None, optional
        Reference band number for the reprojecting and resampling the images to the reference image, by default None
    filename_suffix : str, optional
        Suffix to be added to the processed files, by default "PROC"
    download_only : bool, optional
        If True, only downloads the scenes without processing them, by default False
    composite_band_indexes : list[int] | None, optional
        List of indexes for the bands to be used in the composite image, by default None
    scale_factor : float | list[float], optional
        Scale factor for downsampling the processed scenes, by default 0.2
    scene_name_map : Callable | None, optional
        Function to map scene names to a specific format, by default None
    preserve_depth : bool, optional
        Preserve the depth of the original images, by default False
    min_max_scaling : bool, optional
        Use min-max scaling for the images, by default True
    extra_bands : list[str] | None, optional
        Additional bands to be downloaded, by default None
    three_channel : bool, optional
        If True, the composite image will have three channels, by default False
    remove_nans : bool, optional
        If True, removes NaN values from the processed images, by default False
    force_reprocess : bool, optional
        If True, forces reprocessing of existing files, by default False
    stream_out_scale_factor: float | list[float] | None, optional
        Desired output scale factor for the streamed data (h, w) if list, by default None
        `scale_factor` resamples the streamed data, even if the streamed data already is reshaped.
    stream_reshape_method: Resampling = Resampling.bilinear
        Resampling method used for reshaping the streamed data.
    stream_round_transform: bool = True
        Rounds the transform of the streamed data to the nearest integer.
    fill_nodata: bool = False
        If True, fills nodata values in the streamed data using inverse distance weighting interpolation.
    fill_nodata_max_threshold: int = 10
        Maximum size of nodata regions to fill when `fill_nodata` is True.

    Returns
    -------
    list
        List of dictionaries with scene data, including local paths to the processed scenes.
    Raises
    ------
    ValueError
        If a band is not found in the scene data.
    """
    if composite_band_indexes is None:
        composite_band_indexes = list(range(len(bands)))

    if len(bands) != 3:
        warnings.warn(
            f"Number of bands is different from 3 , The composite image will be created using {len(bands)}.",
            UserWarning,
        )

    os.makedirs(output_dir, exist_ok=True)
    if not download_only:
        assert (
            process_dir is not None
        ), "process_dir must be provided if not in download_only mode."
        assert (
            process_ds_dir is not None
        ), "process_ds_dir must be provided if not in download_only mode."
        os.makedirs(process_dir, exist_ok=True)
        os.makedirs(process_ds_dir, exist_ok=True)

    ext = os.path.splitext(os.path.basename(data[0][bands[0]]))[1]
    print("Using file extension:", ext)

    output_splits = output_dir.split("_")
    for j, el in enumerate(data):
        path_row = ""
        for s in output_splits:
            if f"_{s}_" in el["scene_name"]:
                path_row = s
                break
        path_row_str = f" and path_row: {path_row}" if path_row else ""
        print(
            f"Now downloading and processing scenes for {el['scene_name']}{path_row_str}, scene {j + 1} of {len(data)}.",
        )
        originals_dir = f"{output_dir}/Originals/{el['scene_name']}"
        if scene_name_map is not None:
            el["scene_name"] = scene_name_map(el["scene_name"])
        new_originals_dir = f"{output_dir}/Originals/{el['scene_name']}"
        os.makedirs(new_originals_dir, exist_ok=True)

        post_process_only = False
        proc_exists = False
        proc_ds_exists = False
        proc_file = f"{os.path.join(process_dir, os.path.basename(originals_dir))}_{filename_suffix}{ext}"
        proc_file_ds = (
            os.path.join(process_ds_dir, os.path.basename(proc_file))
            if scale_factor != 1.0
            else ""
        )

        if not download_only:
            if os.path.isfile(proc_file):
                el["local_path"] = proc_file
                if (
                    (not edge_detection)
                    and (not equalise_histogram)
                    and (not stretch_contrast)
                ):
                    print(f"Scene {proc_file} already exists, skipping.")
                    proc_exists = True
                elif force_reprocess:
                    print(
                        f"Scene {proc_file} already exists, running post-processing only."
                    )
                    post_process_only = True
                else:
                    print(f"Scene {proc_file} already exists, skipping.")
                    proc_exists = True

            if os.path.isfile(proc_file_ds):
                print(f"Scene {proc_file_ds} already exists, skipping.")
                el["local_path_ds"] = proc_file_ds
                proc_ds_exists = True

        if not proc_exists:
            to_download = bands.copy()
            if extra_bands is not None:
                to_download.extend(extra_bands)
            for band in to_download:
                if band not in el:
                    raise ValueError(f"Band {band} not found in the scene data.")
                band_url = el[band + "_alternate"]

                band_output = os.path.join(
                    new_originals_dir, os.path.basename(band_url)
                )
                if os.path.isfile(band_output):
                    print(f"Original file for {band} band already exists, skipping.")
                else:
                    band_img, band_meta = stream_scene(
                        band_url,
                        aws_session,
                        scale_factor=stream_out_scale_factor,
                        reshape_method=stream_reshape_method,
                        round_transform=stream_round_transform,
                    )
                    with rasterio.open(band_output, "w", **band_meta["profile"]) as ds:
                        for i in range(band_meta["profile"]["count"]):
                            ds.write(band_img[i, :, :], i + 1)

        if download_only:
            el["local_path"] = proc_file
            el["local_path_ds"] = proc_file_ds
            print("Download only mode is enabled, skipping processing.")
            continue

        if not proc_exists:
            files = glob.glob(f"{new_originals_dir}/**")
            proc_bands = []
            for bi in range(len(bands)):
                proc_band = list(
                    filter(
                        lambda f: f.endswith(
                            f"{bands_suffixes[composite_band_indexes[bi]]}{ext}"
                        ),
                        files,
                    )
                )[0]
                proc_bands.append(proc_band)

            if post_process_only:
                to_process = proc_file
            else:
                to_process = proc_bands

            make_composite_scene(
                to_process,
                proc_file,
                gamma,
                equalise_histogram,
                stretch_contrast,
                gray_scale,
                averaging,
                edge_detection,
                edge_detection_mode,
                post_process_only,
                reference_band_number,
                preserve_depth,
                min_max_scaling,
                three_channel,
                remove_nans,
                fill_nodata,
                fill_nodata_max_threshold,
            )
        if not proc_ds_exists and scale_factor != 1.0:
            downsample_dataset(proc_file, scale_factor, proc_file_ds)

        el["local_path"] = proc_file
        el["local_path_ds"] = proc_file_ds

        if not keep_original_band_scenes:
            shutil.rmtree(
                f"{output_dir}/Originals",
                ignore_errors=True,
            )

    print("Processing scenes done!")
    return data


def get_band_suffixes(
    data: dict,
    bands: list[str],
) -> list[str]:
    """Extracts band suffixes from the provided data dictionary or list.

    Parameters
    ----------
    data : dict | list
        Dictionary containing scene data, where each entry has a date key and a list of band URLs.
    bands : list[str]
        List of bands for which to extract the suffixes, e.g. ["red", "green", "blue"].

    Returns
    -------
    list[str]
        List of band suffixes corresponding to the provided bands.
    """
    bands_suffixes = []
    for band in bands:
        if band not in data:
            raise ValueError(f"Band {band} not found in the data.")
        band_url = data[band]
        band_suffix = os.path.splitext(os.path.basename(band_url))[0].split("_")
        band_suffix = band_suffix[-1] if band_suffix else ""
        bands_suffixes.append(band_suffix)
    return bands_suffixes


def _get_band_suffixes(
    data: dict | list,
    bands: list[str],
) -> list[str]:
    """Extracts band suffixes from the provided data dictionary or list.

    Parameters
    ----------
    data : dict | list
        Dictionary or list containing scene data, where each entry has a date key and a list of band URLs.
    bands : list[str]
        List of bands for which to extract the suffixes, e.g. ["red", "green", "blue"].

    Returns
    -------
    list[str]
        List of band suffixes corresponding to the provided bands.
    """
    date_dict = data[0] if type(data) == list else data

    date = list(date_dict.keys())[0]

    bands_suffixes = []
    for band in bands:
        if band not in date_dict[date][0]:
            raise ValueError(f"Band {band} not found in the data for date {date}.")
        band_url = date_dict[date][0][band]
        band_suffix = os.path.splitext(os.path.basename(band_url))[0].split("_")
        band_suffix = band_suffix[-1] if band_suffix else ""
        bands_suffixes.append(band_suffix)
    return bands_suffixes


def download_and_process_pairs(
    data: dict | list,
    bands: list[str],
    output_dir: str,
    aws_session: rasterio.session.AWSSession | None = None,
    keep_original_band_scenes: bool = False,
    reference_month: str | list = "01",
    edge_detection: bool = False,
    edge_detection_mode: Literal["sobel", "laplacian", "canny"] = "canny",
    gamma: float = 1.0,
    equalise_histogram: bool = False,
    stretch_contrast: bool = False,
    gray_scale: bool = False,
    averaging: bool = False,
    subdir: str = "true_color",
    force_reprocess: bool = False,
    reference_band_number: int | None = None,
    filename_suffix: str = "PROC",
    fill_nodata: bool = False,
    fill_nodata_max_threshold: int = 10,
    download_only: bool = False,
):
    """Downloads scenes from the provided data dictionary or list, processes them into composites, and saves them to the specified output directory.

    Parameters
    ----------
    data : dict | list
        Dictionary or list containing scene data.
    bands : list[str]
        Bands to be used for processing, e.g. ["red", "green", "blue"] for true color.
    output_dir : str
        Directory where processed scenes will be saved.
    aws_session : rasterio.session.AWSSession | None, optional
        AWS session for interacting with AWS, where required, by default None
    keep_original_band_scenes : bool, optional
        Keep original band scenes in the output directory, by default False
    reference_month : str | list, optional
        Month to use as a reference for finding closest and farthest pairs, by default "01"
    edge_detection : bool, optional
        Using edge detection in the processing, by default False
    edge_detection_mode : Literal[&quot;sobel&quot;, &quot;laplacian&quot;, &quot;canny&quot;], optional
        Edge detection mode, by default "canny"
    gamma : float, optional
        Gamma correction value for the images, by default 1.0
    equalise_histogram : bool, optional
        Equalise histogram of the images, by default False
    stretch_contrast : bool, optional
        Intensity enhancement of the images, by default False
    gray_scale : bool, optional
        Use gray scale images, by default False
    averaging : bool, optional
        Use averaging for generating gray scale images instead of NTSC formula, by default False
    subdir : str, optional
        Subdirectory name for processed images, by default "true_color"
    force_reprocess : bool, optional
        Force reprocessing of existing files, by default False
    reference_band_number : int | None, optional
        Reference band number for the reprojecting and resampling the images to the reference image, by default None
    filename_suffix : str, optional
        Suffix to be added to the processed files, by default "PROC"
    fill_nodata : bool, optional
        Whether to fill nodata values in the images, by default False
    fill_nodata_max_threshold : int, optional
        Maximum searching threshold for pixels with nodata values, by default 10
    download_only : bool, optional
        If True, only downloads the scenes without processing them, by default False
    """
    os.makedirs(output_dir, exist_ok=True)

    process_dir = f"{output_dir}/{subdir}"
    process_ds_dir = f"{output_dir}/{subdir}_ds"

    if force_reprocess:
        print("Force reprocessing is enabled, all existing files will be reprocessed.")
        shutil.rmtree(process_dir, ignore_errors=True)
        shutil.rmtree(process_ds_dir, ignore_errors=True)

    os.makedirs(process_dir, exist_ok=True)
    os.makedirs(process_ds_dir, exist_ok=True)

    bands_suffixes = _get_band_suffixes(data, bands[0:3])

    if type(data) == list and type(reference_month) == str:
        reference_month = [reference_month] * len(data)

    if type(data) == list:
        closest_pair = get_pair_dict_alternate(
            data[0],
            data[1],
            "closest",
            reference_month_1=reference_month[0],
            reference_month_2=reference_month[1],
        )
        farthest_pair = get_pair_dict_alternate(
            data[0],
            data[1],
            "farthest",
            reference_month_1=reference_month[0],
            reference_month_2=reference_month[1],
        )
    else:
        closest_pair = get_pair_dict(data, "closest", reference_month=reference_month)
        farthest_pair = get_pair_dict(data, "farthest", reference_month=reference_month)

    pr_date_list = closest_pair + [farthest_pair[1]]

    pr_date_list_processed = download_and_process_series(
        pr_date_list,
        bands,
        bands_suffixes,
        output_dir,
        process_dir,
        process_ds_dir,
        aws_session,
        keep_original_band_scenes,
        edge_detection,
        edge_detection_mode,
        gamma,
        equalise_histogram,
        stretch_contrast,
        gray_scale,
        averaging,
        reference_band_number,
        filename_suffix,
        download_only=download_only,
        fill_nodata=fill_nodata,
        fill_nodata_max_threshold=fill_nodata_max_threshold,
    )

    cols = ["Reference", "Closest_target", "Farthest_target"]
    df = pd.DataFrame(
        {
            cols[i]: [
                el["local_path"],
                el["local_path_ds"],
            ]
            for i, el in enumerate(pr_date_list_processed)
        },
        columns=cols,
    )
    df.to_csv(
        f"{output_dir}/pairs.csv",
        index=False,
    )


def combine_comparison_results(
    root_output: str,
    coreg_default_params: list[bool] | None,
    dir_suffix: str | None = None,
) -> pd.DataFrame:
    """Creates a Dataframe with the results of the co-registration methods.

    Parameters
    ----------
    root_output : str
        Output directory where the results are stored.
    coreg_default_params : list[bool] | None
        Defult parameters for the Co-Register method, by default None
    dir_suffix : str | None, optional
        Suffix for the directory names, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe with the results of the co-registration methods.
    """

    dir_suffix = f"_{dir_suffix}" if dir_suffix else ""

    methods = []
    df_list = []
    col_names = ["Title", "SSIM Raw", "MSE Raw"]
    try:
        try:
            coreg_df = pd.read_csv(f"{root_output}/Co_Register{dir_suffix}/output.csv")
        except:
            coreg_df = pd.read_csv(
                f"{root_output}/Co_Register{dir_suffix}_lpc/output.csv"
            )
        coreg_df["Method"] = ["Co-Register"] * len(coreg_df)
        df_list.append(coreg_df)
        methods.append("Co-Register")
        col_names.extend(
            [
                "Co-Register SSIM Aligned",
                "Co-Register MSE Aligned",
                "Co-Register Run Time",
                "Co-Register Shifts",
                "Co-Register Defaults",
            ]
        )
    except:
        print("Co-Register results not found, skipping.")

    if coreg_default_params is None:
        coreg_default_params = [True] * len(coreg_df)

    try:
        karios_df = pd.read_csv(f"{root_output}/Karios{dir_suffix}/output.csv")
        karios_df["Method"] = ["Karios"] * len(karios_df)
        df_list.append(karios_df)
        methods.append("Karios")
        col_names.extend(
            [
                "Karios SSIM Aligned",
                "Karios MSE Aligned",
                "Karios Run Time",
                "Karios Shifts",
                "Karios Defaults",
            ]
        )
    except:
        print("Karios results not found, skipping.")

    try:
        arosics_df = pd.read_csv(f"{root_output}/AROSICS{dir_suffix}/output.csv")
        arosics_df["Method"] = ["AROSICS"] * len(arosics_df)
        df_list.append(arosics_df)
        methods.append("AROSICS")
        col_names.extend(
            [
                "AROSICS SSIM Aligned",
                "AROSICS MSE Aligned",
                "AROSICS Run Time",
                "AROSICS Shifts",
                "AROSICS Defaults",
            ]
        )
    except:
        print("AROSICS results not found, skipping.")

    try:
        arosics_edge_df = pd.read_csv(
            f"{root_output}/AROSICS{dir_suffix}_edge/output.csv"
        )
        arosics_edge_df["Method"] = ["AROSICS Edge"] * len(arosics_edge_df)
        df_list.append(arosics_edge_df)
        methods.append("AROSICS Edge")
        col_names.extend(
            [
                "AROSICS Edge SSIM Aligned",
                "AROSICS Edge MSE Aligned",
                "AROSICS Edge Run Time",
                "AROSICS Edge Shifts",
                "AROSICS Edge Defaults",
            ]
        )
    except:
        print("AROSICS Edge results not found, skipping.")

    if len(df_list) == 0:
        print("No results found, cannot combine.")
        return pd.DataFrame()

    # Combine all dataframes
    output_dfs = (
        pd.concat(
            df_list,
        )
        .reset_index(drop=True)
        .drop("Unnamed: 0", axis=1)
    )
    output_dfs

    target_0 = []
    target_1 = []

    target_0.extend(["target_0"])
    target_1.extend(["target_1"])

    for df in df_list:
        try:
            target_0.extend(
                df[df["Title"] == "target_0"][["SSIM Raw", "MSE Raw"]]
                .values[0]
                .tolist()
            )
            break
        except:
            continue

    for df in df_list:
        try:
            target_1.extend(
                df[df["Title"] == "target_1"][["SSIM Raw", "MSE Raw"]]
                .values[0]
                .tolist()
            )
            break
        except:
            continue

    for i, method in enumerate(methods):
        method_slice = output_dfs[output_dfs["Method"] == method].drop("Method", axis=1)

        target_0_vals = method_slice[method_slice["Title"] == "target_0"]
        if target_0_vals.empty:
            target_0.extend(["Failed"] * 4)
        else:
            target_0.extend(target_0_vals.values[0][3:].tolist())

        if i == 0:
            target_0.extend([coreg_default_params[0]])
        elif i == 1 or i == 2:
            target_0.extend([True])
        else:
            target_0.extend([False])

        target_1_vals = method_slice[method_slice["Title"] == "target_1"]
        if target_1_vals.empty:
            target_1.extend(["Failed"] * 4)
        else:
            target_1.extend(target_1_vals.values[0][3:].tolist())

        if i == 0:
            target_1.extend([coreg_default_params[1]])
        elif i == 1 or i == 2:
            target_1.extend([True])
        else:
            target_1.extend([False])

    out_df = pd.DataFrame([target_0, target_1], columns=col_names)
    out_df.to_csv(
        f"{root_output}/co_registration_results.csv",
        index=False,
    )
    return out_df


def reproject_box_to_geojson(
    bbox: list,
    src_crs: str | int = "EPSG:4326",
    dst_crs: str | int = "EPSG:3031",
    always_xy: bool = True,
):
    """
    Reprojects a bounding box from the source CRS to the destination CRS and returns it as a GeoJSON Polygon.

    Parameters
    ----------
    bbox : list
        Bounding box in the form [minx, miny, maxx, maxy].
    src_crs : str | int, optional
        Source coordinate reference system. Default is "EPSG:4326".
    dst_crs : str | int, optional
        Destination coordinate reference system. Default is "EPSG:3031".
    always_xy : bool, optional
        If True, the transformer will always expect and return coordinates in (x, y) order.
        Default is True.
    Returns
    -------
    dict
        GeoJSON representation of the reprojected bounding box.
    """

    src_crs = str(src_crs)
    dst_crs = str(dst_crs)

    if "EPSG:" not in src_crs:
        src_crs = f"EPSG:{src_crs}"
    if "EPSG:" not in dst_crs:
        dst_crs = f"EPSG:{dst_crs}"

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=always_xy)
    xx, yy = box(*bbox).exterior.coords.xy
    bbs = transformer.transform(xx, yy)
    bbsp = gloads(to_geojson(Polygon(list(zip(bbs[0], bbs[1])))))
    return bbsp


def reproject_bounds(
    bounds: list[list | rasterio.coords.BoundingBox],
    source_crs: str | int | list[str] | list[int],
    dest_crs: str | int = "EPSG:4326",
    always_xy: bool = True,
    rounding_precision: int | None = None,
) -> list[list]:
    """
    Reprojects list of bounding boxes to the given `crs`.

    Parameters
    ----------
    bounds : list[list | rasterio.coords.BoundingBox]
        List of bounding boxes to reproject. Each bounding box can be a list in the form
        [minx, miny, maxx, maxy] or a rasterio.coords.BoundingBox object.
    source_crs : str | int | list[str] | list[int]
        Source coordinate reference system(s). Can be a single CRS (str or int) or a list of CRSs.
    dest_crs : str | int, optional
        Destination coordinate reference system. Default is "EPSG:4326".
    always_xy : bool, optional
        If True, the transformer will always expect and return coordinates in (x, y) order.
        Default is True.
    rounding_precision : int | None, optional
        If provided, the coordinates will be rounded to this many decimal places. Default is None.
    Returns
    -------
    list[list]
        List of reprojected bounding boxes in the form [minx, miny, maxx, maxy].
    Notes
    -----
    This function uses the `pyproj` library for coordinate transformations and `rasterio`
    for handling bounding boxes.
    """

    if isinstance(source_crs, str) or isinstance(source_crs, int):
        source_crs = [source_crs] * len(bounds)

    source_crs = [str(crs) for crs in source_crs]
    dest_crs = str(dest_crs)

    for i, crs in enumerate(source_crs):
        if "EPSG:" not in crs:
            source_crs[i] = f"EPSG:{crs}"
    if "EPSG:" not in dest_crs:
        dest_crs = f"EPSG:{dest_crs}"

    new_bounds = []
    for c, b in zip(source_crs, bounds):
        transformer = Transformer.from_crs(c, dest_crs, always_xy=always_xy)
        if isinstance(b, rasterio.coords.BoundingBox):
            xmin = b.left
            xmax = b.right
            ymax = b.top
            ymin = b.bottom
        else:
            xmin, ymin, xmax, ymax = b
        tl = transformer.transform(xmin, ymax)
        tr = transformer.transform(xmax, ymax)
        br = transformer.transform(xmax, ymin)
        bl = transformer.transform(xmin, ymin)

        new_x = [tl[0], tr[0], br[0], bl[0]]
        new_y = [tl[1], tr[1], br[1], bl[1]]

        if rounding_precision is not None:
            new_x = [round(coord, rounding_precision) for coord in new_x]
            new_y = [round(coord, rounding_precision) for coord in new_y]

        new_bounds.append([min(new_x), min(new_y), max(new_x), max(new_y)])

    return new_bounds


def write_bounds_to_kml(
    bounds: list[BoundingBox] | BoundingBox, filename, poly_names: list | None = None
):
    if type(bounds) is not list:
        bounds = [bounds]
    with open(filename, "w") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            "<Document>\n"
        )
        for i, b in enumerate(bounds):
            if poly_names is not None:
                name = poly_names[i]
            else:
                name = f"Bound {i + 1}"
            f.write(
                f"<Placemark><name>{name}</name><Polygon><outerBoundaryIs><LinearRing><coordinates>{b[0]},{b[1]} {b[2]},{b[1]} {b[2]},{b[3]} {b[0]},{b[3]} {b[0]},{b[1]}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>\n"
            )
        f.write("</Document>\n</kml>")


def create_dataset_from_files(
    paths: list[str],
    times: list[datetime] | list[str] | None = None,
    crs: int | None = None,
    bands: list[str] | None = None,
    remove_val: float | int | None = 0,
    band_scale: float | None = 1.0,
    band_offset: float | None = 0.0,
    chunks: dict = {},
    scale_factor: float | None = None,
    bbox: list | None = None,
    bbox_crs: int | None = None,
    use_geometry: bool = False,
    optimize_dataset: bool = False,
) -> xr.Dataset:
    """Create an xarray dataset from the list of files and times.

    Parameters
    ----------
    paths : list[str]
        List of file or dir paths to the raster files.
    times : list[datetime] | list[str] | None, optional
        List of times corresponding to the files. If None, uses indices as strings.
    crs : int | None, optional
        Coordinate reference system to assign to the dataset, by default None.
    bands : list[str] | None, optional
        List of band names to rename the dataset variables. If None, uses default band names.
    remove_val : float | int | None, optional
        Value to remove from the dataset. If None, no values are removed. Default is 0.
        If a float or int, all values equal to this will be set to NaN.
    band_scale : float | None, optional
        Scale factor to apply to the dataset values. If None, no scaling is applied. Default is 1.0.
    band_offset : float | None, optional
        Offset to apply to the dataset values. If None, no offset is applied. Default is 0.0.
    chunks : dict, optional
        Dictionary specifying the chunk sizes for the dataset. If None, no chunking is applied.
    scale_factor : float | None, optional
        Factor by which to scale the dataset. If None, no scaling is applied.
    bbox : list | None, optional
        Bounding box to which to clip the dataset in the form [minx, miny, maxx, maxy]. If None, no clipping is applied.
    bbox_crs : int | None, optional
        CRS of the bounding box, required if bbox is provided. If None, it is assumed to be the same as the dataset CRS.
    use_geometry : bool, optional
        Whether to use the geometry for clipping. Default is False.
    optimize_dataset : bool, optional
        Whether to optimize the dataset by removing variables and attributes that are not needed. Default is False.

    Returns
    -------
    xr.Dataset
        The created xarray dataset.
    """

    if bbox is not None and crs is None:
        raise ValueError("CRS must be provided if bbox is provided.")

    if bbox is not None:
        if use_geometry:
            bbox = [
                reproject_box_to_geojson(
                    bbox,
                    src_crs=bbox_crs if bbox_crs is not None else crs,
                    dst_crs=crs,
                )
            ]
        else:
            bbox = reproject_bounds(
                [bbox],
                source_crs=bbox_crs if bbox_crs is not None else crs,
                dest_crs=crs,
            )[0]

    if times is None:
        times = [str(i) for i in range(len(paths))]

    if os.path.isfile(paths[0]):
        if bands is None:
            bands = [f"band_{i+1}" for i in range(rasterio.open(paths[0]).count)]
        dsl = [
            rxr.open_rasterio(f, band_as_variable=True, chunks=chunks)[
                [f"band_{i+1}" for i in range(len(bands))]
            ]
            .astype("float32")
            .assign_coords(time=t)
            .expand_dims("time", axis=2)
            for (f, t) in zip(paths, times)
        ]
    else:
        dsl = []
        for j, dir in enumerate(paths):
            files = glob.glob(f"{dir}/**")
            if bands is None:
                bands = [f"band_{i+1}" for i in range(len(files))]
            band_array = [
                rxr.open_rasterio(f, chunks=chunks)
                .astype("float32")
                .assign_coords(time=times[j])
                .expand_dims("time", axis=2)
                for f in files
            ]
            temp_ds = xr.Dataset(
                {
                    f"band_{i+1}": (("time", "y", "x"), br.data[:, :, 0, :])
                    for i, br in enumerate(band_array)
                },
                coords={
                    "time": [times[j]],
                    "y": np.unique(
                        np.sort(
                            xr.concat(
                                [b.coords["y"] for b in band_array], dim="time"
                            ).data.ravel()
                        )
                    ),
                    "x": np.unique(
                        np.sort(
                            xr.concat(
                                [b.coords["x"] for b in band_array], dim="time"
                            ).data.ravel()
                        )
                    ),
                },
            )
            dsl.append(temp_ds)
    print(len(dsl), "datasets found in the target directory.")
    ds = xr.concat(dsl, dim="time").transpose("time", "y", "x").drop_attrs()
    if crs is not None:
        ds["spatial_ref"] = np.int32(crs)
        ds = ds.rio.write_crs(f"epsg:{crs}")

    ds = ds.rename_vars({f"band_{i+1}": b for i, b in enumerate(bands)})
    ds = ds[["y", "x", "spatial_ref", "time"] + bands]

    if scale_factor is not None:
        ds = resample_xarray_dataset(ds, scale_factor).chunk(chunks=chunks)

    if bbox is not None:
        try:
            if use_geometry:
                print("Clipping dataset to the provided geometry.")
                ds = ds.rio.clip(bbox, all_touched=True)
            else:
                print("Clipping dataset to the provided bounding box.")
                ds = ds.rio.clip_box(*bbox)
        except Exception as e:
            print(f"Error clipping dataset: {e}")

    ds[bands] = ds[bands] * band_scale + band_offset
    if remove_val is not None:
        print(f"Removing values equal to {remove_val} from the dataset.")
        ds = ds.where(ds > remove_val)

    if optimize_dataset:
        ds = optimize(ds)[0]
    return ds
