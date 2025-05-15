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
import numpy as np
import imageio
import cv2 as cv
from pyproj import Proj
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
from shapely import ops, from_wkt
from shapely import Polygon
from pathlib import Path
from typing import Literal
from datetime import datetime
from datetime import timedelta


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
    """
    if not adjust_to_centre:
        return transform
    new_orig_x = transform.c + ((1 - scale_factor_x) * abs(transform.a) / 2)
    new_orig_y = transform.f - ((1 - scale_factor_y) * abs(transform.e) / 2)
    return rasterio.Affine(
        transform.a, transform.b, new_orig_x, transform.d, transform.e, new_orig_y
    )


def downsample_dataset(
    dataset_path: str,
    scale_factor: Union[float, list[float]] = 1.0,
    output_file: str = "",
    enhance_function=None,
    force_shape: tuple = (),  # (height, width)
    readjust_origin: bool = False,
):
    """
    Downsamples the output data and returns the new downsampled data and its new affine transformation according to `scale_factor`
    The output shape could also be forced using `forced_shape` parameter.
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
        )

        if enhance_function is not None:
            data = enhance_function(data)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
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


def reproject_tif(src_path, dst_path, dst_crs):

    with rasterio.open(src_path) as src:
        print(f"reprojecting from {src.crs} to {dst_crs}")
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

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
                    resampling=Resampling.nearest,
                )


flip_img = lambda img: np.flipud(np.rot90(img.T))


def adjust_resolutions(
    dataset_paths: list[str],
    output_paths: list[str],
    resampling_resolution: str = "lower",
) -> tuple:
    """
    Adjusts the resolutions for two or more datasets with different ones. Rounding errors might cause a slightly different output resolutions.
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
                f"WARNING: Ground resolutions are different for the provided images. Setting it to the {resampling_resolution} resolution."
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

    assert x_condition and y_condition, "The provided scenes do not overlap"

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
):
    """
    Creates a mosaic of overlapping scenes. Offsets will be added to the size of the final mosaic if specified.
    NOTE: dataset ground resolutions should be the same. Use `resolution_adjustment` flag to fix the unequal resolutions.
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
    crss = []
    transforms = []
    boundss = []
    for p in dataset_paths:
        raster = rasterio.open(p)
        transform = raster.transform
        ps_x.append(abs(transform.a))
        ps_y.append(abs(transform.e))
        rasters.append(raster)
        transforms.append(transform)
        crs = raster.crs.data
        crss.append(crs)
        boundss.append(utm_bounds(raster.bounds, crs))

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
    for i, bounds in enumerate(boundss):
        lefts.append(bounds.left)
        rights.append(bounds.right)
        tops.append(bounds.top)
        bottoms.append(bounds.bottom)

    min_left = min(lefts)
    min_bottom = min(bottoms)
    max_right = max(rights)
    max_top = max(tops)

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

    mosaic = np.zeros((*new_shape, 3)).astype("uint8")
    warps = []
    for i, rs in enumerate(rasters):
        img = flip_img(rs.read())
        imgw = cv.warpAffine(
            img, new_transforms[i], (new_shape[1], new_shape[0]), flags=cv.INTER_LINEAR
        )

        if len(imgw.shape) == 2:
            idx = np.where(imgw != 0)
            for i in range(0, 3):
                mosaic[idx[0], idx[1], i] = imgw[idx[0], idx[1]]
        else:
            idx = np.where(cv.cvtColor(imgw, cv.COLOR_BGR2GRAY) != 0)
            mosaic[idx[0], idx[1], :] = imgw[idx[0], idx[1], :]
        if return_warps:
            warp = np.zeros_like(imgw).astype("uint8")
            if len(imgw.shape) == 2:
                warp[idx[0], idx[1]] = imgw[idx[0], idx[1]]
            else:
                warp[idx[0], idx[1], :] = imgw[idx[0], idx[1], :]
            warps.append(warp)

    if resolution_adjustment:
        shutil.rmtree("temp/res_adjustment", ignore_errors=True)

    if mosaic_output_path != "":
        print("Writing mosaic file.")
        mosaic_profile = rasterio.open(dataset_paths[0]).profile
        mosaic_profile["height"] = new_shape[0]
        mosaic_profile["width"] = new_shape[1]
        mosaic_profile["count"] = 3
        mosaic_profile["transform"] = rasterio.Affine(
            selected_res_x, 0.0, min_left, 0.0, -selected_res_y, max_top
        )
        with rasterio.open(mosaic_output_path, "w", **mosaic_profile) as ds:
            for i in range(0, 3):
                ds.write(mosaic[:, :, i], i + 1)

    return mosaic, warps, new_transforms


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
):
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


def shift_targets_to_origin(
    tgt_imgs: list[np.ndarray],
    ref_transform: rasterio.Affine,
    tgt_transforms: list[rasterio.Affine],
) -> list[np.ndarray]:
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


def resize_bbox(bbox, scale_factor=1.0):
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


def utm_bounds(bounds: BoundingBox, crs: dict, skip_stereo: bool = True) -> BoundingBox:
    has_negative = any(b < 0 for b in bounds)
    if crs == {}:
        print("No CRS data found. Returning original.")
        return bounds
    if (crs["proj"] == "stere") and (skip_stereo):
        return bounds
    if (crs["proj"] == "stere") or (
        (crs["proj"] == "utm") and ("south" not in crs) and (has_negative)
    ):
        lla_bl = UTMtoLLA(UTM(bounds.left, bounds.bottom), crs)
        lla_tr = UTMtoLLA(UTM(bounds.right, bounds.top), crs)
        utm_bl = LLAtoUTM(
            lla_bl, {"south": True} | crs if crs["proj"] == "utm" else crs
        )
        utm_tr = LLAtoUTM(
            lla_tr, {"south": True} | crs if crs["proj"] == "utm" else crs
        )
        return BoundingBox(utm_bl.x, utm_bl.y, utm_tr.x, utm_tr.y)
    elif crs["proj"] == "utm":
        return bounds


def find_scene_bounding_box_lla(scene: str, scale_factor=1.0):
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
):
    """
    Transforms the dataset accroding to given translation, rotation and scale params and writes it to the `output_path` file.
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
                for i in range(0, profile["count"]):
                    ds.write(warped_img[:, :, i], i + 1)

    return warped_img


def detect_edges(img):
    """
    simple canny edge detector
    """
    out = cv.Canny(img, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    out = cv.dilate(out, kernel, iterations=2)
    out = cv.erode(out, kernel, iterations=3)
    kernel = np.ones((15, 15), np.uint8)
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel, iterations=10)
    out = cv.Canny(out, 0, 255)
    return out


def find_cells(img, points, window_size, invert_points=False):
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
) -> tuple:

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
    for ref, tgt in zip(final_ref_cells, final_tgt_cells):
        corrs.append(cv.phaseCorrelate(np.float32(tgt), np.float32(ref), None)[1])

    valid_idx = np.where(np.array(corrs) > signal_power_thresh)

    if len(valid_idx[0]) == 0:
        print(
            "WARNING: No points were found with the given correlation threshold, turning off phase correlation filter..."
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
    number_of_iterations=30,
    termination_eps=0.03,
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
    generate_gif: bool = True,
    generate_csv: bool = True,
    fps: int = 3,
    of_dist_thresh: Union[None, int, float] = 2,  # pixels
    phase_corr_filter: bool = False,
    phase_corr_signal_thresh: float = 0.9,
    use_overlap: bool = False,
    rethrow_error: bool = False,
    resampling_resolution: str = "lower",
    return_shifted_images: bool = False,
    laplacian_kernel_size: Union[None, int] = None,
    lower_of_dist_thresh: Union[None, int, float] = None,
    band_number: Union[None, int] = None,
) -> tuple:

    ORIGIN_DIST_THRESHOLD = 1

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
        if type(tgt) == str:
            tgt_raster = rasterio.open(tgt)
            tgt_profiles.append(tgt_raster.profile)

            ref_orig_x = abs(
                ref_raster.profile["transform"].c / ref_raster.profile["transform"].a
            )
            ref_orig_y = abs(
                ref_raster.profile["transform"].f / ref_raster.profile["transform"].e
            )
            ref_width = ref_raster.profile["width"]
            ref_height = ref_raster.profile["height"]

            tgt_orig_xs = [
                abs(p["transform"].c / p["transform"].a) for p in tgt_profiles
            ]
            tgt_orig_ys = [
                abs(p["transform"].f / p["transform"].e) for p in tgt_profiles
            ]
            tgt_widths = [p["width"] for p in tgt_profiles]
            tgt_heights = [p["height"] for p in tgt_profiles]

            orig_x_diff = np.abs(np.diff(np.array([ref_orig_x] + tgt_orig_xs)))
            orig_y_diff = np.abs(np.diff(np.array([ref_orig_y] + tgt_orig_ys)))
            w_diff = np.abs(np.diff(np.array([ref_width] + tgt_widths)))
            h_diff = np.abs(np.diff(np.array([ref_height] + tgt_heights)))

            if (
                (not np.all(orig_x_diff < ORIGIN_DIST_THRESHOLD))
                or (not np.all(orig_y_diff < ORIGIN_DIST_THRESHOLD))
                or (not np.all(w_diff == 0))
                or (not np.all(h_diff == 0))
            ) and (not use_overlap):
                print(
                    "WARNING: Origins or shapes of the reference or target images do not match. Consider using the `use_overlap` flag."
                )

    if use_overlap:
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
            ref_img = cv.Laplacian(ref_img, cv.CV_8U, ksize=laplacian_kernel_size)

    tgt_imgs_temp = []
    if laplacian_kernel_size is not None:
        for tgt_img in tgt_imgs:
            tgt_imgs_temp.append(
                cv.Laplacian(tgt_img, cv.CV_8U, ksize=laplacian_kernel_size)
            )
        grey_tgts = tgt_imgs.copy()
        tgt_imgs = tgt_imgs_temp.copy()
        tgt_imgs_temp = None

    if generate_gif:
        export_outputs = True
    if export_outputs or generate_gif:
        if (type(reference) != str) or (any(type(el) != str for el in targets)):
            print(
                "To generate output GeoTiffs or GIF animation all inputs should be string paths to input scenes. Setting related flags to False."
            )
            export_outputs = False
            generate_gif = False
            generate_csv = False
    if export_outputs:
        os.makedirs(output_path, exist_ok=True)

    tgt_aligned_list = []
    processed_tgt_images = []
    processed_output_images = []
    shifts = []
    process_ids = []

    temp_dir = "temp/outputs"
    os.makedirs(temp_dir, exist_ok=True)
    if return_shifted_images:
        aligned_output_dir = os.path.join(output_path, "Aligned")
        os.makedirs(aligned_output_dir, exist_ok=True)
    for i, tgt_img in enumerate(tgt_imgs):
        if use_overlap:
            ref_img = ref_imgs[i]
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
                    f"""For target {i} ({os.path.basename(targets[i])}), couldn't find enough good features for target or reference. Num features: {tgt_good.shape[0]}"""
                )
                continue
            _, inliers = cv.estimateAffine2D(tgt_good, ref_good)

            ref_good_temp = ref_good.copy()[0, :, :]
            tgt_good_temp = tgt_good.copy()[0, :, :]
            ref_good_temp = ref_good_temp[inliers.ravel().astype(bool)]
            tgt_good_temp = tgt_good_temp[inliers.ravel().astype(bool)]

            if phase_corr_filter:
                ref_good_temp, tgt_good_temp = find_corrs_shifts(
                    ref_img,
                    tgt_img,
                    ref_good_temp,
                    tgt_good_temp,
                    of_params["lk_params"]["winSize"],
                    phase_corr_signal_thresh,
                )

            shift_x, shift_y = np.mean(ref_good_temp - tgt_good_temp, axis=0)
            num_features = ref_good_temp.shape[0]

            print(
                f"For target {i} ({os.path.basename(targets[i])}), Num features: {num_features}"
            )

            print(
                f"For target {i} ({os.path.basename(targets[i])}), shifts => x: {shift_x / scale_factors[i][1]}, y: {shift_y / scale_factors[i][0]} pixels."
            )
            shifts.append(
                (shift_x / scale_factors[i][1], shift_y / scale_factors[i][0])
            )

            to_warp = tgt_img if laplacian_kernel_size is None else grey_tgts[i]
            if shift_x == np.inf:
                print(
                    f"No valid shifts found for target {i} ({os.path.basename(targets[i])})"
                )
                continue

            tgt_aligned = warp_affine_dataset(
                to_warp,
                translation_x=shift_x,
                translation_y=shift_y,
            )
            tgt_aligned_list.append(tgt_aligned)
            process_ids.append(i)

            if export_outputs:
                profile = tgt_profiles[i]
                updated_profile = profile.copy()
                temp_path = os.path.join(temp_dir, f"out_{i}.tiff")
                out_path = os.path.join(
                    aligned_output_dir, os.path.basename(targets[i])
                )
                updated_profile["transform"] = rasterio.Affine(
                    profile["transform"].a,
                    profile["transform"].b,
                    profile["transform"].c
                    + (shift_x / scale_factors[i][1]) * profile["transform"].a,
                    profile["transform"].d,
                    profile["transform"].e,
                    profile["transform"].f
                    + (shift_y / scale_factors[i][0]) * profile["transform"].e,
                )
                with rasterio.open(
                    out_path if return_shifted_images else temp_path,
                    "w",
                    **updated_profile,
                ) as ds:
                    for j in range(0, updated_profile["count"]):
                        ds.write(tgt_origs[i][:, :, j], j + 1)
                processed_tgt_images.append(targets[i])
                processed_output_images.append(
                    out_path if return_shifted_images else temp_path
                )
        except Exception as e:
            print(
                f"Algorithm did not converge for target {i} ({os.path.basename(targets[i])}) {'for the reason below:' if rethrow_error else ''}"
            )
            if rethrow_error:
                raise
            else:
                print(e)

    if export_outputs:
        if laplacian_kernel_size is not None:
            if use_overlap:
                ref_imgs = grey_refs
            else:
                ref_img = grey_ref
            tgt_imgs = grey_tgts

        if use_overlap:
            ref_imgs = [ref_imgs[id] for id in process_ids]
            tgt_imgs = [tgt_imgs[id] for id in process_ids]

        out_gif = os.path.join(
            output_path,
            f"output.gif",
        )
        target_titles = [f"target_{id}" for id in process_ids]

        if os.path.isfile(out_gif):
            os.remove(out_gif)
        datasets_paths = [reference] + processed_output_images
        ssims_aligned = [
            np.round(
                ssim(
                    ref_imgs[k] if use_overlap else ref_img,
                    tgt_aligned_list[k],
                    win_size=3,
                ),
                3,
            )
            for k in range(len(tgt_aligned_list))
        ]
        mse_aligned = [
            np.round(
                mse(ref_imgs[k] if use_overlap else ref_img, tgt_aligned_list[k]), 3
            )
            for k in range(len(tgt_aligned_list))
        ]
        datasets_titles = ["Reference"] + [
            f"{target_title}, ssim:{ssim_score}, mse:{mse_score}"
            for target_title, ssim_score, mse_score in zip(
                target_titles, ssims_aligned, mse_aligned
            )
        ]

        if generate_gif:
            make_difference_gif(
                datasets_paths,
                out_gif,
                datasets_titles,
                fps=fps,
                mosaic_scenes=True,
            )

        out_gif = os.path.join(
            output_path,
            f"raw_output.gif",
        )
        if os.path.isfile(out_gif):
            os.remove(out_gif)
        datasets_paths = [reference] + processed_tgt_images
        ssims_raw = [
            np.round(
                ssim(
                    ref_imgs[k] if use_overlap else ref_img,
                    tgt_imgs[k] if laplacian_kernel_size is None else grey_tgts[k],
                    win_size=3,
                ),
                3,
            )
            for k in range(len(tgt_aligned_list))
        ]
        mse_raw = [
            np.round(
                mse(
                    ref_imgs[k] if use_overlap else ref_img,
                    tgt_imgs[k] if laplacian_kernel_size is None else grey_tgts[k],
                ),
                3,
            )
            for k in range(len(tgt_aligned_list))
        ]
        datasets_titles = ["Reference"] + [
            f"{target_title}, ssim:{ssim_score}, mse:{mse_score}"
            for target_title, ssim_score, mse_score in zip(
                target_titles, ssims_raw, mse_raw
            )
        ]

        if generate_gif:
            make_difference_gif(
                datasets_paths,
                out_gif,
                datasets_titles,
                fps=fps,
                mosaic_scenes=True,
            )

        if generate_csv:
            out_ssim = os.path.join(
                output_path,
                f"output.csv",
            )
            out_ssim_df = pd.DataFrame(
                zip(target_titles, ssims_raw, ssims_aligned, mse_raw, mse_aligned),
                columns=["Title", "SSIM Raw", "SSIM Aligned", "MSE Raw", "MSE Aligned"],
                index=None,
            )
            out_ssim_df.to_csv(out_ssim, encoding="utf-8")

    shutil.rmtree(temp_dir, ignore_errors=True)

    return tgt_aligned_list, shifts


def apply_gamma(data, gamma=0.5, stretch_hist: bool = False, adjust_hist: bool = False):
    data = np.power(data, gamma)
    if adjust_hist:
        data = equalize_hist(data)
    if stretch_hist:
        p2, p98 = np.percentile(data, (2, 98))
        data = rescale_intensity(data, in_range=(p2, p98), out_range="uint8")
    else:
        data *= 255 / data.max()
        data = data.astype("uint8")
    return data


def query_stac_server(query: dict, server_url: str, pystac: bool = False) -> list:
    """
    Queries the stac-server (STAC) backend.
    This function handles pagination.
    query is a python dictionary to pass as json to the request.

    server_url example: https://landsatlook.usgs.gov/stac-server/search
    """

    if pystac:
        client = Client.open(server_url)
        search = client.search(**query)
        features = list(search.item_collection())
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
                itertools.chain(features, query_stac_server(query, server_url, pystac))
            )

    return features


def find_scenes_dict(
    features: list,
    one_per_month: bool = True,
    start_end_years: list[int] = [],
    acceptance_list: list[str] = [],
    remove_duplicate_times: bool = True,
) -> dict | tuple:
    scene_list = []
    scene_dict = dict()
    for feature in features:
        is_item = type(feature) != dict

        if is_item:
            feature = feature.to_dict()

        id = feature["id"]

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
                    url_alternate = None

                scene_dict[id][s] = url
                scene_dict[id][f"{s}_alternate"] = url_alternate

    if "landsat:scene_id" in feature["properties"]:
        path_rows = [k.split("_")[2] for k in scene_dict]
        time_ind = 3
    else:
        if type(features[0]) == dict:
            path_rows = ["_".join(k.split("_")[3:6]) for k in scene_dict]
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
                times_idx = sorted(
                    np.unique(
                        [re.findall(r"\d{8}", d["scene_name"])[0] for d in temp_list],
                        return_index=True,
                    )[1].tolist()
                )
                temp_list = [temp_list[idx] for idx in times_idx]

            scene_list.extend(temp_list)
            temp_dict_time[t] = temp_list
        scene_dict_pr_time[pr] = temp_dict_time

    return scene_dict_pr_time, scene_list


def get_search_query(
    bbox: Union[list, BoundingBox],
    collections: list[str] | None = ["landsat-c2l2-sr", "landsat-c2l2-st"],
    collection_category: list[str] | None = ["T1", "T2", "RT"],
    platform: str | None = "LANDSAT_8",
    start_date: str = "2014-10-30T00:00:00",
    end_date: str = "2015-01-23T23:59:59",
    cloud_cover: int | None = 80,
    is_landsat: bool = True,
) -> dict:
    if type(bbox) != list:
        bbox = [bbox.left, bbox.bottom, bbox.right, bbox.top]
    query = {
        "bbox": bbox,
        "page": 1,
        "limit": 100,
    }
    if not is_landsat:
        platform = None
        collection_category = None
        cloud_cover = None
    if platform is not None:
        query["query"] = {"platform": {"in": [platform]}}
    if collections is not None:
        query["collections"] = collections
    if collection_category is not None:
        query["query"] = {"landsat:collection_category": {"in": collection_category}}
    if (start_date != "") and (end_date != ""):
        query["datetime"] = f"{start_date}.000Z/{end_date}.999Z"
    if cloud_cover is not None:
        query["query"] = {"eo:cloud_cover": {"lte": cloud_cover}}

    return query


def make_true_color_scene(
    dataset_paths: list[str],
    output_path: str | None = None,
    enhance: bool = False,
) -> np.ndarray:
    red = dataset_paths[0]
    green = dataset_paths[1]
    blue = dataset_paths[2]
    profile = rasterio.open(red).profile
    profile["count"] = 3
    profile["dtype"] = "uint8"

    reds = rasterio.open(red).read()
    redf = flip_img(reds)

    greens = rasterio.open(green).read()
    greenf = flip_img(greens)

    blues = rasterio.open(blue).read()
    bluef = flip_img(blues)

    img = cv.merge([redf, greenf, bluef])
    if enhance:
        img = apply_gamma(img, 1.0, True)

    if output_path is not None:
        with rasterio.open(output_path, "w", **profile) as ds:
            ds.write(img[:, :, 0], 1)
            ds.write(img[:, :, 1], 2)
            ds.write(img[:, :, 2], 3)

    return img


def tracking_image(
    ref_points: np.ndarray,
    tgt_points: np.ndarray,
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    line_width: int = 100,
    dot_radius: int = 100,
) -> tuple:

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
    """
    `read_mode` could be `poly` or `point`
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


def stream_scene_from_aws(
    geotiff_file,
    aws_session: rasterio.session.AWSSession | None = None,
    metadata_only: bool = False,
):
    def get_data():
        with rasterio.open(geotiff_file) as geo_fp:
            profile = geo_fp.profile
            bounds = geo_fp.bounds
            crs = geo_fp.crs
            if not metadata_only:
                scene = geo_fp.read()
        return scene, profile, bounds, crs

    scene = np.zeros(0)
    if aws_session is None:
        with rasterio.Env(aws_session):
            scene, profile, bounds, crs = get_data()
    else:
        scene, profile, bounds, crs = get_data()

    return scene, {"profile": profile, "bounds": bounds, "crs": crs}


def hillshade(
    array: np.ndarray,
    azimuth: float = 30.0,
    angle_altitude: float = 30.0,
    skip_negative: bool = True,
) -> np.ndarray:

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
    data: dict,
    time_distance: str = Literal["closest", "farthest"],
    reference_month: str = "01",
) -> list:
    """Finds the closest or farthest member of the given scene dictionary in time in it to a given reference scence.

    Parameters
    ----------
    data : dict
        Scenes dictionary
    reference_id : str
        Id of the reference scenes in the scene dictionary
    time_distance : str, optional
        Distance option, by default Literal["closest", "farthest"]
    prefered_month: str, optional
        Prefered month in the scenes dict to retrieve data for the reference scene, by default 01

    Returns
    -------
    list
        list of reference data and its closest/farthest target
    """

    data = data.copy()

    if time_distance not in ["closest", "farthest"]:
        raise ValueError("time distance options are only closest or farthest")

    scene_dates = list(data.keys())
    try:
        reference_date_idx = [reference_month in date for date in scene_dates].index(
            True
        )
        reference_date = scene_dates[reference_date_idx]
    except:
        raise Exception(
            "Could not find data for the provided month for the reference scene."
        )
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
