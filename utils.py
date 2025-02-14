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
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.exposure import equalize_hist, rescale_intensity
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, numpy_image_to_torch
from torch import Tensor, cuda
import itertools
from pykml import parser
from collections import namedtuple
import warnings
from bs4 import BeautifulSoup
import html5lib


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
        # transform = readjust_origin_for_new_pixel_size(transform, *scale_factor)

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
):
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

    return [[abs(t.a), abs(t.e), t] for t in transforms]


def find_overlap(
    dataset_1: str,
    dataset_2: str,
    return_images: bool = False,
    return_pixels: bool = False,
    resampling_resolution: str = "lower",
):
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

            (raster_1_px_size, raster_1_py_size, _), (
                raster_2_px_size,
                raster_2_py_size,
                _,
            ) = outputs

        min_left = min(
            bounds_1.left // raster_1_px_size, bounds_2.left // raster_2_px_size
        )
        max_top = max(
            bounds_1.top // raster_1_py_size, bounds_2.top // raster_2_py_size
        )

        bounds_1 = rasterio.coords.BoundingBox(
            int(bounds_1.left // raster_1_px_size - min_left),
            int(max_top - bounds_1.bottom // raster_1_py_size),
            int(bounds_1.right // raster_1_px_size - min_left),
            int(max_top - bounds_1.top // raster_1_py_size),
        )

        bounds_2 = rasterio.coords.BoundingBox(
            int(bounds_2.left // raster_2_px_size - min_left),
            int(max_top - bounds_2.bottom // raster_2_py_size),
            int(bounds_2.right // raster_2_px_size - min_left),
            int(max_top - bounds_2.top // raster_2_py_size),
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
            :,
        ]
        raster_overlap_1 = warps[0][
            overlap_in_mosaic.top : overlap_in_mosaic.bottom,
            overlap_in_mosaic.left : overlap_in_mosaic.right,
            :,
        ]
        raster_overlap_2 = warps[1][
            overlap_in_mosaic.top : overlap_in_mosaic.bottom,
            overlap_in_mosaic.left : overlap_in_mosaic.right,
            :,
        ]

    shutil.rmtree("temp", ignore_errors=True)

    return overlap_in_mosaic, (
        mosaic,
        mosaic_overlap,
        raster_overlap_1,
        raster_overlap_2,
    )


def make_mosaic(
    dataset_paths: list[str],
    offset_x: int = 0,
    offset_y: int = 0,
    return_warps: bool = False,
    resolution_adjustment: bool = False,
    resampling_resolution: str = "lower",
):
    """
    Creates a mosaic of overlapping scenes. Offsets will be added to the size of the final mosaic if specified.
    NOTE: dataset ground resolutions should be the same. Use `resolution_adjustment` flag to fix the unequal resolutions.
    """

    if resolution_adjustment:
        os.makedirs("temp", exist_ok=True)
        new_dataset_paths = [
            os.path.join("temp", f"scaled_raster_{i}.tif")
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
    for p in dataset_paths:
        raster = rasterio.open(p)
        transform = raster.profile["transform"]
        ps_x.append(abs(round(transform.a)))
        ps_y.append(abs(round(transform.e)))
        rasters.append(raster)
        transforms.append(transform)

    ps_x_condition = all(ps == ps_x[0] for ps in ps_x)
    ps_y_conditoin = all(ps == ps_y[0] for ps in ps_y)
    assert (
        ps_x_condition and ps_y_conditoin
    ), "Ground resolutions are different for datasets. Please use `resolution_adjustment` flag first to fix the issue."

    lefts = []
    rights = []
    tops = []
    bottoms = []
    for r in rasters:
        bounds = r.bounds
        lefts.append(abs(bounds.left // transform.a))
        rights.append(abs(bounds.right // transform.a))
        tops.append(abs(bounds.top // transform.e))
        bottoms.append(abs(bounds.bottom // transform.e))

    min_left = min(lefts)
    min_bottom = min(bottoms)
    max_right = max(rights)
    max_top = max(tops)

    new_shape = (
        int(max_top - min_bottom) + offset_y,
        int(max_right - min_left) + offset_x,
    )

    new_transforms = []
    for t in transforms:
        new_transforms.append(
            np.array(
                [
                    [1.0, abs(t.b // t.e), t.c // t.a - min_left],
                    [t.d // t.a, 1.0, max_top - abs(t.f // t.e)],
                ]
            )
        )

    mosaic = np.zeros((*new_shape, 3)).astype("uint8")
    warps = []
    for i, rs in enumerate(rasters):
        img = flip_img(rs.read())
        imgw = cv.warpAffine(img, new_transforms[i], (new_shape[1], new_shape[0]))

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
        shutil.rmtree("temp", ignore_errors=True)

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
    mosaic_offsets_x: list[int] = [],
    mosaic_offsets_y: list[int] = [],
    fps: int = 1,
    use_overlap: bool = False,
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
        assert (
            len(titles_list) == len(images_list)
            if not mosaic_scenes
            else len(images_list) - 1
        ), "Length of provided list of titles does not match the number of images."
    else:
        titles_list = [os.path.splitext(os.path.basename(f))[0] for f in images_list]

    images = []
    font = cv.FONT_HERSHEY_SIMPLEX
    if mosaic_scenes:
        ref_scene = temp_paths[0]
        tgt_scenes = temp_paths[1:]
        if len(mosaic_offsets_x) == 0:
            mosaic_offsets_x = [0] * len(tgt_scenes)
        if len(mosaic_offsets_y) == 0:
            mosaic_offsets_y = [0] * len(tgt_scenes)
        for i, p in enumerate(tgt_scenes):
            img, _, _ = make_mosaic(
                [ref_scene, p], mosaic_offsets_x[i], mosaic_offsets_y[i]
            )
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

        if use_overlap:
            temp_images = [temp_images[0]] + shift_targets_to_origin(
                temp_images[1:], transforms[0], transforms[1:]
            )
        for img in temp_images:
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


def UTMtoLLA(utm: UTM, crs: dict):
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
    return LLA(lla[1], lla[0])


def LLAtoUTM(lla: LLA, crs: dict):
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
    utm = proj(lla.lon, lla.lat)
    return UTM(utm[0], utm[1])


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
    ref_img,
    tgt_img,
    ref_points,
    tgt_points,
    inliers=[],
    corr_win_size=(15, 15),
    corr_thresh=0.75,
    drop_unbound=True,
    invert_points=True,
):
    ref_points = ref_points[0, :, :].astype("int")
    tgt_points = tgt_points[0, :, :].astype("int")
    if len(inliers) != 0:
        ref_points = ref_points[inliers.ravel().astype(bool)]
        tgt_points = tgt_points[inliers.ravel().astype(bool)]
        if (len(ref_points) == 0) or (len(tgt_points) == 0):
            print("Could not find enough points in the images.")
            return np.inf, np.inf

    ref_cells = find_cells(
        ref_img,
        ref_points,
        corr_win_size,
        invert_points,
    )
    tgt_cells = find_cells(
        tgt_img,
        tgt_points,
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
        corrs.append(cv.phaseCorrelate(np.float32(tgt), np.float32(ref), None))
    corrs = [c[0] for c in corrs if c[1] > corr_thresh]

    if len(corrs) == 0:
        print(
            "WARNING: No points were found with the given correlation threshold, returning zero shifts..."
        )
        return (0.0, 0.0)

    shift_x = np.mean([c[0] for c in corrs])
    shift_y = np.mean([c[1] for c in corrs])
    return shift_x, shift_y


def filter_features(
    ref_points: np.ndarray,
    tgt_points: np.ndarray,
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    bounding_shape: tuple,
    dists: np.ndarray,
    dist_thresh: Union[None, int, float] = None,
    lower_of_dist_thresh: Union[None, int, float] = None,
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

        tgt_good = np.expand_dims(tgt_good[valid_idx], axis=0)
        ref_good = np.expand_dims(ref_good[valid_idx], axis=0)
    else:
        tgt_good = np.expand_dims(tgt_points, axis=0)
        ref_good = np.expand_dims(ref_points, axis=0)

    if (ref_good.shape[1] < 4) or (tgt_good.shape[1] < 4):
        print(
            f"WARNING: couldn't find enough good features for target or reference. num ref features: {ref_good.shape[0]}, num tgt features = {tgt_good.shape[0]}"
        )
    return ref_good, tgt_good


def co_register(
    reference: Union[str, np.ndarray],
    targets=Union[
        str, np.ndarray, list[str], list[np.ndarray], list[Union[str, np.ndarray]]
    ],
    filtering_mode: str = "of",  # "lg", "of", "pca" or "pca_of"
    number_of_iterations=10,
    termination_eps=1e-5,
    of_params: dict = dict(
        # params for ShiTomasi corner detection
        feature_params=dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
        ),
        # Parameters for lucas kanade optical flow
        lk_params=dict(
            winSize=(50, 50),
            maxLevel=2,
        ),
    ),
    output_path: str = "",
    export_outputs: bool = True,
    generate_gif: bool = True,
    generate_csv: bool = True,
    fps: int = 3,
    of_dist_thresh: Union[None, int, float] = 5,  # pixels
    pca_dist_thresh: int = 25,  # pixels
    filter_by_ground_res: bool = False,
    ground_resolution: float = 10.0,  # meters
    ground_resolution_limit: float = 10.0,  # meters
    grey_output: bool = True,
    corr_win_size: tuple = (15, 15),
    corr_thresh: float = 0.75,
    enhanced_shift_method: str = "",  # empty, "mean" or "corr"
    remove_outlilers: bool = True,
    use_overlap: bool = False,
    rethrow_error: bool = False,
    resampling_resolution: str = "lower",
    ligh_glue_max_points: int = 1000,
    return_shifted_images: bool = False,
    laplacian_kernel_size: Union[None, int] = None,
    lower_of_dist_thresh: Union[None, int, float] = None,
    origin_dist_threshold: int = 1,  # pixels,
) -> tuple:

    pca = PCA(2)
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
                (not np.all(orig_x_diff < origin_dist_threshold))
                or (not np.all(orig_y_diff < origin_dist_threshold))
                or (not np.all(w_diff == 0))
                or (not np.all(h_diff == 0))
            ) and (not use_overlap):
                warnings.warn(
                    "Origins or shapes of the reference or target images do not match. Consider using the `use_overlap` flag."
                )

    if use_overlap:
        tgt_imgs = []
        tgt_origs = []
        tgt_raws = []
        ref_imgs = []
        for i, tgt in enumerate(targets):
            tgt_raws.append(flip_img(rasterio.open(tgt).read().copy().astype("uint8")))
            _, (_, _, ref_overlap, tgt_overlap) = find_overlap(
                reference, tgt, True, resampling_resolution=resampling_resolution
            )
            ref_imgs.append(cv.cvtColor(ref_overlap, cv.COLOR_BGR2GRAY).astype("uint8"))
            tgt_imgs.append(cv.cvtColor(tgt_overlap, cv.COLOR_BGR2GRAY).astype("uint8"))
            grey_output = False
    else:
        if type(reference) == str:
            ref_img = flip_img(ref_raster.read().copy())
            ref_img = (
                cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
                if ref_img.shape[2] != 1
                else ref_img[:, :, 0]
            )
        else:
            if len(reference.shape) == 2:
                ref_img = reference
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
                img = (
                    cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    if img.shape[2] != 1
                    else img[:, :, 0]
                )
                tgt_imgs.append(img.astype("uint8"))
            else:
                if len(tgt.shape) == 2:
                    tgt_imgs.append(tgt.astype("uint8"))
                else:
                    tgt_imgs.append(cv.cvtColor(tgt, cv.COLOR_BGR2GRAY).astype("uint8"))
                tgt_origs.append(tgt.astype("uint8"))
        tgt_raws = tgt_origs.copy()

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

    temp_dir = "temp/outputs"
    os.makedirs(temp_dir, exist_ok=True)
    if return_shifted_images:
        aligned_output_dir = os.path.join(output_path, "Aligned")
        os.makedirs(aligned_output_dir, exist_ok=True)
    for i, tgt_img in enumerate(tgt_imgs):
        if use_overlap:
            ref_img = ref_imgs[i]
        try:
            if filtering_mode == "pca":
                pca_diff = pca.fit_transform(ref_img - tgt_img)
                pca_idx = np.where(abs(pca_diff[:, 1]) < pca_dist_thresh)

                ref_good = ref_img[pca_idx]
                tgt_good = tgt_img[pca_idx]

                ((shift_x, shift_y), _) = cv.phaseCorrelate(
                    np.float32(tgt_good), np.float32(ref_good), None
                )
                enhanced_shift_method = ""
            else:
                if filtering_mode == "pca_of":
                    pca_diff = pca.fit_transform(ref_img - tgt_img)
                    pca_idx = np.where(abs(pca_diff[:, 1]) < pca_dist_thresh)

                    ref_img_pca = ref_img[pca_idx]
                    tgt_img_pca = tgt_img[pca_idx]
                else:
                    ref_img_pca = ref_img.copy()
                    tgt_img_pca = tgt_img.copy()

                if (filtering_mode == "of") or (filtering_mode == "pca_of"):
                    p0 = cv.goodFeaturesToTrack(
                        ref_img_pca, mask=None, **of_params["feature_params"]
                    )
                    p1, st, _ = cv.calcOpticalFlowPyrLK(
                        ref_img_pca,
                        tgt_img_pca,
                        p0,
                        None,
                        **of_params["lk_params"],
                        criteria=criteria,
                    )

                    dist = np.linalg.norm(p1[st == 1] - p0[st == 1], axis=1)
                else:
                    p0, p1 = extract_light_glue_features(
                        ref_img, tgt_img, ligh_glue_max_points
                    )
                    dist = np.linalg.norm(p1 - p0, axis=1)

                ref_good, tgt_good = filter_features(
                    p0,
                    p1,
                    ref_img,
                    tgt_img,
                    tgt_img_pca.shape,
                    dist,
                    of_dist_thresh,
                    lower_of_dist_thresh,
                )
                h, inliers = cv.estimateAffine2D(tgt_good, ref_good)

                if enhanced_shift_method == "":
                    shift_x = h[0, 2]
                    shift_y = h[1, 2]
                else:
                    if enhanced_shift_method == "corr":
                        if not remove_outlilers:
                            inliers = []
                        shift_x, shift_y = find_corrs_shifts(
                            ref_img_pca,
                            tgt_img_pca,
                            ref_good,
                            tgt_good,
                            inliers,
                            corr_win_size,
                            corr_thresh,
                        )
                    else:
                        ref_good_temp = ref_good[0, :, :]
                        tgt_good_temp = tgt_good[0, :, :]
                        if remove_outlilers:
                            ref_good_temp = ref_good_temp[inliers.ravel().astype(bool)]
                            tgt_good_temp = tgt_good_temp[inliers.ravel().astype(bool)]
                        shift_x, shift_y = np.mean(
                            ref_good_temp - tgt_good_temp, axis=0
                        )

            shifts.append((shift_x, shift_y))

            if shift_x == np.inf:
                print(f"No valid shifts found.")
                tgt_aligned_list.append(np.zeros(0))
                continue

            if (filter_by_ground_res) and (
                (abs(shift_x * ground_resolution) > ground_resolution_limit)
                or (abs(shift_y * ground_resolution) > ground_resolution_limit)
            ):
                print(f"Shifts too high for target {i}. Ignoring the scene.")
                tgt_aligned_list.append(np.zeros(0))
                continue

            tgt_aligned = warp_affine_dataset(
                tgt_img if laplacian_kernel_size is None else grey_tgts[i],
                translation_x=shift_x,
                translation_y=shift_y,
            )
            tgt_aligned_list.append(tgt_aligned)

            if export_outputs:
                profile = tgt_profiles[i]
                temp_path = os.path.join(temp_dir, f"out_{i}.tiff")
                if return_shifted_images:
                    updated_profile = profile.copy()
                    updated_profile["transform"] = rasterio.Affine(
                        profile["transform"].a,
                        profile["transform"].b,
                        profile["transform"].c + shift_x * profile["transform"].a,
                        profile["transform"].d,
                        profile["transform"].e,
                        profile["transform"].f + shift_y * profile["transform"].e,
                    )
                    with rasterio.open(
                        os.path.join(aligned_output_dir, os.path.basename(targets[i])),
                        "w",
                        **updated_profile,
                    ) as ds:
                        for j in range(0, updated_profile["count"]):
                            ds.write(tgt_raws[i][:, :, j], j + 1)
                processed_output_images.append(temp_path)
                processed_tgt_images.append(targets[i])
                if grey_output:
                    warped = tgt_aligned
                else:
                    warped = warp_affine_dataset(
                        tgt_raws[i] if use_overlap else tgt_origs[i],
                        translation_x=shift_x,
                        translation_y=shift_y,
                    )
                with rasterio.open(temp_path, "w", **profile) as ds:
                    for j in range(0, profile["count"]):
                        if grey_output:
                            ds.write(warped, j + 1)
                        else:
                            ds.write(warped[:, :, j], j + 1)
        except Exception as e:
            print(f"Algorithm did not converge for target {i} for the reason below:")
            if rethrow_error:
                raise
            else:
                print(e)
                tgt_aligned_list.append(np.zeros(0))

    if generate_gif:
        if laplacian_kernel_size is not None:
            ref_img = grey_ref
            ref_imgs = grey_refs
        out_gif = os.path.join(
            output_path,
            f'{filtering_mode}{"" if enhanced_shift_method == "" else "_" + enhanced_shift_method}.gif',
        )
        target_titles = [f"target_{id}" for id in range(len(tgt_aligned_list))]

        if os.path.isfile(out_gif):
            os.remove(out_gif)
        datasets_paths = [reference] + processed_output_images
        ssims_aligned = [
            np.round(
                ssim(
                    ref_imgs[id] if use_overlap else ref_img,
                    tgt_aligned_list[id],
                    win_size=3,
                ),
                3,
            )
            for id in range(len(tgt_aligned_list))
        ]
        mse_aligned = [
            np.round(
                mse(ref_imgs[id] if use_overlap else ref_img, tgt_aligned_list[id]), 3
            )
            for id in range(len(tgt_aligned_list))
        ]
        datasets_titles = ["Reference"] + [
            f"{target_title}, ssim:{ssim_score}, mse:{mse_score}"
            for target_title, ssim_score, mse_score in zip(
                target_titles, ssims_aligned, mse_aligned
            )
        ]

        make_difference_gif(
            datasets_paths,
            out_gif,
            datasets_titles,
            fps=fps,
            use_overlap=use_overlap,
        )

        out_gif = os.path.join(
            output_path,
            f'raw_{filtering_mode}{"" if enhanced_shift_method == "" else "_" + enhanced_shift_method}.gif',
        )
        if os.path.isfile(out_gif):
            os.remove(out_gif)
        datasets_paths = [reference] + processed_tgt_images
        ssims_raw = [
            np.round(
                ssim(
                    ref_imgs[id] if use_overlap else ref_img,
                    tgt_imgs[id] if laplacian_kernel_size is None else grey_tgts[id],
                    win_size=3,
                ),
                3,
            )
            for id in range(len(tgt_aligned_list))
        ]
        mse_raw = [
            np.round(
                mse(
                    ref_imgs[id] if use_overlap else ref_img,
                    tgt_imgs[id] if laplacian_kernel_size is None else grey_tgts[id],
                ),
                3,
            )
            for id in range(len(tgt_aligned_list))
        ]
        datasets_titles = ["Reference"] + [
            f"{target_title}, ssim:{ssim_score}, mse:{mse_score}"
            for target_title, ssim_score, mse_score in zip(
                target_titles, ssims_raw, mse_raw
            )
        ]
        make_difference_gif(
            datasets_paths,
            out_gif,
            datasets_titles,
            fps=fps,
            use_overlap=use_overlap,
        )

        if generate_csv:
            out_ssim = os.path.join(
                output_path,
                f'{filtering_mode}{"" if enhanced_shift_method == "" else "_" + enhanced_shift_method}.csv',
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


def fetch_landsat_stac_server(query):
    """
    Queries the stac-server (STAC) backend.
    This function handles pagination.
    query is a python dictionary to pass as json to the request.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip",
        "Accept": "application/geo+json",
    }

    url = f"https://landsatlook.usgs.gov/stac-server/search"
    data = requests.post(url, headers=headers, json=query).json()
    error = data.get("message", "")
    if error:
        raise Exception(f"STAC-Server failed and returned: {error}")

    context = data.get("context", {})
    if not context.get("matched"):
        return []
    print(context)

    features = data["features"]
    if data["links"]:
        query["page"] += 1
        query["limit"] = context["limit"]

        features = list(itertools.chain(features, fetch_landsat_stac_server(query)))

    return features


def find_landsat_scenes_dict(features: dict) -> dict:
    feat_dict = dict()
    for feature in features:
        id = feature["id"]

        scene_id = feature["properties"]["landsat:scene_id"]

        assets = feature["assets"]

        red = ""
        red_alternate = ""
        if "red" in assets:
            red = assets["red"]["href"]
            red_alternate = assets["red"]["alternate"]["s3"]["href"]

        green = ""
        green_alternate = ""
        if "green" in assets:
            green = assets["green"]["href"]
            green_alternate = assets["green"]["alternate"]["s3"]["href"]

        blue = ""
        blue_alternate = ""
        if "blue" in assets:
            blue = assets["blue"]["href"]
            blue_alternate = assets["blue"]["alternate"]["s3"]["href"]

        feat_dict[id] = dict(
            scene_id=scene_id,
            red=(red, red_alternate),
            green=(green, green_alternate),
            blue=(blue, blue_alternate),
        )
    return feat_dict


def get_landsat_search_query(
    bbox: Union[list, BoundingBox],
    collections: list[str] = ["landsat-c2l2-sr", "landsat-c2l2-st"],
    platform: str = "LANDSAT_8",
    start_date: str = "2014-10-30T00:00:00",
    end_date: str = "2015-01-23T23:59:59",
    cloud_cover: int = 80,
) -> dict:
    if type(bbox) != list:
        bbox = [bbox.left, bbox.bottom, bbox.right, bbox.top]
    query = {
        "bbox": bbox,
        "collections": collections,
        "query": {
            "eo:cloud_cover": {"lte": cloud_cover},
            "platform": {"in": [platform]},
            "landsat:collection_category": {"in": ["T1", "T2", "RT"]},
        },
        "datetime": f"{start_date}.000Z/{end_date}.999Z",
        "page": 1,
        "limit": 100,
    }
    return query


def make_landsat_true_color_scene(
    dataset_paths: list[str], output_path: str
) -> np.ndarray:
    red = dataset_paths[0]
    green = dataset_paths[1]
    blue = dataset_paths[2]
    profile = rasterio.open(red).profile
    profile["count"] = 3
    profile["dtype"] = "uint8"

    reds = apply_gamma(rasterio.open(red).read(), 1.0, True)[0, :, :]
    redf = flip_img(reds)

    greens = apply_gamma(rasterio.open(green).read(), 1.0, True)[0, :, :]
    greenf = flip_img(greens)

    blues = apply_gamma(rasterio.open(blue).read(), 1.0, True)[0, :, :]
    bluef = flip_img(blues)

    img = cv.merge([redf, greenf, bluef])

    if output_path != "":
        with rasterio.open(output_path, "w", **profile) as ds:
            ds.write(reds, 1)
            ds.write(greens, 2)
            ds.write(blues, 3)

    return img


def load_array_image(
    image: np.ndarray,
    grayscale: bool = True,
    use_cuda: bool = False,
) -> Tensor:
    if not grayscale:
        image = image[..., ::-1]
    image = numpy_image_to_torch(image)
    if use_cuda:
        image.cuda()
    return image


def extract_light_glue_features(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    max_num_points: int = 1000,
    grayscale: bool = True,
) -> tuple:

    use_cuda = False
    if cuda.is_available():
        use_cuda = True

    extractor = SuperPoint(
        max_num_keypoints=max_num_points
    ).eval()  # load the extractor
    matcher = LightGlue(features="superpoint").eval()  # load the matcher

    if use_cuda:
        extractor = extractor.cuda()
        matcher = matcher.cuda()

    image0 = load_array_image(ref_img, grayscale, use_cuda)
    image1 = load_array_image(tgt_img, grayscale, use_cuda)

    feats0 = extractor.extract(
        image0
    )  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension
    matches = matches01["matches"]  # indices with shape (K,2)
    points0 = feats0["keypoints"][
        matches[..., 0]
    ]  # coordinates in image #0, shape (K,2)
    points1 = feats1["keypoints"][
        matches[..., 1]
    ]  # coordinates in image #1, shape (K,2)
    features = points0.numpy().astype("int")
    moved_features = points1.numpy().astype("int")
    return features, moved_features


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


PointXYZ = namedtuple("PointXYZ", ["x", "y", "z"])
PointLLA = namedtuple("PointLLA", ["lat", "lon", "alt"])


def read_kml_polygon(
    kml_path: str, is_lat_lon: bool = True, is_kml_str: bool = False
) -> tuple:

    if is_kml_str:
        os.makedirs("temp_kml_dir", exist_ok=True)
        with open("temp_kml_dir/temp_kml.kml", "w") as f:
            f.write(kml_path)
        kml_path = "temp_kml_dir/temp_kml.kml"
    with open(kml_path) as f:
        doc = parser.parse(f).getroot().Document.Placemark
    shutil.rmtree("temp_kml_dir", ignore_errors=True)

    coords = []
    for pm in doc:
        coord = pm.Polygon.outerBoundaryIs.LinearRing.coordinates.text.strip()
        coord_list = coord.split(" ")
        for c in coord_list:
            c_splits = c.split(",")
            c_nums = [float(i) for i in c_splits]
            if len(c_nums) == 2:
                c_nums.append(0.0)
            if is_lat_lon:
                coords.append(PointLLA(c_nums[0], c_nums[1], c_nums[2]))
            else:
                coords.append(PointXYZ(c_nums[0], c_nums[1], c_nums[2]))

    if is_lat_lon:
        lats = [p.lat for p in coords]
        lons = [p.lon for p in coords]
        bbox = BoundingBox(min(lats), min(lons), max(lats), max(lons))
    else:
        xs = [p.x for p in coords]
        ys = [p.y for p in coords]
        bbox = BoundingBox(min(xs), min(ys), max(xs), max(ys))

    return coords, bbox


def stream_scene_from_aws(geotiff_file, aws_session, metadata_only: bool = False):
    scene = np.zeros(0)
    with rasterio.Env(aws_session):
        with rasterio.open(geotiff_file) as geo_fp:
            profile = geo_fp.profile
            bounds = geo_fp.bounds
            crs = geo_fp.crs
            if not metadata_only:
                scene = geo_fp.read()
    return scene, {"profile": profile, "bounds": bounds, "crs": crs}

