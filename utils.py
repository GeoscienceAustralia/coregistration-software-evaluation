import glob
import os
import asyncio
import json
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


def get_s1_filenames(
    polygon_list:list[str], 
    years:list[str], 
    output_file_path:str,
    products:list[str]=["GRD", "SLC"],
    is_poly_bbox:bool = True,
):
    """
    Get scene names for S1 via direct requet to SARA server
    """
    s1_names = []
    if os.path.isfile(output_file_path):
        with open(output_file_path, "r") as f:
            for l in f:
                s1_names.append(l.strip())
    else:
        start = f"{years[0]}-01-01"
        end = f"{years[1]}-12-12"
        for poly in polygon_list:
            for prod in products:
                page = 1
                query_resp = ["start"]
                while query_resp != []:
                    query = f"https://copernicus.nci.org.au/sara.server/1.0/api/collections/S1/search.json?_pretty=1&{"box" if is_poly_bbox else "geometry"}={poly}&startDate={start}&completionDate={end}&instrument=C-SAR&sensor=IW&maxRecords=500&productType={prod}&page={page}"
                    response = json.loads(requests.get(query).content)
                    query_resp = [r["properties"]["title"] for r in response["features"]]
                    s1_names.extend(query_resp)
                    page += 1
        with open(output_file_path, "w") as f:
            for n in s1_names:
                f.write(f"{n}\n")
    return s1_names

def save_file_list(file_list:dict, save_path:str) -> None:
    """
    Saves the retrieved data.
    """
    with open(save_path, "w") as f:
        for (k, v) in file_list.items():
            for filename in v:
                f.write(f"{k},{filename}\n")
    return None

async def find_all_files_for_case(query_case:tuple, sat_data_dir:str) -> bool:
    """
    Finds all files for a selected case of product/year/month
    """
    case_path = os.path.join(sat_data_dir, query_case[0], query_case[1], f"{query_case[1]}-{query_case[2]}")
    print(f"Retrieving files for {case_path}", end="\r")
    return glob.glob(case_path + "/*/*.zip")

async def find_aoi_files(aoi:str, all_files:list[str]) -> list[str]:
    """
    Filters all files and finds files for the area of interest.
    """
    print(f"filtering files for {aoi}", end="\r")
    return list(filter(lambda p: aoi in p, all_files))

def flatten(l:list[list]) -> list:
    """
    Flattens the list
    """
    return[x for xs in l for x in xs]

# Not sure if async runs well in notebook
async def find_files_for_aios_async(
        query_cases:list[tuple], 
        sat_data_dir:str, 
        aoi_list:list[str], 
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
        query_cases:list[tuple], 
        sat_data_dir:str, 
        aoi_list:list[str], 
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

def find_files_for_s1_aois(nci_files_dict:dict, s1_file_names:list[str]) -> dict:
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


def find_polarisation_files_s1(dir:str) -> list[str]:
    """
    Finds Sentinel 1 data from the provided path.
    """
    return glob.glob(os.path.join(dir, "measurement", "*"))


def load_s1_scenes(
        zip_file_path:str, 
        zip_file_id:str, 
        subdir_name:str = "", 
        remove_input:bool = True,
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
        scale_factor = 0.03, 
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

def enhance_color_s1(data, is_slc:bool = True):
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


def plot_scenes_s1(data, data_names, data_transforms, is_slc:bool = True):
    """
    Plots the data given the names of the scenes and their affine transformations
    """
    _, axes = plt.subplots(1, len(data), figsize=(10 * len(data), 10 * len(data)))
    if type(axes) != np.ndarray:
        axes = [axes]
    for i, d in enumerate(data):
        ax = axes[i]
        show(enhance_color_s1(d, is_slc), ax=ax, title=f"{data_names[i]}", transform=data_transforms[i])
        ax.set_title(f"{data_names[i]}")
        ax.title.set_size(10)

def get_scenes_dict(data_df:pd.DataFrame, product:list[str] = [], is_s1:bool=True) -> dict:
    scenes_dict = {}
    id_list = data_df.ID.unique()
    for id in id_list:
        filtered_df = data_df[data_df.ID == id].reset_index(drop=True)
        if product != []:
            for p in product:
                filtered_df = filtered_df[filtered_df.Path.apply(lambda x: p in x)].reset_index(drop=True)

        grouper = filtered_df.Path.apply(lambda r: os.path.split(r)[1].split("_")[5 if is_s1 else 2][0:6])
        secene_list = [list(filtered_df.groupby(grouper))[i][1].Path.iloc[0] for i in range(0, len(grouper.unique()))]
        scenes_dict[id] = secene_list
    return scenes_dict

def downsample_dataset(
        dataset_path:str, 
        scale_factor:float, 
        output_file:str = "", 
        enhance_function = None,
    ):
    """
    Downsamples the output data and returns the new downsampled data and its new affine transformation according to `scale_factor`
    """
    with rasterio.open(dataset_path) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor),
                int(dataset.width * scale_factor)
            ),
            resampling=Resampling.bilinear
        )

        if enhance_function is not None:
            data = enhance_function(data)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        profile = dataset.profile
        profile.update(
            transform = transform,
            width = data.shape[2],
            height = data.shape[1],
            dtype = data.dtype,
        )

    if output_file != "":
        with rasterio.open(output_file, "w", **profile) as ds:
            for i in range(0, profile["count"]):
                ds.write(data[i], i + 1)

    return data, transform

def enhance_color_matching(data, uint16:bool=False):
    """
    Increases the brightness of the output data
    """
    if uint16:
        data = data / 256
    data = data.astype("float64")
    data *= 255 / data.max()
    return data.astype("uint8")

def plot_matching(datasets, alpha:float = 0.65, plot_title:str = "", save_fig_path:str = ""):
    _, axes = plt.subplots(3,3, figsize=(10, 10))
    show(datasets[0][0], ax=axes[0,0], title="Reference scene")
    show(datasets[0][0], ax=axes[1,0], title="Reference scene")
    show(datasets[0][0], ax=axes[2,0], title="Reference scene")

    show(datasets[1][0], ax=axes[0,1], title="Target scene")
    show(datasets[2][0], ax=axes[1,1], title="Global matching")
    show(datasets[3][0], ax=axes[2,1], title="Local matching")

    show(((datasets[0][0] * alpha) + (datasets[1][0] * (1 - alpha))).astype("uint8"), ax=axes[0,2], title="Reference + Target ")
    show(((datasets[0][0] * alpha) + (datasets[2][0] * (1 - alpha))).astype("uint8"), ax=axes[1,2], title="Reference + Global matching")
    show(((datasets[0][0] * alpha) + (datasets[3][0] * (1 - alpha))).astype("uint8"), ax=axes[2,2], title="Reference + Local matching")
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
        print(f'reprojecting from {src.crs} to {dst_crs}')
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        print(f'saving - {dst_path}')
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
                
flip_img = lambda img: np.flipud(np.rot90(img.T))

def get_mosaic_params(datasets_paths:list[str], offset:int = 0):
    lefts = []
    rights = []
    tops = []
    bottoms = []
    transforms = []
    for p in datasets_paths:
        raster = rasterio.open(p)
        bounds = raster.bounds
        transform = raster.profile["transform"]
        lefts.append(bounds.left // transform.a)
        rights.append(bounds.right // transform.a)
        tops.append(-1 * bounds.top // transform.e)
        bottoms.append(-1 * bounds.bottom // transform.e)
        transforms.append(transform)

    min_left = min(lefts)
    min_bottom = min(bottoms)
    max_right = max(rights)
    max_top = max(tops)
    min_top = min(tops)

    new_shape = (int(max_top - min_bottom) + offset, int(max_right - min_left) + offset)

    new_transforms = []
    for t in transforms:
        new_transforms.append(
            np.array(
                [
                    [1.0, -1 * t.b // t.e, t.c // t.a - min_left],
                    [t.d // t.a, -1.0, -1 * t.f // t.e - min_top],
                ]
            )
        )
    return new_shape, new_transforms

def make_difference_gif(
        images_list:list[str], 
        output_path:str, 
        titles_list:list[str] = [], 
        scale_factor = -1,
        mosaic_scenes:bool = False,
        mosaic_offset = 0,
    ):
    os.makedirs("temp", exist_ok=True)
    temp_paths = [os.path.join("temp", os.path.basename(f)) for f in images_list]

    if scale_factor != -1:
        for i, p in enumerate(temp_paths):
            downsample_dataset(images_list[i], 0.1, p)
    else:
        temp_paths = images_list

    if mosaic_scenes:
        new_shape, new_transforms = get_mosaic_params(temp_paths, mosaic_offset)

    if len(titles_list) > 0:
        assert len(titles_list) == len(images_list), "Length of provided list of titles does not match the number of images."
    else:
        titles_list = [os.path.splitext(os.path.basename(f))[0] for f in images_list]

    images = []
    font = cv.FONT_HERSHEY_SIMPLEX
    # org
    org = (5, 50)
    # fontScale
    fontScale = 1.5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2
    thickness = 3

    for i, p in enumerate(temp_paths):
        img = rasterio.open(p).read()
        img = flip_img(img).copy()
        if mosaic_scenes:
            img = cv.warpAffine(img, new_transforms[i], (new_shape[1], new_shape[0]))
        cv.putText(img, titles_list[i], org, font, fontScale, color, thickness, cv.LINE_AA)
        images.append(img)
        imageio.mimwrite(output_path, images, loop = 0, fps = 1)
    shutil.rmtree("temp", ignore_errors=True)


def find_band_files_s2(dir:str, selected_res_index:int = -1) -> list[str]:
    """
    Retrieving band files from the data path `dir`, If there are multiple resolutions of the data, `selected_res_index` must be specified as a positive integer.
    """
    if selected_res_index == -1:
        band_files = [file for file in glob.glob(os.path.join(dir, "GRANULE", "*", "IMG_DATA", "*"))]
    else:
        band_dirs = [file for file in glob.glob(os.path.join(dir, "GRANULE", "*", "IMG_DATA", "*"))]
        selected_res = band_dirs[selected_res_index]
        band_files = glob.glob(f"{selected_res}/*")
    return band_files


def load_s2_bands(
        zip_file_path:str, 
        zip_file_id:str, 
        s2_other_bands_list:list[str], 
        subdir_name:str = "", 
        remove_input:bool = True,
        selected_res_index:int = -1,
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
        zip_id:str, 
        true_bands, 
        band_profile, 
        scale_factor = 0.03, 
        subdir_name = "",
    ) -> tuple:
    """
    writes a true color image using true color bands (not TCI). The updated profile for true bands shold be provided.
    also downsamples the output data and returns the new downsampled data and its new affine transformation according to `scale_factor`
    """
    output_dir = f"./data/outputs/{zip_id}/{subdir_name}/"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(f"{output_dir}/out.tif", 'w', **band_profile) as dest_file:
        for i, b in enumerate(true_bands):
            dest_file.write(b.read(1), i + 1)
            
    new_data, new_transform = downsample_dataset(f"{output_dir}/out.tif", scale_factor)

    return new_data, new_transform

def enhance_color_s2(data, uint16:bool=False):
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
    true_bands = [b for b in bands if os.path.basename(b.name).split("_")[2].replace(".jp2", "").strip() in ["B02", "B03", "B04"]]
    sortperm = np.argsort([b.name for b in true_bands])
    true_bands = [true_bands[i] for i in sortperm]
    band_profile = true_bands[0].profile
    band_profile.update({"count": len(true_bands)})
    return true_bands, band_profile


def plot_scene_s2(data, data_transform):
    """
    Plots the true color image and blue band of the scene given the names of the scenes and their affine transformation
    """
    _, (axt, axb) = plt.subplots(1,2, figsize=(10, 20))
    show(enhance_color_s2(data[1]), ax=axb, title="Single color band (B)", cmap="Blues", transform=data_transform)
    show(enhance_color_s2(data), ax=axt, title="True colour bands image", transform=data_transform)


def resize_bbox(bbox, scale_factor = 1.0):
    x_dim = bbox.right - bbox.left
    y_dim = bbox.top - bbox.bottom

    dx = ((scale_factor - 1) * x_dim) / 2
    dy = ((scale_factor - 1) * y_dim) / 2

    return BoundingBox(bbox.left - dx, bbox.bottom - dy, bbox.right + dx, bbox.top + dy)


def find_scene_bounding_box_lla(scene:str, scale_factor = 1.0):
    raster = rasterio.open(scene)
    raster_bounds = raster.bounds

    raster_bounds = resize_bbox(raster_bounds, scale_factor)

    raster_crs = raster.crs

    raster_proj = Proj(**raster_crs.data)

    west, south = raster_proj(raster_bounds.left, raster_bounds.bottom, inverse = True)
    east, north = raster_proj(raster_bounds.right, raster_bounds.top, inverse = True)

    bbox = f"{west},{south},{east},{north}"

    return bbox


def warp_affine_dataset(
    dataset_path:str, 
    output_path:str, 
    translation_x:float = 0.0, 
    translation_y:float = 0.0, 
    rotation_angle:float = 0.0, 
    scale:float = 1.0
):
    """
    Transforms the dataset accroding to given translation, rotation and scale params and writes it to the `output_path` file.
    """
    ref = rasterio.open(dataset_path)
    img = flip_img(ref.read()).copy()
    img_centre = (img.shape[1]//2, img.shape[0]//2)
    rotation_mat = cv.getRotationMatrix2D(img_centre, rotation_angle, scale)
    translation_mat = np.array([[1.0, 0.0, translation_x], [0.0, 1.0, translation_y]])
    affine_transform = np.matmul(rotation_mat, np.vstack([translation_mat, np.array([0, 0, 1])]))
    warped_img = cv.warpAffine(img, affine_transform, (img.shape[1], img.shape[0]))
    profile = ref.profile
    with rasterio.open(output_path, "w", **profile) as ds:
            for i in range(0, profile["count"]):
                ds.write(warped_img[:, :, i], i + 1)