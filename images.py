from osgeo import gdal
import numpy as np

import rasterio
from rasterio.enums import Resampling

import torch

import os

list1 = [
    "byte",
    "uint8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "float32",
    "float64",
    "cint16",
    "cint32",
    "cfloat32",
    "cfloat64",
]
list2 = [
    gdal.GDT_Byte,
    gdal.GDT_Byte,
    gdal.GDT_UInt16,
    gdal.GDT_Int16,
    gdal.GDT_UInt32,
    gdal.GDT_Int32,
    gdal.GDT_Float32,
    gdal.GDT_Float64,
    gdal.GDT_CInt16,
    gdal.GDT_CInt32,
    gdal.GDT_CFloat32,
    gdal.GDT_CFloat64,
]


def imgread(path) -> np.ndarray:
    img = gdal.Open(path)
    c = img.RasterCount
    img_arr = img.ReadAsArray()
    if c > 1:
        img_arr = img_arr.swapaxes(1, 0)
        img_arr = img_arr.swapaxes(2, 1)
    del img
    return img_arr


def imgwrite(path, narray, compress="None") -> None:
    s = narray.shape
    dt_name = narray.dtype.name
    for i in range(len(list1)):
        if list1[i] in dt_name.lower():
            datatype = list2[i]
            break
        else:
            datatype = list2[0]
    if len(s) == 2:
        row, col, c = s[0], s[1], 1
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            path, col, row, c, datatype, options=["COMPRESS=" + compress]
        )
        dataset.GetRasterBand(1).WriteArray(narray)
        del dataset
    elif len(s) == 3:
        row, col, c = s[0], s[1], s[2]
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, col, row, c, datatype)
        for i in range(c):
            dataset.GetRasterBand(i + 1).WriteArray(narray[:, :, i])
        del dataset


bands = [
    ["10m", ("02", "03", "04", "08")],
    ["20m", ("05", "06", "07", "8A", "11", "12")],
    ["60m", ("01", "09", "10")],
]


def generate_bands(file_path, save_path):
    granule_dir = os.path.join(file_path, "GRANULE")
    granule = os.listdir(granule_dir)[0]
    granule_path = os.path.join(granule_dir, granule)

    img_dir = os.path.join(granule_path, "IMG_DATA")
    imgs = os.listdir(img_dir)

    img_10m = None
    img_20m = None
    img_60m = None

    for band in bands:
        print(band)
        band_name = band[0]

        img_concat = []
        for band_num in band[1]:
            print(band_num)
            band_num = "B" + band_num
            band_path = os.path.join(
                img_dir, [img for img in imgs if band_num in img][0]
            )

            img = imgread(band_path)
            img_dn = img[:, :, np.newaxis]
            img_concat.append(img_dn)

        fused_img = np.concatenate(img_concat, axis=2)

        os.makedirs(save_path, exist_ok=True)
        imgwrite(os.path.join(save_path, f"{band_name}.tif"), fused_img)


def crop_images(img_10m, img_20m, img_60m, save_path):
    img_10m = img_10m.astype(np.float32)
    img_20m = img_20m.astype(np.float32)
    img_60m = img_60m.astype(np.float32)

    img_20m = np.moveaxis(img_20m, -1, 0)
    img_60m = np.moveaxis(img_60m, -1, 0)

    img_20m = img_20m[np.newaxis, :, :, :]
    img_60m = img_60m[np.newaxis, :, :, :]

    upscale = torch.nn.Upsample(scale_factor=2)
    img_20m = torch.from_numpy(img_20m)
    img_20m = upscale(img_20m)
    img_20m = img_20m.numpy()
    print(img_20m.shape)

    upscale = torch.nn.Upsample(scale_factor=6)
    img_60m = torch.from_numpy(img_60m)
    img_60m = upscale(img_60m)
    img_60m = img_60m.numpy()
    print(img_60m.shape)

    img_20m = img_20m[0, :, :, :]
    img_60m = img_60m[0, :, :, :]

    img_20m = np.moveaxis(img_20m, 0, -1)
    img_60m = np.moveaxis(img_60m, 0, -1)

    crops_10m = generate_crops(img_10m)
    crops_20m = generate_crops(img_20m)
    crops_60m = generate_crops(img_60m)

    for i in range(len(crops_10m)):
        crop = np.concatenate([crops_10m[i], crops_20m[i], crops_60m[i]], axis=2)
        crop_path = f"{save_path}_{i}.tif"
        imgwrite(crop_path, crop)
        # print(crop_path)
        # print(crop.shape)


def generate_crops(img, crop_size=305):
    crops = []
    for i in range(0, img.shape[0], crop_size):
        for j in range(0, img.shape[1], crop_size):
            if len(img.shape) == 3:
                crops.append(img[i : i + crop_size, j : j + crop_size, :])
            else:
                crops.append(img[i : i + crop_size, j : j + crop_size])
    return crops


def main():
    gdal.UseExceptions()

    os.makedirs(os.path.join("bands", "train"), exist_ok=True)
    os.makedirs(os.path.join("bands", "test"), exist_ok=True)
    os.makedirs(os.path.join("data", "train"), exist_ok=True)
    os.makedirs(os.path.join("data", "test"), exist_ok=True)

    for dir in os.listdir("WHUS2-CD+"):
        for filename in os.listdir(os.path.join("WHUS2-CD+", dir)):
            if filename.endswith(".SAFE"):
                print(filename)
                save_dir = os.path.join("bands", dir, filename[:-5])

                if os.path.exists(os.path.join(save_dir, "60m.tif")):
                    continue

                generate_bands(
                    os.path.join("WHUS2-CD+", dir, filename),
                    save_dir,
                )

    for dir in os.listdir("bands"):
        for img_dir in os.listdir(os.path.join("bands", dir)):
            print(img_dir)
            save_dir = os.path.join("data", dir, img_dir)

            if os.path.exists(f"{save_dir}_1295_label.tif"):
                continue

            img_10m = imgread(os.path.join("bands", dir, img_dir, "10m.tif"))
            img_20m = imgread(os.path.join("bands", dir, img_dir, "20m.tif"))
            img_60m = imgread(os.path.join("bands", dir, img_dir, "60m.tif"))

            crop_images(img_10m, img_20m, img_60m, save_dir)

            label = imgread(os.path.join("ReferenceMask", img_dir) + "_Mask.tif")
            label[label == 128] = 0
            label[label == 255] = 1

            crops = generate_crops(label)
            for i in range(len(crops)):
                crop = crops[i]
                crop_path = f"{save_dir}_{i}_label.tif"
                imgwrite(crop_path, crop)


if __name__ == "__main__":
    main()
