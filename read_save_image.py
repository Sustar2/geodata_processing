"""
read save image in the coordinates and projection
"""
from osgeo import gdal
import os
import numpy as np



def read_image(img_path):
    img = gdal.Open(img_path)
    img_width = img.RasterXSize  # 栅格矩阵的列数
    img_height = img.RasterYSize  # 栅格矩阵的行数
    img_geotrans = img.GetGeoTransform()  # 仿射矩阵
    img_proj = img.GetProjection()  # 地图投影信息
    img_data = img.ReadAsArray(0, 0, img_width, img_height)  # 将数据写成数组，对应栅格矩阵

    return img_data, img_geotrans, img_proj

def get_data_resolution(data_info):
    """

    :param:data_info: obtained from the excel file:
    region_list = [dem_name,
                   [img_name,  info_dict],
                   [img_name,  info_dict],
                   ...]
    data_info = [region_listA,
                 region_listB,
                 region_listC,
                 ...]
    :return: resolution_data_groups: [[img_name1, x_resolution1, y_resolution1],
                                    [img_name2, x_resolution2, y_resolution2],
                                    [img_name3, x_resolution3, y_resolution3],
                                    ...]
    """
    resolution_data_groups = []
    img_paths = []

    for region_group in data_info:
        img_groups = region_group[1:]
        for img_group in img_groups:
            img_path = img_group[0]
            img_paths.append(img_path)


    for img_path in img_paths:
        base_name = os.path.basename(img_path)
        img_data, img_geotrans, img_proj = read_image(img_path)
        x_resolution = abs(img_geotrans[1])
        y_resolution = abs(img_geotrans[5])
        resolution_data_group = [base_name, x_resolution, y_resolution]
        resolution_data_groups.append(resolution_data_group)

    return resolution_data_groups

def save_image(save_path, img, img_geotrans, img_projection):
    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

        # 判读数组维数
    if len(img.shape) == 3:
        img_bands, img_height, img_width = img.shape
    else:
        img_bands, (img_height, img_width) = 1, img.shape

        # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(save_path, img_width, img_height, img_bands, datatype)

    dataset.SetGeoTransform(img_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(img_projection)  # 写入投影

    if img_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img)  # 写入数组数据
        dataset.GetRasterBand(1).SetNoDataValue(255)
    else:
        for i in range(img_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img[i])
            dataset.GetRasterBand(i + 1).SetNoDataValue(255)

    del dataset

def read_image_memmap(img_path, memmap_path=r'G:\Image_shadowcam\temp_array.dat'):
    img = gdal.Open(img_path)
    img_width = img.RasterXSize
    img_height = img.RasterYSize
    img_geotrans = img.GetGeoTransform()
    img_proj = img.GetProjection()

    # 使用 memmap 存储大数据，避免内存溢出
    img_data = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(img.RasterCount, img_height, img_width))

    for i in range(img.RasterCount):
        band = img.GetRasterBand(i + 1)
        img_data[i, :, :] = band.ReadAsArray()

    return img_data, img_geotrans, img_proj


def read_image_by_blocks(img_path, block_size=1024):
    img = gdal.Open(img_path)
    img_width = img.RasterXSize
    img_height = img.RasterYSize
    img_geotrans = img.GetGeoTransform()  # 仿射矩阵
    img_proj = img.GetProjection()  # 地图投影信息
    img_data = []

    for i in range(img.RasterCount):
        band = img.GetRasterBand(i + 1)
        band_data = []

        for y in range(0, img_height, block_size):
            num_rows = min(block_size, img_height - y)
            row_data = []

            for x in range(0, img_width, block_size):
                num_cols = min(block_size, img_width - x)
                block = band.ReadAsArray(x, y, num_cols, num_rows)
                row_data.append(block)

            band_data.append(row_data)
        img_data.append(band_data)

    return img_data, img_geotrans, img_proj


if __name__ == '__main__':
    pass
    raster_path = r"G:\Image_shadowcam\raster_collection\shp_00\M031081886S_map_proj.tif"
    raster_data, raster_geotrans, raster_proj = read_image_by_blocks(raster_path)
    pass