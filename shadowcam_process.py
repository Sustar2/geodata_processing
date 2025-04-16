# -*- coding: utf-8 -*-
# @Author  : XUNWJ
# @Contact : ssssustar@163.com
# @File    : shadowcam_process.py
# @Time    : 2024/10/25 14:47
# @Desc    :
from __future__ import print_function
from __future__ import division

import sys, os, glob, arcpy
from tqdm import tqdm
from osgeo import ogr, osr
import gc
import shutil
# from pyproj import CRS, Transformer

from read_save_image import *
from image_clip_transform import calculate_shp_XY, shp_feature_clip_raster_single, get_raster_range, transform_proj_Coor

def raster_mosaicing(raster_folder_path):
    # 找到所有的栅格文件
    raster_files = glob.glob(os.path.join(raster_folder_path, "*.tif")) # 替换为实际路径

    # 打开栅格文件并进行镶嵌
    output_path = os.path.join(raster_folder_path, os.path.basename(raster_folder_path)[:2] + "_mosaic.tif")

    for raster in raster_files:
        raster_data, raster_geotrans, raster_proj = read_image(raster)
        pass

def raster_clipping(raster_folder_path, shp_layer, refernce_proj_info):
    shp_FID = int(raster_folder_path.split('_')[-1])
    shp_feature = shp_layer.GetFeature(shp_FID)
    shp_range = calculate_shp_XY(shp_feature)

    for raster_path in tqdm(os.listdir(raster_folder_path)):
        if raster_path.endswith('.tif') and 'raw' in raster_path:
            raster_all_path = os.path.join(raster_folder_path, raster_path)
            raster_save_name = raster_all_path.replace('raw', 'clip')
            raster_proj_name = raster_all_path.replace('raw', 'proj')

            if os.path.exists(raster_save_name):
                # raster_data, raster_geotrans, raster_proj = read_image(raster_save_name)
                # raster_data[raster_data > 1] = 255
                # raster_data[raster_data < -1] = 255
                # save_image(raster_save_name, raster_data, raster_geotrans, raster_proj)
                # del raster_data, raster_geotrans, raster_proj
                # gc.collect()
                continue

            raster_data, raster_geotrans, raster_proj = read_image(raster_all_path)
            raster_data[raster_data > 1] = 255
            raster_data[raster_data < -1] = 255


            if raster_proj == refernce_proj_info:
                # raster_range = get_raster_range(raster_geotrans, raster_data)
                raster_clip, update_trans_clip = shp_feature_clip_raster_single(shp_range, raster_data, raster_geotrans)
                save_image(raster_save_name, raster_clip, update_trans_clip, raster_proj)
                # 释放内存
                del raster_data, raster_geotrans, raster_proj, raster_clip, update_trans_clip
                gc.collect()  # 强制垃圾回收
            else:
                # update_proj, update_trans = transform_p roj_Coor(raster_proj, raster_geotrans, reference_proj_info)
                # update_proj, update_trans2 = transform_proj_Coor_proj(raster_proj, raster_geotrans, reference_proj_info)
                # update_range = get_raster_range(update_trans, raster_data)
                del raster_data, raster_geotrans, raster_proj
                gc.collect()
                if not os.path.exists(raster_proj_name):
                    transofrm_proj_arcpy(raster_all_path, raster_proj_name, refernce_proj_info)
                new_raster_data, new_raster_geotrans, new_raster_proj = read_image(raster_proj_name)
                new_raster_data[new_raster_data > 1] = 255
                new_raster_data[new_raster_data < -1] = 255
                raster_clip, update_trans_clip = shp_feature_clip_raster_single(shp_range, new_raster_data, new_raster_geotrans)
                save_image(raster_save_name, raster_clip, update_trans_clip, refernce_proj_info)
                # 释放内存
                del new_raster_data, new_raster_geotrans, new_raster_proj, raster_clip, update_trans_clip
                gc.collect()  # 强制垃圾回收

def transform_proj_Coor_proj(projection_info, original_transformation, new_proj=None):
    """
    使用 pyproj 库进行投影变换。
    :param projection_info: 原始投影信息（WKT、EPSG 代码等）
    :param original_transformation: 原始的地理变换 (包括原始的 X 和 Y 坐标)
    :param new_proj: 新的投影信息 (WKT、EPSG 代码等)；默认为 None。
    :return: 新的投影信息和变换后的坐标。
    """
    # 定义原始投影和目标投影
    source_crs = CRS.from_user_input(projection_info)

    if new_proj:
        target_crs = CRS.from_user_input(new_proj)
    else:
        # 创建一个新投影，将 central_meridian 设置为 0
        target_crs = source_crs.to_proj4()

    # 建立投影转换
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # 获取原始坐标
    oriX, oriY = original_transformation[0], original_transformation[3]

    # 执行坐标转换
    updateX, updateY = transformer.transform(oriX, oriY)

    # 更新地理变换参数
    update_trans = list(original_transformation)
    update_trans[0], update_trans[3] = updateX, updateY

    return target_crs.to_wkt(), update_trans

def transofrm_proj_arcpy(input_raster_path, output_raster_path, target_project_info):

    # 创建 SpatialReference 对象并导入 GDAL 的 WKT
    srs = osr.SpatialReference()
    srs.ImportFromWkt(target_project_info)

    # 将 GDAL 的 WKT 转换为 ESRI WKT
    srs.MorphToESRI()
    esri_wkt = srs.ExportToWkt()
    # 投影的目标坐标系统 (可以是坐标系统名称、WKT 或 WKID)
    target_coordinate_system = arcpy.SpatialReference()
    target_coordinate_system.loadFromString(esri_wkt)

    # 投影栅格
    arcpy.management.ProjectRaster(
        in_raster=input_raster_path,
        out_raster=output_raster_path,
        out_coor_system=target_coordinate_system,
        resampling_type="NEAREST",  # 重采样方法，例如 NEAREST、BILINEAR、CUBIC 等
        cell_size="",
        geographic_transform=""
    )
    del srs, target_coordinate_system

def read_geotrans(raster_path):
    img = gdal.Open(raster_path)
    img_geotrans = img.GetGeoTransform()  # 仿射矩阵

    return img_geotrans

def raster_copy(raster_origin_folder, raster_destination_folder):
    for folder_name in os.listdir(raster_origin_folder):
        if 'shp_' in folder_name:
            print('The processing folder is {}'.format(folder_name))
            os.makedirs(os.path.join(raster_destination_folder, folder_name), exist_ok=True)
            for raster_path in tqdm(os.listdir(os.path.join(raster_origin_folder, folder_name))):
                if raster_path.endswith('.tif') and 'clip' in raster_path:
                    shutil.copy2(os.path.join(raster_origin_folder, folder_name, raster_path), os.path.join(raster_destination_folder, folder_name))



if __name__ == '__main__':
    pass
    raster_folder = r'G:\Image_shadowcam\new_version'
    shp_path = r"G:\Image_shadowcam\shapes\PSR_clip.shp"
    proj_reference_raster = r"G:\Image_shadowcam\new_version\shp_00\M015846391S_map_proj.tif"
    raster_destination_folder = r'G:\Image_shadowcam\raster_mosaic'

    # raster_mosaicing(raster_folder)

    # region clip raster
    shp_data = ogr.Open(shp_path)
    shp_layer = shp_data.GetLayer()
    _, _, reference_proj_info = read_image(proj_reference_raster)

    for folder in os.listdir(raster_folder):
        if 'shp_' in folder:
            print('The processing folder is {}'.format(folder))
            raster_clipping(os.path.join(raster_folder, folder), shp_layer, reference_proj_info)
    # endregion

    # region sort file
    # raster_copy(raster_folder, raster_destination_folder)
    # endregion