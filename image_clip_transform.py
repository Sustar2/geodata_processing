"""
To use the shp clip raster, get independent raster intersect with feature in shp and transform them into 416*416
by Wenjing XUN

"""

from osgeo import gdal, gdalconst
from osgeo import ogr, osr
import numpy as np
import shutil
import os
from tqdm import tqdm
import cv2 as cv
import tifffile as tff

"""
def read_image(self):
    img = gdal.Open(self.img)
    img_width = img.RasterXSize
    img_height = img.RasterYSize
    img_geotrans = img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_data = img.ReadAsArray(0, 0, img_width, img_height)

    img_data = np.where(img_data<=0, 0, img_data)

    if (len(self.mask.split('/')[-1])>0):
        mask_data = tff.imread(self.mask)
    else:
        mask_data = np.ones((img_height,img_width),dtype=np.float32)

    nodata_value = img.GetRasterBand(1).GetNoDataValue()
    return img_data, mask_data, img_geotrans, img_proj, nodata_value, img_width, img_height

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
    else:
        for i in range(img_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset
"""

def read_image(self):
    img = gdal.Open(self)
    img_width = img.RasterXSize
    img_height = img.RasterYSize
    img_geotrans = img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_data = img.ReadAsArray(0, 0, img_width, img_height)

    img_data = np.where(img_data <= 0, 0, img_data)

    # if (len(self.mask.split('/')[-1])>0):
    #     mask_data = tff.imread(self.mask)
    # else:
    #     mask_data = np.ones((img_height,img_width),dtype=np.float32)

    nodata_value = img.GetRasterBand(1).GetNoDataValue()
    return img_data, img_geotrans, img_proj, nodata_value, img_width, img_height

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
    else:
        for i in range(img_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset

# def get_projection_Info(projection_info):
#     """
#     To get the value of 'false_easting' and 'false_northing'
#     :param projection_info: type:str
#     :return:
#     """
#     """
#     projection:
#     PROJCS["unnamed",GEOGCS["unnamed ellipse",DATUM["unknown",SPHEROID["unnamed",3396190,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]
#     split_projection:
#     ['PROJCS["unnamed",GEOGCS["unnamed ellipse",DATUM["unknown",SPHEROID["unnamed",3396190,0]',  'PRIMEM["Greenwich",0',
#     'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]',  'PROJECTION["Equirectangular"',  'PARAMETER["standard_parallel_1",0',
#      'PARAMETER["central_meridian",0',  'PARAMETER["false_easting",0', 'PARAMETER["false_northing",0', 'UNIT["metre",1,AUTHORITY["EPSG","9001"]', 'AXIS["Easting",EAST', 'AXIS["Northing",NORTH]]']
#
#     """
#     split_projection = projection_info.split('],')
#     projection_info_list = []
#     for sp in split_projection:
#         list_sp = list(sp)
#         list_new_sp = list_sp[:9]
#         str_sp = ''.join(list_new_sp)
#         if str_sp == 'PARAMETER':
#             projection_info_list.append(sp)
#     false_easting = 'NULL'
#     false_northing = 'NULL'
#     for proj in projection_info_list:
#         """
#         ['PARAMETER["central_meridian"', '0']
#         """
#         lis_proj = proj.split(',')
#         pro_param = lis_proj[0].split('[')[1]
#         if pro_param == '"false_easting"':
#             false_easting = lis_proj[1]
#         if pro_param == '"false_northing"':
#             false_northing = lis_proj[1]
#
#     return false_easting, false_northing

def get_projection_Info_by_osr(projection_info):
    """
    To extract the false_easting, false_northing and central_meridian from the projection info.
    PROJCS["unnamed",GEOGCS["unnamed ellipse",DATUM["unknown",SPHEROID["unnamed",3396190,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]
    """
    sr_proj = osr.SpatialReference()
    sr_proj.SetFromUserInput(projection_info)
    false_easting = sr_proj.GetProjParm("false_easting")
    false_northing = sr_proj.GetProjParm("false_northing")
    central_meridian = sr_proj.GetProjParm("central_meridian")
    del sr_proj

    return false_easting, false_northing, central_meridian

def transform_proj_Coor(projection_info, original_tranformation, new_proj= -1):
    """
    To transform the projection, espically the central_merdian
    then tranform the coordinate to the new one
    :param projection_info:
    :return:
    """
    if new_proj != -1:
        update_projection = new_proj
        sr_proj = None
    else:
        sr_proj = osr.SpatialReference()
        sr_proj.SetFromUserInput(projection_info)
        sr_proj.SetProjParm("central_meridian", 0.0)
        update_projection = str(sr_proj)

    sourceProj = osr.SpatialReference()
    targetProj = osr.SpatialReference()
    sourceProj.SetFromUserInput(projection_info)
    targetProj.SetFromUserInput(update_projection)
    coortranform = osr.CoordinateTransformation(targetProj, sourceProj)
    oriX = original_tranformation[0]
    oriY = original_tranformation[3]
    update_trans = list(original_tranformation)
    tranformXY = coortranform.TransformPoint(oriX, oriY)
    if tranformXY:
        updateX = tranformXY[0]
        updateY = tranformXY[1]
    else:
        print('wrong')

    update_trans[0] = updateX
    update_trans[3] = updateY

    del sr_proj, sourceProj, targetProj, coortranform

    return update_projection, update_trans


def reproject_raster(raster_proj, raster_trans):

    update_trans = raster_trans
    false_easting, false_northing, central_meridian = get_projection_Info_by_osr(raster_proj)
    if central_meridian != 0:
        update_proj, update_trans = transform_proj_Coor(raster_proj, raster_trans)
        return True, update_trans, update_proj
    else:
        return False, update_trans, -1

# def generate_name(count, fold_path, name, add_string, type):
#     """
#     To group the name between the folder, name, type and so on
#     :param count:
#     :param fold_path:
#     :param name:
#     :param add_string:
#     :param type:
#     :return:
#     """
#     out_names = []
#     for c in count:
#         names = fold_path + name + add_string + str(c) + type
#         out_names.append(names)
#     return out_names

def calculate_shp_XY(feature):
    """
    To calcalute the maxX,minX,maxY,minY
    for Polygon
    :param feature:
    :return:
    """
    geom = feature.GetGeometryRef()
    string_geom = str(geom)
    """
    data show as:
    POLYGON ((-10509507.9652 -3393304.4149 0,-10509568.8918866 -3393358.12999881 0,-10509447.5696223 -3393250.10332512 0,-10509507.9652 -3393304.4149 0))
    ['POLYGON ((-10509507.9652 -3393304.4149 0', '-10509568.8918866 -3393358.12999881 0','-10509447.5696223 -3393250.10332512 0', '-10509507.9652 -3393304.4149 0))']
    after split twice, data shown as:
    [['POLYGON', '((-10509507.9652', '-3393304.4149', '0'], ['-10509568.8918866', '-3393358.12999881', '0'],['-10509447.5696223', '-3393250.10332512', '0'], ['-10509507.9652', '-3393304.4149', '0))']]
    """
    first_split = string_geom.split(',')

    second_split = []
    for value, fir in enumerate(first_split):
        split = fir.split(' ')
        final_split = split
        if value != 0:
            if len(split) > len(second_split[0]) - 1:
                final_split = split[-(len(second_split[0]) - 1):]
        for m, sp in enumerate(final_split):
            if ')' in sp:
                final_split[m] = sp.replace(")", "")
            if '(' in sp:
                final_split[m] = sp.replace("(", "")
        second_split.append(final_split)

    """
    when spliting, there are something wrongly split in the first and last spliting, it will be optimize[cthe first and last point are same]
    first_coors:['((-10509507.9652', '-3393304.4149', '0']
    opti_coor:-10509507.9652
    coordinames:[['-10509507.9652', '-3393304.4149', '0'], ['-10509568.8918866', '-3393358.12999881', '0'],['-10509447.5696223', '-3393250.10332512', '0'], ['-10509507.9652', '-3393304.4149', '0']]
    """
    first_coors = second_split[0][-(len(second_split[0]) - 1):]
    coordinates = second_split
    coordinates[0] = first_coors

    "To find the maxX,minX,maxY,minY"
    geoXs = []
    geoYs = []
    for coor in coordinates:
        geoXs.append(float(coor[0]))
        geoYs.append(float(coor[1]))
    maxX = max(geoXs)
    minX = min(geoXs)
    maxY = max(geoYs)
    minY = min(geoYs)
    test_data = [minX, maxX, minY, maxY]
    return test_data

def get_raster_range(raster_geotrans, raster_data):
    """
    to get the raster range
    :param raster_path:
    :return:
    """
    transform = list(raster_geotrans)
    minX = transform[0]
    maxY = transform[3]
    maxX = transform[0] + transform[1] * (raster_data.shape[1] - 1) + (raster_data.shape[0] - 1) * transform[2]
    minY = transform[3] + transform[4] * (raster_data.shape[0] - 1) + (raster_data.shape[1] - 1) * transform[5]

    polar_data = [minX, maxX, minY, maxY]
    return polar_data

# def judge_contain(test_data, polar_data):
#     """
#     to judge if the location is in the polar_data range
#     :param test_data: [minX,maxX,minY,maxY]
#     :param polar_data: [minX, maxX, minY, maxY]
#     :return:
#     """
#     if test_data[0] >= polar_data[0] and test_data[1] < polar_data[1] and test_data[2] >= polar_data[2] and test_data[3] < polar_data[3]:
#         return True
#     else: return False

def get_shp_info(clip_shp_path):
    """
    To get the shp range
    :param shp_range:
    :param raster_range:
    :return:
    """
    # shp_centerX = []
    # shp_centerY = []
    shp_range = []
    shp_minX = []
    shp_maxX = []
    shp_minY = []
    shp_maxY = []
    shp_data = ogr.Open(clip_shp_path)
    shp_layer = shp_data.GetLayer()
    dir(shp_layer)
    shp_layer_count = shp_layer.GetFeatureCount()
    for i in range(shp_layer_count):
        feature = shp_layer.GetFeature(i)
        feature_range = calculate_shp_XY(feature)
        # centerX = (feature_range[0] + feature_range[1])/2
        # centerY = (feature_range[2] + feature_range[3])/2
        # shp_centerX.append(centerX)
        # shp_centerY.append(centerY)
        shp_minX.append(feature_range[0])
        shp_maxX.append(feature_range[1])
        shp_minY.append(feature_range[2])
        shp_maxY.append(feature_range[3])
    shp_range.append(shp_minX)
    shp_range.append(shp_maxX)
    shp_range.append(shp_minY)
    shp_range.append(shp_maxY)

    del shp_data

    return np.array(shp_range)

def get_shp_in_raster_FIDs(shp_range, raster_geotrans, raster_data):
    """

    :param shp_range:
    :param raster_geotrans:
    :return:
    shp_range = [shp_minX,shp_maxX,shp_minY,shp_maxY]
    raster_range = [raster_minX,raster_maxX,raster_minY,raster_maxY]
    """
    raster_range = get_raster_range(raster_geotrans, raster_data)
    shp_in_raster_FIDs = np.argwhere(
        (raster_range[0] <= shp_range[0]) & (shp_range[1] <= raster_range[1]) & (raster_range[2] <= shp_range[2]) & (
                shp_range[3] <= raster_range[3]))

    return shp_in_raster_FIDs


def shp_clip_raster(in_shp_path, shp_range, in_fold_path, raster_name, raster_geotrans, raster_data, out_fold_path):
    """
    to clip the raster by the shp
    clip shape is circle
    :param in_shp_path:
    :param in_fold_path:
    :param raster_name:
    :param out_fold_path:
    :return:
    """

    out_raster_paths = []
    raster_path = in_fold_path + raster_name + '.tif'
    shp_in_raster_FIDs = get_shp_in_raster_FIDs(shp_range, raster_geotrans, raster_data)

    for ID in shp_in_raster_FIDs:
        FID = ID[0]
        out_raster_path = out_fold_path + raster_name + '_clip' + str(FID) + '.tif'
        out_raster_paths.append(out_raster_path)
        wheretext = 'FID = {}'.format(FID)
        result = gdal.Warp(
            out_raster_path,
            raster_path,
            format='GTiff',
            cutlineDSName=in_shp_path,
            cutlineWhere=wheretext,
            cropToCutline=True
        )
        result.FlushCache()
        del result
    return shp_in_raster_FIDs, out_raster_paths

def shp_clip_raster_without_warp(shp_range, raster_data, raster_geotrans, raster_proj):
    """
    clip shape is square
    :param in_shp_path:
    :param shp_range:
    :param in_fold_path:
    :param raster_name:
    :param raster_geotrans:
    :param out_fold_path:
    :return:
    """
    clip_rasters = []
    shp_in_raster_FIDs = get_shp_in_raster_FIDs(shp_range, raster_geotrans, raster_data)

    for ID in shp_in_raster_FIDs:
        FID = ID[0]
        FID_shp_range = [shp_range[0][FID], shp_range[1][FID], shp_range[2][FID], shp_range[3][FID]]

        raster_clip, update_trans = shp_feature_clip_raster_single(FID_shp_range, raster_data, raster_geotrans)
        clip_raster_info = [FID, raster_clip, update_trans, raster_proj]
        clip_rasters.append(clip_raster_info)

    return clip_rasters

def shp_feature_clip_raster_single(shp_range, raster_data, raster_geotrans):

    pixel_minX = max(0, (shp_range[0] - raster_geotrans[0]) / raster_geotrans[1])
    pixel_maxX = min(raster_data.shape[1], (shp_range[1] - raster_geotrans[0]) / raster_geotrans[1])
    pixel_minY = max(0, (shp_range[3] - raster_geotrans[3]) / raster_geotrans[5])
    pixel_maxY = min(raster_data.shape[0], (shp_range[2] - raster_geotrans[3]) / raster_geotrans[5])
    raster_clip = raster_data[int(pixel_minY):int(pixel_maxY), int(pixel_minX):int(pixel_maxX)]

    update_trans = list(raster_geotrans)
    update_trans[0] = raster_geotrans[0] + raster_geotrans[1] * int(pixel_minX) + int(pixel_minY) * raster_geotrans[2]
    update_trans[3] = raster_geotrans[3] + raster_geotrans[4] * int(pixel_minX) + int(pixel_minY) * raster_geotrans[5]

    return raster_clip, update_trans

def pad_image(opsize, image, oritrans, orisize, loca):
    """
    when the length of the image smaller than the wanted size, this function could help to pad the image by and '0' value
    to make the image zoom out to be same as the wanted size.
    :param opsize: wanted size
    :param image: image matrix
    :param oritrans:[0图像左上角的X坐标, 1图像东西方向分辨率, 2旋转角度如果图像北方朝上该值为0, 3图像左上角的Y坐标,
                     4旋转角度如果图像北方朝上该值为0,5图像南北方向分辨率 ]
    :param orisize: original size
    :param loca: the location is row or column,row(y):0 column(x):1
    :return:
    """
    update_image = image
    update_trans = list(oritrans)
    diff = opsize - orisize
    if diff/2 == 0:
        if loca == 0:
            update_image = np.pad(image, ((int(diff/2), int(diff/2)),(0, 0)), 'constant')
            update_trans[3] = oritrans[3] - diff/2 * oritrans[5]
        if loca == 1:
            update_image = np.pad(image, ((0, 0), (int(diff/2), int(diff/2))), 'constant')
            update_trans[0] = oritrans[0] - diff/2 * oritrans[1]
    if diff/2 != 0:
        before_move = int((diff - 1) / 2)
        after_move = int((diff + 1) / 2)
        if loca == 0:
            update_image = np.pad(image, ((before_move, after_move), (0, 0)), 'constant')
            update_trans[3] = oritrans[3] - before_move * oritrans[5]
        if loca == 1:
            update_image = np.pad(image, ((0, 0), (before_move, after_move)), 'constant')
            update_trans[0] = oritrans[0] - after_move * oritrans[1]
    return update_image, update_trans

def calcute_transform(opsize, size, image, original_transform, loca):
    """
    to judge between the length and oplength
    :param opsize:
    :param size:
    :param image:
    :param original_transform:
    :param loca: x:0 y:1
    :return:
    """
    update_image = image
    update_transform = list(original_transform)
    if opsize == size:
        _ = image
    if opsize < size:
        if loca == 0:
            update_image = cv.resize(image, (416, image.shape[1]))
            update_transform[5] = original_transform[5] * image.shape[0] / 416
        if loca == 1:
            update_image = cv.resize(image, (416, image.shape[0]))
            update_transform[1] = original_transform[1] * image.shape[1] / 416
    if opsize > size:
        if loca == 0:
            update_image, update_transform = pad_image(opsize, image, original_transform, image.shape[0], loca)
        if loca == 1:
            update_image, update_transform = pad_image(opsize, image, original_transform, image.shape[1], loca)

    return update_transform, update_image


def recalcute_transform(img, img_geotrans):
    """
    """
    update_img = cv.resize(img, (416, 416))
    update_geotrans = list(img_geotrans)
    ori_geotransX = img_geotrans[1] * img.shape[1] / 416
    ori_geotransY = img_geotrans[5] * img.shape[0] / 416
    update_geotrans[1] = ori_geotransX
    update_geotrans[5] = ori_geotransY
    update_geotrans[1] = ori_geotransX * img.shape[1] / 416
    update_geotrans[5] = ori_geotransY * img.shape[0] / 416

    return update_img, update_geotrans



"_main_"
if __name__ == '__main__':
    "parameters config"
    # clip_shp_path = r'G:\test_data\E-180_N32clip_shp.shp'
    # in_fold_path = r'G:\test_data\test_data2/'
    # save_fold_path = r'G:\test_data\test_result/'
    clip_shp_path = r'G:\clip_shp_data\cone_data_20231215\global_volcano_catalog_private.shp'
    in_fold_path = r'F:\ctx_global_mosaic/'
    save_fold_path = r'G:\result\transform_result_1222/'
    opsizeX = 416
    opsizeY = 416

    "get raster_names list"
    # raster_name 设置时不需要加.tif
    raster_names = []
    raster_list = os.listdir(in_fold_path)
    raster_list.sort()
    for raster in raster_list:
        raster_name, raster_type = os.path.splitext(raster)
        if raster_type == '.tif':
            raster_names.append(raster_name)

    shp_range = get_shp_info(clip_shp_path)

    for raster_name in tqdm(raster_names, desc='clip_transform'):
        raster_path = in_fold_path + raster_name + '.tif'
        img_data, img_geotrans, img_proj, nodata_value, img_width, img_height = read_image(raster_path)
        flag, update_trans, update_proj = reproject_raster(img_proj, img_geotrans)

        if flag:
            clip_rasters = shp_clip_raster_without_warp(shp_range, img_data, update_trans, update_proj)
        else:
            clip_rasters = shp_clip_raster_without_warp(shp_range, img_data, img_geotrans, img_proj)

        for clip_raster in clip_rasters:
            "clip_raster_info = [FID, raster_clip, update_trans, raster_proj]"

            save_path = save_fold_path + str(clip_raster[0]) + '.tif'
            transformX, imageX = calcute_transform(opsizeX, clip_raster[1].shape[0], clip_raster[1], clip_raster[2],
                                                   0)
            transformXY, imageXY = calcute_transform(opsizeY, imageX.shape[1], imageX, transformX, 1)
            update_transform = tuple(transformXY)
            update_image = imageXY

            save_image(save_path, update_image, update_transform, clip_raster[3])


        # if flag:
        #     shp_in_raster_FIDs, out_raster_paths = shp_clip_raster(clip_shp_path, shp_range, reproject_fold_path, raster_name, update_trans, img_data, clip_fold_path)
        # else:
        #     shp_in_raster_FIDs, out_raster_paths = shp_clip_raster(clip_shp_path, shp_range, in_fold_path, raster_name, img_geotrans, img_data, clip_fold_path)

        # save_image_paths = []
        # for fid in shp_in_raster_FIDs:
        #     fold_path = save_fold_path + str(fid[0]) + '.tif'
        #     save_image_paths.append(fold_path)


        # for c, path in enumerate(out_raster_paths):
        #     img_data, img_geotrans, img_proj, nodata_value, img_width, img_height = read_image(path)
        #     "tranform"
        #     transformX, imageX = calcute_transform(opsizeX, img_data.shape[0], img_data, img_geotrans, 0)
        #     transformXY, imageXY = calcute_transform(opsizeY, imageX.shape[1], imageX, transformX, 1)
        #     update_transform = tuple(transformXY)
        #     update_image = imageXY
        #
        #     save_image(save_image_paths[c], update_image, update_transform, img_proj)
        #     "transform"
        #     # print('The {}th volcano transformation is finished!'.format(c))
        # # print('The {} image searching is finished!'.format(raster_name))
        # "clear the middle folder"
        # shutil.rmtree(clip_fold_path)
        # os.mkdir(clip_fold_path)


