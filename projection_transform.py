# -*- coding: utf-8 -*-
# @Author  : XUNWJ
# @Contact : ssssustar@163.com
# @File    : projection_transform.py
# @Time    : 2025/4/16 19:19
# @Desc    :
from __future__ import print_function
from __future__ import division
from image_clip_transform import *



if __name__ == '__main__':
    input_img_path = r"G:\Albedo\Lunar_albedo\LRO_WAC_albedo\WAC_EMP_643NM_E300N3150_304P_project.tif"
    output_img_path = r"G:\Albedo\Lunar_albedo\LRO_WAC_albedo\WAC_EMP_643NM_E300N3150_304P_project2.tif"
    img_data, img_geotrans, img_proj, nodata_value, img_width, img_height = read_image(input_img_path)
    update_projection, update_trans = transform_proj_Coor(img_proj, img_geotrans, new_proj=-1)
    save_image(output_img_path, img_data, update_trans, update_projection)
