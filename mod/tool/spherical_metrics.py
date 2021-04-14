# -*- coding: utf-8 -*-
"""
Created on 2020/1/19 上午11:37

@Project -> File: pollution-pdmc-relevance-analyzer -> spherical_metrics.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 球面距离计算
"""

from math import sin, asin, cos, radians, fabs, sqrt


def haversine(theta):
    s = sin(theta / 2)
    return s * s


def get_haver_distance(lat0, lng0, lat1, lng1):
    """
    用haversine公式计算球面两点间的距离。
    :param lat0: 第一个地址的纬度
    :param lng0: 第一个地址的经度
    :param lat1: 第二个地址的纬度
    :param lng1: 第二个地址的经度
    :return: 两点间的球面距离，单位：米（m）
    """

    earth_radius = 6371  # 地球平均半径, 6371km

    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = haversine(dlat) + cos(lat0) * cos(lat1) * haversine(dlng)
    distance = 2 * earth_radius * asin(sqrt(h))

    return 1000 * distance


def is_in_radius(loc, target_loc, radius):
    """判断loc是否在以target_loc为圆心，半径radius的圆内"""
    d = get_haver_distance(loc[1], loc[0], target_loc[1], target_loc[0])
    if d < radius:
        return True
    else:
        return False
