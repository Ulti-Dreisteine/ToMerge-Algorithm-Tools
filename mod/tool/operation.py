# -*- coding: utf-8 -*-
"""
Created on 2020/2/11 11:01

@Project -> File: ruima_galvanization_optimization -> operation.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""


def sort_dict_by_keys(adict: dict, reverse=False) -> dict:
    """将dict按照键进行排序"""
    adict = dict(sorted(adict.items(), key=lambda a: a[0], reverse=reverse))
    return adict


if __name__ == '__main__':
    adict = {3: 1, 2: 2, 1: 3}
    adict = sort_dict_by_keys(adict)
