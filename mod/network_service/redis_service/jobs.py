# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

redis job
"""
import requests
import time


def count_words_at_url(url):
    """
    计算目标url地址返回结果的字数
    :param url: str, 目标url
    :return: int, 字数
    """
    resp = requests.get(url)
    return len(resp.text.split())


def func_low(url):
    """
    低优先级job
    :param url:
    :return:
    """
    time.sleep(10)
    return 0


def func_default(url):
    """
    低优先级job
    :param url:
    :return:
    """
    time.sleep(5)
    return 1


def func_high(url):
    """
    低优先级job
    :param url:
    :return:
    """
    time.sleep(1)
    return 2