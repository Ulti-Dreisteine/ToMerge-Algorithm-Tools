# -*- coding: utf-8 -*-
"""
Created on 2020/2/24 15:25

@Project -> File: realtime-wind-rose-diagram -> upload_file.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 文件上传
"""

import requests
import json


class UploadFile(object):
    """文件上传"""

    def __init__(self):
        pass

    @staticmethod
    def obtain_object_id_and_url(request_upload_file_url, expires_in=None):
        """
        拉取上传任务id和地址url
        :param request_upload_file_url: str, 请求url
        :param expires_in: int, 失效时间
        """
        if expires_in is None:
            expires_in = 600

        params = {'expiresIn': expires_in}

        retry = 0
        while True:
            resp = requests.put(request_upload_file_url, params=params)
            if retry < 3:
                if resp.status_code >= 500:
                    # 重试请求.
                    print('Retry requesting, retry = {}'.format(retry))
                    retry += 1
                    continue
                elif resp.status_code in [200, 204]:
                    break
            else:
                raise RuntimeError(
                    'ERROR: reach max request time = 3, cannot obtain object_id & upload_url')

        resp_data = json.loads(resp.text)
        object_id, put_file_url = resp_data['objectID'], resp_data['putURL']
        return object_id, put_file_url

    @staticmethod
    def upload_file(put_file_url, file_path, content_type):
        """
        上传文件
        :param file_path: str, 待上传文件本地路径
        :param content_type: str, 指定文件类型, 如'image/png'等
        :param put_file_url: str, 用于上传文件的url地址

        Example:
        ------------------------------------------------------------
        upload = UploadFile()
        object_id, put_file_url = upload.obtain_object_id_and_url(request_upload_file_url)
        upload.upload_file(put_file_url, file_path = file_path, content_type = 'image/png')
        ------------------------------------------------------------
        """
        headers = {
            'Content-Type': content_type
        }
        file = open(file_path, 'rb')

        retry = 0
        while True:
            resp = requests.put(put_file_url, data=file, headers=headers)
            if retry < 3:
                if resp.status_code >= 500:
                    # 重试请求.
                    print('Retry requesting, retry = {}'.format(retry))
                    retry += 1
                    continue
                elif resp.status_code in [200, 204]:
                    break
            else:
                raise RuntimeError(
                    'ERROR: reach max request time = 3, cannot upload file')
