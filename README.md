# general-algorithm-tools

## 介绍

通用算法工具

## 软件架构

```text
general-algorithm-tools
├─ .gitignore
├─ config
│  ├─ default.yml
│  ├─ model_config.yml
│  └─ test_params.yml
├─ data
│  ├─ patient
│  │  └─ data.csv
│  ├─ siso
│  │  └─ siso.csv
│  └─ weather
│     └─ data.csv
├─ mod
│  ├─ config
│  │  ├─ config_loader.py
│  │  └─ __init__.py
│  ├─ data_binning              # 数据分箱
│  │  ├─ multivar_binning.py
│  │  ├─ univar_binning.py
│  │  └─ __init__.py
│  ├─ data_encoding             # 数据编码
│  │  ├─ univar_encoding.py
│  │  └─ __init__.py
│  ├─ data_process              # 数据处理
│  │  ├─ numpy.py
│  │  ├─ pandas.py
│  │  └─ __init__.py
│  ├─ decorator                 # 装饰器
│  │  ├─ timeout.py
│  │  └─ __init__.py
│  ├─ math                      # 数学相关
│  │  ├─ partial_derives.py
│  │  ├─ point_surface_distance.py
│  │  ├─ README.md
│  │  ├─ symbolic_operations.py
│  │  └─ __init__.py
│  ├─ network_service           # 网络服务
│  │  ├─ call_back.py
│  │  ├─ pgsql_connector.py
│  │  ├─ request_device_op.py
│  │  ├─ request_task_op.py
│  │  ├─ upload_file.py
│  │  └─ __init__.py
│  ├─ numerical_optimization    # 数值优化
│  │  ├─ genetic_algorithm.py
│  │  ├─ particle_swarm_optimization.py
│  │  └─ __init__.py
│  ├─ statistic                 # 统计工具
│  │  ├─ entropy.py
│  │  ├─ stat_tools.py
│  │  └─ __init__.py
│  ├─ tool                      # 工具
│  │  ├─ dir_file_op.py
│  │  ├─ operation.py
│  │  ├─ spherical_metrics.py
│  │  ├─ time_conversion.py
│  │  └─ __init__.py
│  └─ __init__.py
├─ proj_dir_struc.yml
├─ README.md
├─ src
│  └─ settings.py
├─ test
│  ├─ test_ga_optimization.py
│  └─ test_univar_encoding.py
└─ __init__.py

```

## 安装教程

略

## 使用说明

略
