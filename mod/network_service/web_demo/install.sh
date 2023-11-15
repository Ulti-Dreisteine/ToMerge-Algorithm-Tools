# 安装lake
mkdir tmp_lib
cd tmp_lib
git clone https://github.com/CosmosShadow/lake && cd lake && git checkout develop && python setup.py install
cd ..

# 安装gunicorn
pip install gunicorn