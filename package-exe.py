import os

# NAME = '异常流量数据分析软件'
from NAME import NAME

os.system(f'python scripts/convert-ico.py ./desk.png')
os.system(
    f'pyinstaller ./app.py --name {NAME} --noconfirm -i desk.ico')
