import os

NAME = '异常流量数据分析软件'
os.system(f'python scripts/convert-ico.py ./desk.png')
os.system(
    f'pyinstaller ./Application.py --name {NAME} -w --noconfirm -i desk.ico')
