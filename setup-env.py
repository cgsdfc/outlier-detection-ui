import os

print('安装依赖。。。')
os.system('pip3 install -i https://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com  -r ./requirements.txt')
print('建立Qt环境。。。')
os.system('pyqt5-tools installuic')
