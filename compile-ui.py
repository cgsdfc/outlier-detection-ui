import os

print('编译资源。。。')
os.system('pyuic5.exe -x -o UserInterface.py UserInterface.ui')
print('编译UI文件。。。')
os.system('pyrcc5 res.qrc -o res_rc.py')
