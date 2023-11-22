import os

# for python 3.8+
#os.add_dll_directory('D:\\Program Files\\Basler\\Runtime\\x64')
os.add_dll_directory('C:\\Program Files\\Basler\\pylon 5\\Runtime\\x64')

# if hasattr(os, 'add_dll_directory'):
#     for item in os.environ['PATH'].split(os.pathsep):
#         item=item.strip()
#         if item and os.path.isdir(item):
#             os.add_dll_directory(item)