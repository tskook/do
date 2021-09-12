import os
path = 'C:\\Users\\Tom\\Desktop\\POOK\\projects\\Py\\red_panda_set\\redpanda'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, str(index).join(['redpanda.', '.jpg'])))