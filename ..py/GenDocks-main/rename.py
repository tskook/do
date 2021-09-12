import os
path = r'C:\Users\u9133908\Documents\GitHub\do\..py\GenDocks-main\redpanda'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, str(index).join(['redpanda.', '.jpg'])))
