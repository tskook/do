import os

folder_name = input("enter the whitning folder name: ")
os.mkdir(f'C:\\Users\\Tom\\{folder_name}')
print(os.path.exists(f'C:\\Users\\Tom\\{folder_name}'))

f_list = ['Pdf', 'Pictures', 'Videos', 'Documents', 'Links', 'IPs', 'Infected', 'Cleaned', 'E-Mails', 'Logs']

for f in f_list:
    os.mkdir(f'C:\\Users\\Tom\\{folder_name}\\{f}')

folders_to_delete = input('what folders do you want to delete?')
if not folders_to_delete.isspace():
    for folder in folders_to_delete.split(' '):
        try:
            os.removedirs(f'C:\\Users\\Tom\\{folder_name}\\{folder}')
        except:
            pass
