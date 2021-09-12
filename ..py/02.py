username = input("Please enter your username => ")
password = input("Please enter your password => ")
if username.lower() != "admin" or username.lower() != "root":
    print("username incorrect")
    exit()
if password.lower() != "default":
    print("password")
    exit()


while True:
    op = input("welcome choose an option: ")
    if op == '1':
        temp_password = input("please enter a new password--> ")
        if temp_password < 6 or not temp_password[-1].isnumeric() or not temp_password[-2].isnumeric():
            print("invalid password")
            temp_password = ''
        else:
            password = temp_password
            print('password changed!')
    if op == '2':
        num = input("please enter hoger number: ")
        if num
