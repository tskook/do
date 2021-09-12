import time
import subprocess
def main(): # Print banner
    print("~~~ DebugMe Ver. 1.0~~~\n\nThis program has a lot of bugs.\n Let's see if you can find Some") 
    # Get input from the user name = inpt("Enter your name ==> ")
    age = input("Enter your age ==> ") 
    color = input("What is you favorite color (Enter num)?\n\n1) Red\n2) Green\n3) Blue\n ==> ") 
    # Print age conclusion if age > "13": 
    print("Nice, you already had bar mitzva!") # If age is even if age % 2 == 0: 
    print('Your age is even!') # Check if user is in the army 
    if age == 18 or age == 19:
        print('Wow! you are going to the army..') 
        personal_num = input('What is you personal number? ') 
    else:
        print("Not in the army.. too bad!") 
    if age == 16: print("Sweet sixteen!") # Print favorite color name 
    if color == 1: print('Your favorite color is Red') 
    elif color == 2: print('Your favorite color is Green') 
    elif color == 3: print('Your favorite color is Blue') 
    print('Your Army\'s personal number is: {}'.format(personal_num)) 
    print('Current time: {}'.format(time.ctime())) 
    cmd = input("Enter command to run ==> ") 
    print(subprocess.check_output("{}").format(cmd)) 
    if __name__ == '__main__':
        main()