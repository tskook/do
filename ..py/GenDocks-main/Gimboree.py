print("Welcome to the Gymboree system")
while True:
    try:
        num_of_soldiers = int(input("enter number of soldiers: "))
        break
    except:
        print("enter a number!")
        continue
sum = 0
for s in range(0, num_of_soldiers):
    first_name = input("please enter soldier no. " + str(s + 1) + "'s first name")
    last_name = input("please enter soldier no. " + str(s + 1) + "'s last name")
    months = int(input("please enter soldier no. " + str(s + 1) + "'s months in the army"))
    if months < 20:
        salary = months * 800
    elif months >= 20 and months <= 30:
        salary = months * 1000
    elif months > 30:
        salary = months * 1500
    initals = str(first_name[0].upper() + "." + last_name[0].upper())
    print(initals + " - " + str(salary))
    sum += salary
print("Sum Of Salarys: " + str(sum))
print("Salarys average: " + str(float(sum) / float(num_of_soldiers)))
if len(str(sum)) > 6:
    print("Too Much Money!")