
def qq():
    count = 0.0
    sum = 0.0
    x = input('enter number or q to quit')
    while x != 'q':
        x = input('enter number or q to quit')
        count = count + 1
        sum += float(x)
    print(sum / count)

def palindrom():
    w = input()
    for i in w:
        if i != w[-w.index(i) - 1]:
            print("not palindrom")
            return
    print("palindrom")

def first():
    num = int(input('enter '))
    if 2**num + 1 == num * 2 + 1:
        print('curazun number')
    else:
        print('not curazun number')

def mintosec():
    num = int(input())
    print('min in sec: ' + str(num*60))

def maxedge():
    firstedge = int(input('enter 1st edge'))
    secondedge = int(input('enter 2nd edge'))
    print("max 3rd edge: " + str(firstedge + secondedge - 1))

def luke():
    p = input()
    if p.lower == 'darth vader':
        print('father')
    if p.lower == 'leah':
        print('sister')
    if p.lower == 'r2d2':
        print('droid')
    if p.lower == 'han':
        print('brother in law')
    else:
        print('unknown')

def repeat():
    w = input('enter word')
    for l in w:
        print(l, end=l)

def vowels():
    w = input('enter word')
    c = 0
    for i in w:
        if i == 'o' or i == 'a' or i == 'e' or i == 'u' or i == 'i':
            c = c + 1
    print(c)

def atm():
    pin = input('enter pin')
    if len(pin) != 4 or len(pin) != 6 or not pin.isnumeric():
        print('not valid')
    else:
        print('valid')

def ldic():
    d = {"darth vader":"father", "leah":"sister"}
    p = input()
    try:
        print(d[p])
    except:
        print('none')
