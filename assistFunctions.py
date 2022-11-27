import time
def escort(x, y):
    if y > 160:
        if x < 190:
            print("⇙")
        elif x > 210:
            print("⇘")
        else:
            print("⇓")
    elif y < 140:
        if x < 190:
            print("⇖")
        elif x > 210:
            print("⇗")
        else:
            print("⇑")
    else:
        if x < 190:
            print("⟸")
        elif x > 210:
            print("⟹")
        else:
            print("Обнаружен человек! 🞪")
            return True

def shoot():
    print("Выстрел")
    time.sleep(1)