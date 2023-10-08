number_of_participater = int(input("Enter the number of participater : "))
female_list = []
male_list = []

for participater in range(number_of_participater):
    
    participater = input("Enter information of participater : ").split(".")
    participater[1] = participater[1].lower().capitalize()
    
    if participater[0] == "f":
        female_list.append(participater)
        
    elif participater[0] == "m":
        male_list.append(participater)
        
female_list = sorted(female_list, key=lambda item : (item[1]))
male_list = sorted(male_list, key=lambda item : (item[1]))

ordered_list = [female_list, male_list]

for gender in ordered_list:
    for info in gender:
        print(" ".join(info))