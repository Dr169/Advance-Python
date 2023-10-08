list_of_matchs = [
    [("Iran","Spain")],
    [("Iran","Portugal")],
    [("Iran","Morocco")],
    [("Spain","Portugal")],
    [("Spain","Morocco")],
    [("Portugal","Morocco")],
]

dict_of_teams = {
    "Iran":{'wins':0 , 'loses':0 , 'draws':0 , 'goal_difference':0 , 'points':0},
    "Spain":{'wins':0 , 'loses':0 , 'draws':0 , 'goal_difference':0 , 'points':0},
    "Portugal":{'wins':0 , 'loses':0 , 'draws':0 , 'goal_difference':0 , 'points':0},
    "Morocco":{'wins':0 , 'loses':0 , 'draws':0 , 'goal_difference':0 , 'points':0},
    }
    
    
for match in range(len(list_of_matchs)):
    
    list_of_matchs[match].append(tuple(input(f"{list_of_matchs[match][0][0]} - {list_of_matchs[match][0][1]} : ").split("-")))
    
    if list_of_matchs[match][1][0] > list_of_matchs[match][1][1]:
        dict_of_teams[list_of_matchs[match][0][0]]["wins"] += 1
        dict_of_teams[list_of_matchs[match][0][1]]["loses"] += 1
        dict_of_teams[list_of_matchs[match][0][0]]["goal_difference"] += (int(list_of_matchs[match][1][0])-int(list_of_matchs[match][1][1]))
        dict_of_teams[list_of_matchs[match][0][1]]["goal_difference"] -= (int(list_of_matchs[match][1][0])-int(list_of_matchs[match][1][1]))
        dict_of_teams[list_of_matchs[match][0][0]]["points"] += 3
        
    elif list_of_matchs[match][1][0] < list_of_matchs[match][1][1]:
        dict_of_teams[list_of_matchs[match][0][1]]["wins"] += 1
        dict_of_teams[list_of_matchs[match][0][0]]["loses"] += 1
        dict_of_teams[list_of_matchs[match][0][1]]["goal_difference"] += (int(list_of_matchs[match][1][1])-int(list_of_matchs[match][1][0]))
        dict_of_teams[list_of_matchs[match][0][0]]["goal_difference"] -= (int(list_of_matchs[match][1][1])-int(list_of_matchs[match][1][0]))
        dict_of_teams[list_of_matchs[match][0][1]]["points"] += 3
        
    else :
        dict_of_teams[list_of_matchs[match][0][0]]["draws"] += 1
        dict_of_teams[list_of_matchs[match][0][1]]["draws"] += 1
        dict_of_teams[list_of_matchs[match][0][0]]["points"] += 1
        dict_of_teams[list_of_matchs[match][0][1]]["points"] += 1


items = sorted(dict_of_teams.items(), key = lambda item:(-(item[1]["points"]-100),-(item[1]["goal_difference"]-100),item[0][0]))

for i in items:
    print(f"{i[0]}  wins:{i[1]['wins']} , loses:{i[1]['loses']} , draws:{i[1]['draws']} , goal difference:{i[1]['goal_difference']} , points:{i[1]['points']}")