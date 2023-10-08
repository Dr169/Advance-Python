# اسامی بازیکنان : حسین-مازیار-اکبر-نیما-مهدی-فرهاد-محمد-خشایار-میلاد-مصطفی-امین-سعید-پویا-پوریا-رضا-علی-بهزاد-سهیل-بهروز-شهروز-سامان-محسن

from random import sample


class Person:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def get_info(self):
        return f"{self.name} is a person."
    
    
class FotballPlayer(Person):
    def __init__(self, name, team, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.team = team
        
    def get_info(self):
        return f"{self.name} : {self.team}"
    

list_of_players = set(input("Enter players names, separated by '-' : ").split("-"))

team_A =  set(sample(list_of_players, 11))
team_B = list_of_players - team_A

for index, player in enumerate(team_A):
    object_player = FotballPlayer(player, "Team A")
    print(object_player.get_info())
    
for index, player in enumerate(team_B):
    object_player = FotballPlayer(player, "Team B")
    print(object_player.get_info())