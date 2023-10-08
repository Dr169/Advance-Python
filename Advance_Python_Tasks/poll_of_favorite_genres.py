geners = {
    'Horror': 0, 
    'Romance': 0, 
    'Comedy': 0, 
    'History': 0, 
    'Adventure': 0, 
    'Action': 0
    }

votes = int(input("Enter Number of votes : "))

for vote in range(votes):
    
    poll = input().split(' ')
    person, favorite_gener = poll[0], poll[1:]
    
    for gener in favorite_gener:
        geners[gener] += 1
        
items = sorted(geners.items(), key=lambda item: (item[1],(ord(item[0][0])-100)), reverse= True)

for item in items:
    print("%s : %i"% (item[0],item[1]))