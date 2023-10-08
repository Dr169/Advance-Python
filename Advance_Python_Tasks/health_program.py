class School:
    def __init__(self,number_of_students):
        self.quantity = number_of_students

    def get_age(self,ages):
        
        for index,_ in enumerate(ages):
            ages[index] = int(ages[index])
            
        return sum(ages)/self.quantity
    
    def get_height(self,heights):
        
        for index,_ in enumerate(heights):
            heights[index] = int(heights[index]) 
                   
        return sum(heights)/self.quantity
    
    def get_weights(self,weights):
        
        for index,_ in enumerate(weights):
            weights[index] = int(weights[index])   
                 
        return sum(weights)/self.quantity
    
    
school_classes = {}

for class_ in ["A","B"]:
    
    students = School(float(input('Enter number of students : ')))
    
    school_classes[class_] = {
        "age": students.get_age(input(f"Enter ages of class {class_} : ").split()),
        "height":students.get_height(input(f"Enter heights of class {class_} : ").split()),
        "weight":students.get_weights(input(f"Enter weights of class {class_} : ").split())
        }
    
mean_weight_of_classes = []

for class_ in school_classes:
    print(school_classes[class_]["age"])
    print(school_classes[class_]["height"])
    print(school_classes[class_]["weight"])

    mean_weight_of_classes.append((class_, school_classes[class_]["weight"]))
    
if mean_weight_of_classes[0][1] == mean_weight_of_classes[1][1]:
    print("Same")
    
else:
    max_mean_weight_of_classes = max(mean_weight_of_classes[0][1],mean_weight_of_classes[1][1])
    
    if max_mean_weight_of_classes in mean_weight_of_classes[1]:
        print(mean_weight_of_classes[1][0])
    else:
        print(mean_weight_of_classes[0][0])