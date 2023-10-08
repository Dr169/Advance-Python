import datetime

class Age:
    
    def __init__(self,date_list):
        year, month, day = date_list
        self.year = int(year)
        self.day = int(day)
        self.month =int(month)
        
        
    def check_date(self):
        try:
            datetime.datetime(self.year,self.month,self.day)
            Age.calculate_age(self)
            
        except Exception as e:
            print("WRONG", e)
        
            
    def calculate_age(self):
        date_now = datetime.datetime.now()
        
        if self.month > date_now.month:
            self.year -= 1
            
        else:
            print(date_now.year - self.year)

    
date = Age(input('Enter your birthday : ').split("/"))

date.check_date()