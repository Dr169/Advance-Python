import requests
from bs4 import BeautifulSoup
import sqlalchemy as sql
import pymysql


brand = input("Enter car brand : ")
model = input("Enter car model : ")

page = requests.get(f"https://www.truecar.com/used-cars-for-sale/listings/{brand}/{model}/?page=1")

content = BeautifulSoup(page.content, "html.parser")

list_of_all_ad = content.find_all("div", class_="card-content order-3 vehicle-card-body")

dict_of_add = {}

for index, ad in enumerate(list_of_all_ad):
    
    amortization = ad.find('div', class_='flex w-full justify-between').get_text("-").split('-')[:2]
    
    list_of_price = ad.find('div', class_='vehicle-card-bottom-pricing-secondary pl-3 lg:pl-2 vehicle-card-bottom-max-50').get_text('-').split('-')
    price_after_off = (list_of_price[1] if len(list_of_price) == 2 else list_of_price[0])
        
    dict_of_add[index] = {"price": price_after_off, "amortization": "".join(amortization)}


## Save and Reread cars information in database using SQLAlchemy ##

pymysql.install_as_MySQLdb()
engine = sql.create_engine("mysql://root:dani81@localhost/test")
connection = engine.connect()
metadata = sql.MetaData()

## Create a Table for save information about cars ##
cars_information = sql.Table(
    'cars_information', 
    metadata, 
    sql.Column('price', sql.String(50)), 
    sql.Column('amortization', sql.String(50)),
    )

metadata.create_all(engine)

## Write information about cars in database ##
connection.execute(cars_information.insert(), list(dict_of_add.values()))

## Read information anout cars from database ##
query = cars_information.select()
result = connection.execute(query)

for row in result:
   print(row)

