import requests
from bs4 import BeautifulSoup as bs
import sqlalchemy as sql
import pymysql
from sklearn import tree

list_of_all_ads = []

for page in range(1, 100):
    if page % 10 == 0:
        print(page)
        
    page = requests.get(f"https://www.truecar.com/used-cars-for-sale/listings/?page={page}")
    content = bs(page.content, "html.parser")
    list_of_ads = content.find_all("div", class_="card-content order-3 vehicle-card-body")

    for ad in list_of_ads:
        
        name = ad.find('span', class_='truncate').get_text()
        
        year = int(ad.find('span', class_='vehicle-card-year text-xs').get_text())
        
        list_of_price = ad.find('div', class_='vehicle-card-bottom-pricing-secondary pl-3 lg:pl-2 vehicle-card-bottom-max-50').get_text('-').split('-')
        price_after_off = int((list_of_price[1] if len(list_of_price) == 2 else list_of_price[0]).replace(",","").replace("$",""))
        
        amortization = int(ad.find('div', class_='flex w-full justify-between').get_text("-").split('-')[0].replace(",",""))
        
        accidents = ad.find('div', {"data-test":"vehicleCardCondition"}).get_text().split(",")[0]
        
        if accidents == "No accidents reported":
            accidents = 0
        else:
            accidents = 1
            
        list_of_all_ads.append({
            "name": name, 
            "year": year, 
            "price": price_after_off, 
            "amortization": amortization,
            "accidents": accidents
            })
        
        
        
pymysql.install_as_MySQLdb()

engine = sql.create_engine("mysql://root:dani81@localhost/test")
connection = engine.connect()
metadata = sql.MetaData()

cars_info = sql.Table(
    'cars_info', 
    metadata, 
    sql.Column('name', sql.String(255)), 
    sql.Column('yaer', sql.Integer()), 
    sql.Column('price', sql.Integer()), 
    sql.Column('amortization', sql.Integer()), 
    sql.Column('accidents', sql.Integer()),
    )

metadata.create_all(engine)

for ad in list_of_all_ads:
    connection.execute(cars_info.insert(), ad)

query = cars_info.select()
result = connection.execute(query)

info = []
for name, year, price, amortization, accidents in result:
    info.append({"name": name, "year": year, "price": price, "amortization": amortization, "accidents": accidents})
    
    
x_train = []
y_train = []

for row in info[:-5]:
    x_train.append([row["year"], row["price"], row["amortization"], row["accidents"]])
    y_train.append(row["name"])
    
x_test= []
y_test = []

for row in info[-5:]:
    x_test.append([row["year"], row["price"], row["amortization"], row["accidents"]])
    y_test.append(row["name"])

model = tree.DecisionTreeClassifier()
model = model.fit(x_train, y_train)

print(y_test)
print(model.predict(x_test))