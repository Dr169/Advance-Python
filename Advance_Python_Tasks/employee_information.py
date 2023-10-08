import sqlalchemy as sql
import pymysql


pymysql.install_as_MySQLdb()

engine = sql.create_engine("mysql://root:dani81@localhost/employee_info")
connection = engine.connect()
metadata = sql.MetaData()

information = sql.Table(
    'information', 
    metadata, 
    sql.Column('Name', sql.String(255)), 
    sql.Column('Weight', sql.Integer()), 
    sql.Column('Height', sql.Integer()),
    )

metadata.create_all(engine)


## Write data in database ##
list_of_name = ["Amin","Mahdi","Mohammad","Ahmad"]
list_of_weight = [75,90,75,60]
list_of_height = [180,190,175,175]

for index in range(len(list_of_name)):
    connection.execute(information.insert(), 
                       {"Name":list_of_name[index],
                        "Weight":list_of_weight[index],
                        "Height":list_of_height[index],
                        })
    
## Read data from database ##
info = {}

query = information.select()
result = connection.execute(query)

for index, (name, weight, height) in enumerate(result):
    info[index] = {"Name":name,"Weight":weight,"Height":height}

info_items = sorted(info.items(), key = lambda item:(-(item[1]["Height"]-100),(item[1]["Weight"]),item[1]["Name"]))

for item in info_items:
    print(f"{item[1]['Name']}  {item[1]['Height']}  {item[1]['Weight']}")