import mysql.connector
import re
import sqlalchemy as sql
import pymysql


pymysql.install_as_MySQLdb()
engine = sql.create_engine("mysql://root:dani81@localhost/employee_info")
connection = engine.connect()
metadata = sql.MetaData()


## Create a Table for save information about users ##
users = sql.Table(
    'users', 
    metadata, 
    sql.Column('username', sql.String(255)), 
    sql.Column('password', sql.String(255)),
    )
metadata.create_all(engine)


def validate_email(email):  
    if re.match(r"[^@]+@[^@]+\.[^@]+", email):  
        return True
    
    return False 


def validate_password(password):
    if re.search(r"[A-Za-z]", password) and re.search(r"\d", password):
       return True
   
    return False


while True:
    user_name = input("Enter your username: ")
    password = input("Enter your password: ")

    if validate_email(user_name) and validate_password(password):
        connection.execute(users.insert(), {"username":user_name, "password":password})

        print("1 record inserted.")
        break
    
    else:
        print("Invalid email address or password")
        continue
    