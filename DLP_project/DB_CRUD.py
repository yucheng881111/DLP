import sqlite3

class DB:

    DATABASE_PATH = ''
    #DATABASE_NAME = 'segments_test.db'

    DATABASE_NAME = 'segments.db'
    DATABASE = DATABASE_PATH + DATABASE_NAME
    
    def __init__(self):
        pass

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.DATABASE)
            print ("Database Opened...")
        except Exception as e1:
             print(e1)
        

    def insert(self, sql_sig, data):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_sig, data)
            self.conn.commit()
            return cursor.lastrowid #回復最後一個data的id
        except Exception as e1:
            print(e1)

        
    def update(self, sql, list):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, list)
        except Exception as e1:
            print(e1)   
           
    def select(self, sql_sig):
        data = []
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_sig)
            rows = cursor.fetchall()
        except Exception as e1:
           print(e1)
        return rows


    def close(self):
            try:
                self.conn.close()
                print("Database Close")
            except Exception as e1:
                print(e1)

