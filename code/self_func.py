import requests
import pandas as pd
from yfinance import download
import MySQLdb
from sqlalchemy import create_engine

class Datasets:
    global url
    url = 'https://nstatdb.dgbas.gov.tw/dgbasall/webMain.aspx?sdmx/'
    def __init__(self, api_key):
        self.api_url = url+api_key
        self.response = requests.get(self.api_url)
        self.data_json = self.response.json()
    def get_data(self):
        try:
            lst_indexs = list(self.data_json['data']['dataSets'][0]['series'])
            lst_datasets = []
            cols = []
            for i in lst_indexs:
                lst_datasets.append(self.data_json['data']['dataSets'][0]['series'][i]['observations'].values())
            df = pd.DataFrame(lst_datasets[0]).T
            for i in range(1,len(lst_datasets)):
                df_dataset = pd.DataFrame(lst_datasets[i]).T
                df = pd.concat([df,df_dataset], join="outer")
            for i in range(len(lst_datasets)):
                cols.append(self.data_json['data']['structure']['dimensions']['series'][-1]['values'][i]['name'])
            df = df.T
            df.columns = cols
        except:
            lst_indexs = list(self.data_json['data']['dataSets'][0]['series'])
            lst_datasets = []
            cols = []
            for i in lst_indexs:
                lst_datasets.append(self.data_json['data']['dataSets'][0]['series'][i]['observations'].values())
            df = pd.DataFrame(lst_datasets[0]).T
            for i in range(1,len(lst_datasets)):
                df_dataset = pd.DataFrame(lst_datasets[i]).T
                df = pd.concat([df,df_dataset], join="outer")
            for i in range(len(lst_datasets)):
                cols.append(self.data_json['data']['structure']['dimensions']['series'][0]['values'][i]['name'])
            df = df.T
            df.columns = cols
        finally:
            return df      
    def get_time(self):
        year = []
        mon = []
        for i in self.data_json['data']['structure']['dimensions']['observation'][0]['values']:
            year.append(i['id'].split("-")[0])
            mon.append(i['id'].split("-")[1])
        return pd.DataFrame([year, mon], index=['year', 'mon']).T
    def get_timezone(self):
        time = []
        for i in self.data_json['data']['structure']['dimensions']['observation'][0]['values']:
            time.append(i['id'])
        return time
    
class Stockassests:
    def __init__(self, id_number, start_date, end_date):
        self.id_number = id_number
        self.start_data = start_date
        self.end_date = end_date
    def get_data(self):
        data = download(self.id_number, self.start_data, self.end_date)
        data = data['Adj Close']
        data.columns = [self.id_number]
        return data

def connect_Mysql(host="localhost", user="root", password='Kimg31820453', 
                  db_name='macro', port=3306, charset="utf8"):
    try:
        conn =  MySQLdb.connect(host=host,    
                                user=user,        
                                password=password,
                                database = db_name,
                                port=port,           
                                charset=charset)    
        return conn
    except MySQLdb.Error as err:
        print('Failed to connecting:', err)
        
def get_engine(host="localhost", user="root", password='Kimg31820453', db_name='macro'):
    
    user = user
    password = password
    host = host
    database = db_name
    
    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')
    
    return engine

    
    
    
    
    
    