import psycopg2
import pandas as pd

config = {
    'host': 'your_localhost',
    'port': 'your_port',
    'database': 'your_database',
    'user': 'your_user',
    'password': 'your_password'
}

conn = psycopg2.connect(**config)

cursor = conn.cursor()

cursor.execute(query="your_query")

data = cursor.fetchall()
columns = [name.name for name in cursor.description]

dataframe = pd.DataFrame(data=data, columns=columns)