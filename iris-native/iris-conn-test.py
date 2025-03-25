import iris
import time
import os

username = 'demo'
password = 'demo'
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972' 
namespace = 'USER'
CONNECTION_STRING = f"{hostname}:{port}/{namespace}"

print(CONNECTION_STRING)

# Note: Ideally conn and cursor should be used with context manager or with try-execpt-finally 
conn = iris.connect(CONNECTION_STRING, username, password)
print('connected!')
cursor = conn.cursor()

sql = 'select current_timestamp, ?'
params = ['Hello!']
cursor.execute(sql, params)

# Fetch all results
results = cursor.fetchall()
for row in results:
    print(row)

conn.close()
print('connection closed')