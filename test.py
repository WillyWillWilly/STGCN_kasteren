import pandas as pd
from tabulate import tabulate
df = pd.read_csv(r"C:\Users\a0665\project\base_kasteren-m.csv", sep=" ", header=None)  # 手動指定用空格分隔
df[0] = df[0] + " " + df[1]  # 合併日期與時間
df = df.drop(columns=[1])  # 刪除原本的時間欄位
df.columns = ["Timestamp", "Sensor", "Location", "Status", "Activity"] #自訂欄位名稱

aligned_table = tabulate(df, headers='keys', tablefmt='psql', showindex=False)

with open('aligned_kasteren-m.csv', 'w') as f:
    f.write(aligned_table)