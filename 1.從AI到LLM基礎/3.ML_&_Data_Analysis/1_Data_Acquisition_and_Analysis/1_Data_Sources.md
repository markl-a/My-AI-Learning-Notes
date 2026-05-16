# 資料來源指南 (Data Sources)

在機器學習與資料分析專案中，資料是整個流程的基礎。無論是從本地檔案、資料庫、API 服務、網頁爬取，或是來自大資料平台與串流系統，都必須了解如何擷取、清理與整合。此文件將詳細介紹常見資料來源類型、取得方式、程式範例與實務建議。

## 目錄
1. [檔案類資料](#1-檔案類資料)
2. [SQL資料庫](#2-sql資料庫)
3. [NoSQL資料庫](#3-nosql資料庫)
4. [API資料獲取](#4-api資料獲取)
5. [網頁爬蟲](#5-網頁爬蟲)
6. [開放資料集](#6-開放資料集)
7. [串流資料](#7-串流資料)
8. [其他資料來源](#8-其他資料來源)

---

## 1. 檔案類資料

### 1.1 CSV/TSV檔案

**特性與適用場景**：  
- CSV 是最常見的純文字表格資料格式，易於讀寫、通用性高、利於不同語言或工具交換資料。  
- 適用於中小規模資料集、初次實作練習、資料分享與開源資料集發布。

**範例程式碼 (Python - pandas)**：
```python
import pandas as pd

# 讀取CSV
df = pd.read_csv('data.csv', encoding='utf-8')

# 自定義參數：處理缺失值、解析日期、讀取特定欄位、限制行數
df = pd.read_csv('data.csv',
    encoding='utf-8',
    na_values=['NA', 'missing'],
    parse_dates=['date'],
    usecols=['col1', 'col2'],
    nrows=1000
)
```

### 1.2 Excel檔案
```python
import pandas as pd

# 讀取單一工作表
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 讀取所有工作表 (返回字典)
excel_file = pd.ExcelFile('data.xlsx')
df_dict = pd.read_excel(excel_file, sheet_name=None)
```

### 1.3 JSON檔案
```python
import pandas as pd

df = pd.read_json('data.json')  # 單純結構化 JSON

# 若是巢狀結構，可先用原生 json.load 再自行展開
import json
with open('data.json', 'r') as f:
    data = json.load(f)
# 可用 pd.json_normalize() 展開巢狀欄位
```

### 1.4 XML檔案
```python
import xml.etree.ElementTree as ET

tree = ET.parse('data.xml')
root = tree.getroot()
# 可自行解析節點、屬性來組裝成 DataFrame
```

**優缺點**：  
- **優點**：易於使用、通用性高、適合中小型資料。  
- **缺點**：對極大資料集效率不足、對結構複雜度有限制（除 JSON 外，CSV、Excel 不適合巢狀資料）。  

---

## 2. SQL資料庫

**常見資料庫**：MySQL、PostgreSQL、SQLite、SQL Server、Oracle Database。

**特性**：  
- 結構化資料、使用 SQL 語言查詢。
- 支援 ACID 特性，確保資料一致性與可靠性。
- 適用於需要嚴謹結構與複雜關聯查詢的應用。

**範例 (PostgreSQL + pandas)**：
```python
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost:5432/db_name')

df = pd.read_sql('SELECT * FROM table_name', engine)
```

**優缺點**：  
- **優點**：結構化、強大查詢能力、有交易安全與一致性。  
- **缺點**：較不適合非結構化資料、需要資料庫管理知識。

---

## 3. NoSQL資料庫

**類型與範例**：  
- Document Store (MongoDB)、Key-Value Store (Redis)、Column-Family (Cassandra)、Graph (Neo4j)
- 適合非結構化或半結構化資料、高可擴展性、水平擴展支援好。

**範例 (MongoDB + Python)**：
```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['database_name']
collection = db['collection_name']

data = collection.find({"age": {"$gt": 25}})
df = pd.DataFrame(list(data))
```

**優缺點**：  
- **優點**：彈性結構、高可用性、高可擴展性。  
- **缺點**：缺乏強一致性、有時需應用程式層面控制資料模式和一致性。

---

## 4. API資料獲取

**RESTful, GraphQL, SOAP** 等多種 API 類型，讓應用程式存取外部服務或資料。

**範例 (RESTful API + requests)**：
```python
import requests

response = requests.get('https://api.example.com/data',
    headers={'Authorization': 'Bearer token'})
if response.status_code == 200:
    data = response.json()
    df = pd.json_normalize(data)
else:
    print(f"Request failed: {response.status_code}")
```

**優缺點**：  
- **優點**：可動態取得資料、整合多元服務。  
- **缺點**：受 API 限制、需考量驗證、速率限制、錯誤處理。

---

## 5. 網頁爬蟲

**原理**：透過 HTTP 請求取得網頁 HTML，再使用解析器 (BeautifulSoup、lxml) 抽取目標資料。  
**注意事項**：遵守 robots.txt、不要過度頻繁請求、法律與版權合規。

**範例 (BeautifulSoup)**：
```python
import requests
from bs4 import BeautifulSoup

response = requests.get('https://example.com')
soup = BeautifulSoup(response.text, 'html.parser')

items = soup.find_all('div', class_='target-class')
for item in items:
    print(item.get_text())
```

**優缺點**：  
- **優點**：可從無 API 的網站取得資料。  
- **缺點**：網頁結構易變動、需額外處理 HTML, JS, CSS、法律與道德考量。

---

## 6. 開放資料集

**平台與來源**：  
- Kaggle Datasets、UCI ML Repository、Google Dataset Search
- 政府開放資料平台 (data.gov, data.gov.tw)
- 常用於教學、實驗、快速 PoC

**特性**：  
- 資料一般經過整理，方便使用。  
- 可能需要與專案需求再行清洗與轉換。

---

## 7. 串流資料

**工具**：  
- Kafka、Kinesis、Pub/Sub  
- 適用於需要即時分析的場景（IoT、金融交易、社交媒體即時事件）

**範例 (Kafka + Python)**：
```python
from kafka import KafkaConsumer

consumer = KafkaConsumer('topic_name', bootstrap_servers=['localhost:9092'])
for message in consumer:
    print(message.value)
```

**優缺點**：  
- **優點**：即時性強、可用於實時預測和事件驅動應用。  
- **缺點**：需建立串流處理架構與監控、對模型與資料處理有即時性要求。

---

## 8. 其他資料來源

### 8.1 雲端儲存
- AWS S3, Google Cloud Storage (GCS) 等可透過 Boto3、GCP Python Client 呼叫。
  
**範例 (AWS S3 + pandas)**：
```python
import boto3
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='bucket-name', Key='file.csv')
df = pd.read_csv(obj['Body'])
```

### 8.2 分散式檔案系統與 Data Lake
- HDFS、Azure Data Lake Store、使用 Spark `spark.read.csv()`、`spark.read.parquet()` 輕鬆讀取。

### 8.3 FTP/SFTP
```python
from ftplib import FTP

ftp = FTP('ftp.example.com')
ftp.login(user='username', passwd='password')
with open('file.csv', 'wb') as fp:
    ftp.retrbinary('RETR file.csv', fp.write)
```

---

## 資料品質與最佳實務

1. **資料品質檢查**：  
   - 計算統計量、檢查缺失值、重複值、資料型別正確性  
   - 確認資料分佈是否符合預期

2. **安全與合規**：  
   - 對敏感資訊（PII）進行脫敏或加密  
   - 遵守 GDPR、CCPA 等法規  
   - 確保 API 密鑰、密碼不硬編碼於程式碼中

3. **效能優化**：  
   - 大型資料採用分批讀取或串流處理  
   - 適當使用快取或中間結果儲存  
   - 使用適合的大資料工具（Spark, Dask）分散處理

4. **錯誤處理與重試機制**：
```python
try:
    # 資料取得程式碼
    pass
except Exception as e:
    print(f"資料獲取錯誤: {str(e)}")
    # 根據需要實作重試或告警
```

5. **版本控制與日誌**：  
   - 為資料集版本編號，便於回溯  
   - 記錄取得資料的時間戳、來源、條件參數  
   - 利用 git 或 DVC 等工具管理資料版本

---

## 總結

面對多元的資料來源，需要根據專案需求、資料特性、規模與即時性要求，選擇最適合的取得方式與工具。在開始實際分析或建模前，花時間確保資料品質、連接安全、紀錄清晰，將能大幅提升後續流程的效率與可靠性。

後續可參考本專案中的其他 Notebook，如 `Data_Cleaning_and_Processing.ipynb`、`Exploratory_Data_Analysis.ipynb`，了解如何將從上述來源取得的資料進行清洗、探索與特徵工程，加速資料分析和機器學習任務的進展。