# Running PySpark Locally with Git (Bitbucket) — No Docker, No __main__

## 1. End-to-End Flow

Bitbucket Repo  
↓ git clone  
Local Machine (Python + Java + Spark)  
↓  
Run PySpark Job (spark-submit)

---

## 2. Prerequisites

### Python
```
python --version
```

### Java
```
java -version
```

Install if needed:
```
sudo apt install openjdk-11-jdk
brew install openjdk@11
```

### PySpark
```
pip install pyspark
```

---

## 3. Clone Repository

```
git clone https://bitbucket.org/<workspace>/<repo>.git
cd <repo>
```

---

## 4. Project Structure

```
pyspark-project/
├── requirements.txt
└── jobs/
    └── main.py
```

---

## 5. PySpark Script (No __main__)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("LocalNoDockerRun") \
    .getOrCreate()

data = [("A", 10), ("B", 20)]
df = spark.createDataFrame(data, ["id", "value"])

df.show()

spark.stop()
```

---

## 6. Install Dependencies

```
pip install -r requirements.txt
```

requirements.txt:
```
pyspark
```

---

## 7. Run the Job

### Option A
```
python jobs/main.py
```
spark-submit --master local[*] jobs/job_customers.py $ Local run
### Option B (Recommended)
```
spark-submit jobs/main.py
```

---

## 8. Multi-Job Setup

```
jobs/
├── job_customers.py
├── job_transactions.py
```

Run:
```
spark-submit jobs/job_transactions.py
```

---

## 9. Virtual Environment

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 10. Common Issues

| Problem | Fix |
|--------|-----|
| JAVA_HOME not set | export JAVA_HOME |
| Spark not found | set SPARK_HOME |
| Memory issues | increase driver memory |
| Slow performance | expected in local mode |

---

## 11. Clean Workflow

```
git clone <repo>
cd <repo>

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

spark-submit jobs/main.py
```

---

## 12. Key Insight

- Git = version control  
- Local machine = execution  
- No Docker required  
- No __main__ required  

---

## Bottom Line

You only need:

- Git  
- Python  
- Java  
- PySpark  
- spark-submit
