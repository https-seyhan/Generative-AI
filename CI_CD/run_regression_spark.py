
from pyspark.sql import SparkSession
import json

spark = SparkSession.builder.appName("LLMRegression").getOrCreate()

df = spark.read.json("data/golden_dataset/v1/classification.json")

def dummy_eval(row):
    return (row['id'], 0.9, True)

rdd = df.rdd.map(dummy_eval)

result_df = rdd.toDF(["id", "score", "pass"])
result_df.write.mode("overwrite").parquet("outputs/spark_results")

spark.stop()
