from pyspark.sql import SparkSession
import yaml
from pipeline.mapping import apply_mapping

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    spark = SparkSession.builder.appName("pipeline").getOrCreate()

    df = spark.read.csv("data/input.csv", header=True)

    mapping = load_yaml("configs/mapping.yml")

    df = apply_mapping(df, mapping)

    df.write.mode("overwrite").parquet("output/")

    spark.stop()

if __name__ == "__main__":
    main()
