from pyspark.sql import functions as F

def apply_mapping(df, mapping):

    for col_map in mapping["columns"]:
        df = df.withColumnRenamed(col_map["source"], col_map["target"])

    for col_map in mapping["columns"]:
        if "type" in col_map:
            df = df.withColumn(
                col_map["target"],
                F.col(col_map["target"]).cast(col_map["type"])
            )

    for t in mapping.get("transformations", []):
        if t["operation"] == "multiply":
            df = df.withColumn(
                t["column"],
                F.col(t["column"]) * t["value"]
            )

    for f in mapping.get("filters", []):
        df = df.filter(f"{f['column']} {f['condition']}")

    return df
