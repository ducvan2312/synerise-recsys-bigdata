
from pyspark.sql import SparkSession
import os

# ==========================================
# 1. Cau hinh
# ==========================================

PROJECT_DIR = "/home/phungducvan/recsys_project"

RAW_DIR = PROJECT_DIR + "/data/raw/synerise"
EXPORT_DIR = PROJECT_DIR + "/data/staging/flume_export"

TABLES = [
    "page_visit",
    "add_to_cart",
    "product_buy",
    "remove_from_cart",
    "search_query",
    "product_properties"
]

# ==========================================
# 2. Tao SPARK SESSION
# ==========================================

spark = (
    SparkSession.builder
    .appName("Synerise-Parquet-to-JSONL")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("SYNERISE PARQUET TO JSONL")
print("=" * 60)

# ==========================================
# 3. Chuyen doi tung bang
# ==========================================

for table in TABLES:

    input_path = "file://" + RAW_DIR + "/" + table + ".parquet"
    output_path = "file://" + EXPORT_DIR + "/" + table

    print(f"\nProcessing table: {table}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    df = spark.read.parquet(input_path)

    print("Schema:")
    df.printSchema()

    # Export as JSON Lines
    # Spark writes partitioned output files.
    df.write \
        .mode("overwrite") \
        .json(output_path)

    print(f"Export completed: {table}")

print("\nAll tables exported successfully.")

spark.stop()
