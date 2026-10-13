from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, trim
from pyspark.sql.types import LongType, StringType


# ============================================================
# CONFIG
# ============================================================

BASE_RAW = "hdfs:///data/synerise_recsys/raw"
BASE_PROCESSED = "hdfs:///data/synerise_recsys/processed"
TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"

SHOW_SAMPLE = True
SAMPLE_ROWS = 5


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Synerise-Data-Cleaning")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# COMMON EVENT CLEANING
# ============================================================

def clean_event_dataset(
    dataset_name,
    required_columns,
    id_columns,
    dedup_columns=None
):
    """
    Clean event data and save it as Parquet.

    dedup_columns=None means no deduplication.
    """

    input_path = f"{BASE_RAW}/{dataset_name}"
    output_path = f"{BASE_PROCESSED}/{dataset_name}"

    print("\n" + "=" * 70)
    print(f"CLEANING DATASET: {dataset_name}")
    print("=" * 70)

    # Read RAW JSON from HDFS
    df = spark.read.json(input_path)

    print("\nOriginal schema:")
    df.printSchema()

    # Remove records missing required fields
    for column_name in required_columns:
        df = df.filter(col(column_name).isNotNull())

    # Convert ID columns to LongType
    for column_name in id_columns:
        df = df.withColumn(
            column_name,
            col(column_name).cast(LongType())
        )
        df = df.filter(col(column_name).isNotNull())

    # Convert timestamp and remove invalid values
    df = df.withColumn(
        "timestamp",
        to_timestamp(
            col("timestamp"),
            TIMESTAMP_FORMAT
        )
    )

    df = df.filter(col("timestamp").isNotNull())

    # Remove duplicates only when required
    if dedup_columns:
        df = df.dropDuplicates(dedup_columns)

    print("\nCleaned schema:")
    df.printSchema()

    if SHOW_SAMPLE:
        print(f"\nSample ({SAMPLE_ROWS} records):")
        df.show(SAMPLE_ROWS, truncate=False)

    # Save cleaned data as Parquet
    df.write \
        .mode("overwrite") \
        .parquet(output_path)

    print("\nSaved to:")
    print(output_path)


# ============================================================
# 1. PAGE VISIT
# ============================================================
# Do not deduplicate page visits because repeated visits can
# represent separate user behaviors.

clean_event_dataset(
    dataset_name="page_visit",
    required_columns=[
        "client_id",
        "url",
        "timestamp"
    ],
    id_columns=[
        "client_id",
        "url"
    ],
    dedup_columns=None
)


# ============================================================
# 2. ADD TO CART
# ============================================================
# Remove duplicate events by user + product + timestamp.

clean_event_dataset(
    dataset_name="add_to_cart",
    required_columns=[
        "client_id",
        "sku",
        "timestamp"
    ],
    id_columns=[
        "client_id",
        "sku"
    ],
    dedup_columns=[
        "client_id",
        "sku",
        "timestamp"
    ]
)


# ============================================================
# 3. PRODUCT BUY
# ============================================================

clean_event_dataset(
    dataset_name="product_buy",
    required_columns=[
        "client_id",
        "sku",
        "timestamp"
    ],
    id_columns=[
        "client_id",
        "sku"
    ],
    dedup_columns=[
        "client_id",
        "sku",
        "timestamp"
    ]
)


# ============================================================
# 4. REMOVE FROM CART
# ============================================================

clean_event_dataset(
    dataset_name="remove_from_cart",
    required_columns=[
        "client_id",
        "sku",
        "timestamp"
    ],
    id_columns=[
        "client_id",
        "sku"
    ],
    dedup_columns=[
        "client_id",
        "sku",
        "timestamp"
    ]
)


# ============================================================
# 5. SEARCH QUERY
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DATASET: search_query")
print("=" * 70)

input_path = f"{BASE_RAW}/search_query"
output_path = f"{BASE_PROCESSED}/search_query"

df = spark.read.json(input_path)

print("\nOriginal schema:")
df.printSchema()

# Required fields
df = df.filter(
    col("client_id").isNotNull()
    & col("query").isNotNull()
    & col("timestamp").isNotNull()
)

# Data types
df = df.withColumn(
    "client_id",
    col("client_id").cast(LongType())
)

df = df.filter(col("client_id").isNotNull())

df = df.withColumn(
    "query",
    trim(col("query").cast(StringType()))
)

# Remove empty search queries
df = df.filter(
    col("query").isNotNull()
    & (col("query") != "")
)

df = df.withColumn(
    "timestamp",
    to_timestamp(
        col("timestamp"),
        TIMESTAMP_FORMAT
    )
)

df = df.filter(col("timestamp").isNotNull())

# Remove duplicate searches
df = df.dropDuplicates([
    "client_id",
    "query",
    "timestamp"
])

print("\nCleaned schema:")
df.printSchema()

if SHOW_SAMPLE:
    print(f"\nSample ({SAMPLE_ROWS} records):")
    df.show(SAMPLE_ROWS, truncate=False)

df.write \
    .mode("overwrite") \
    .parquet(output_path)

print("\nSaved to:")
print(output_path)


# ============================================================
# 6. PRODUCT PROPERTIES
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DATASET: product_properties")
print("=" * 70)

input_path = f"{BASE_RAW}/product_properties"
output_path = f"{BASE_PROCESSED}/product_properties"

df = spark.read.json(input_path)

print("\nOriginal schema:")
df.printSchema()

# SKU is the required product identifier
df = df.filter(
    col("sku").isNotNull()
)

df = df.withColumn(
    "sku",
    col("sku").cast(LongType())
)

df = df.filter(col("sku").isNotNull())

# Product metadata types
df = df.withColumn(
    "category",
    col("category").cast(LongType())
)

df = df.withColumn(
    "price",
    col("price").cast(LongType())
)

df = df.withColumn(
    "name",
    trim(col("name").cast(StringType()))
)

# Keep one metadata record per SKU
df = df.dropDuplicates(["sku"])

print("\nCleaned schema:")
df.printSchema()

if SHOW_SAMPLE:
    print(f"\nSample ({SAMPLE_ROWS} records):")
    df.show(SAMPLE_ROWS, truncate=False)

df.write \
    .mode("overwrite") \
    .parquet(output_path)

print("\nSaved to:")
print(output_path)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("ALL DATA CLEANING COMPLETED")
print("=" * 70)

print("\nProcessed datasets:")
print("  - page_visit")
print("  - add_to_cart")
print("  - product_buy")
print("  - remove_from_cart")
print("  - search_query")
print("  - product_properties")

print(f"\nOutput location: {BASE_PROCESSED}")
print("\nNext step: 02_build_interactions.py")

spark.stop()
