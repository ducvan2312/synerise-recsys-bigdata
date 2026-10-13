from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    sum as spark_sum,
    min as spark_min,
    max as spark_max,
    log1p,
    to_timestamp
)
from pyspark.sql.types import LongType, DoubleType

# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Synerise-Build-User-Item-Interactions")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. HDFS INPUT / OUTPUT
# ============================================================

BASE = "hdfs:///data/synerise_recsys/processed"

ADD_TO_CART = BASE + "/add_to_cart"
PRODUCT_BUY = BASE + "/product_buy"

OUTPUT = "hdfs:///data/synerise_recsys/interactions/user_item"


# ============================================================
# 3. READ ADD_TO_CART
# ============================================================

print("\n===== READ ADD_TO_CART =====")

add_to_cart = (
    spark.read.parquet(ADD_TO_CART)
    .select("client_id", "sku", "timestamp")
    .filter(
        col("client_id").isNotNull()
        & col("sku").isNotNull()
        & col("timestamp").isNotNull()
    )
    .withColumn("client_id", col("client_id").cast(LongType()))
    .withColumn("sku", col("sku").cast(LongType()))
    .withColumn(
        "timestamp",
        to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")
    )
    .withColumn("interaction", lit(3.0))
    .select("client_id", "sku", "timestamp", "interaction")
)

print("Add to cart events:", add_to_cart.count())


# ============================================================
# 4. READ PRODUCT_BUY
# ============================================================

print("\n===== READ PRODUCT_BUY =====")

product_buy = (
    spark.read.parquet(PRODUCT_BUY)
    .select("client_id", "sku", "timestamp")
    .filter(
        col("client_id").isNotNull()
        & col("sku").isNotNull()
        & col("timestamp").isNotNull()
    )
    .withColumn("client_id", col("client_id").cast(LongType()))
    .withColumn("sku", col("sku").cast(LongType()))
    .withColumn(
        "timestamp",
        to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")
    )
    .withColumn("interaction", lit(5.0))
    .select("client_id", "sku", "timestamp", "interaction")
)

print("Product buy events:", product_buy.count())


# ============================================================
# 5. COMBINE POSITIVE EVENTS
# ============================================================

print("\n===== COMBINE POSITIVE EVENTS =====")

events = (
    add_to_cart
    .unionByName(product_buy)
)

print("Total positive events:", events.count())


# ============================================================
# 6. AGGREGATE USER + ITEM
# ============================================================

print("\n===== AGGREGATE USER-ITEM =====")

interactions = (
    events
    .groupBy("client_id", "sku")
    .agg(
        spark_sum("interaction").alias("raw_score"),
        spark_max("timestamp").alias("last_timestamp")
    )
)


# ============================================================
# 7. REMOVE INVALID SCORES
# ============================================================

interactions = interactions.filter(
    col("raw_score") > 0
)


# ============================================================
# 8. LOG TRANSFORMATION
# ============================================================

print("\n===== APPLY LOG TRANSFORMATION =====")

interactions = (
    interactions
    .withColumn(
        "rating",
        log1p(col("raw_score"))
    )
    .withColumn(
        "rating",
        col("rating").cast(DoubleType())
    )
)


# ============================================================
# 9. FINAL ALS DATA
# ============================================================

als_data = (
    interactions
    .select(
        col("client_id").alias("userId"),
        col("sku").alias("itemId"),
        col("rating"),
        col("last_timestamp")
    )
)


# ============================================================
# 10. SCHEMA
# ============================================================

print("\n===== FINAL USER-ITEM INTERACTION =====")

als_data.printSchema()


# ============================================================
# 11. SAMPLE
# ============================================================

print("\n===== SAMPLE =====")

als_data.show(20, truncate=False)


# ============================================================
# 12. STATISTICS
# ============================================================

print("\n===== STATISTICS =====")

print(
    "Number of users:",
    als_data.select("userId").distinct().count()
)

print(
    "Number of items:",
    als_data.select("itemId").distinct().count()
)

print(
    "Number of interactions:",
    als_data.count()
)


# ============================================================
# 13. TIMESTAMP RANGE
# ============================================================

print("\n===== TIMESTAMP RANGE =====")

als_data.select(
    spark_min("last_timestamp").alias("min_timestamp"),
    spark_max("last_timestamp").alias("max_timestamp")
).show()


# ============================================================
# 14. RATING DISTRIBUTION
# ============================================================

print("\n===== RATING DISTRIBUTION =====")

als_data.select("rating").summary(
    "count",
    "min",
    "25%",
    "50%",
    "75%",
    "max"
).show()


# ============================================================
# 15. SAVE TO HDFS
# ============================================================

print("\n===== SAVE TO HDFS =====")

(
    als_data
    .write
    .mode("overwrite")
    .parquet(OUTPUT)
)

print("Saved to:")
print(OUTPUT)


# ============================================================
# 16. FINISH
# ============================================================

spark.stop()

print("\n===== DONE =====")
