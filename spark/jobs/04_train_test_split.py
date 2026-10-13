from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    row_number,
    floor,
    when
)
from pyspark.sql.window import Window


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Synerise-Chronological-Train-Test-Split")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 2. PATHS
# ============================================================

INPUT = "hdfs:///data/synerise_recsys/interactions/als_ready"

TRAIN_OUTPUT = "hdfs:///data/synerise_recsys/interactions/train"
TEST_OUTPUT = "hdfs:///data/synerise_recsys/interactions/test"

MIN_INTERACTIONS = 5


# ============================================================
# 3. READ USER-ITEM DATA
# ============================================================

print("\n===== READ USER-ITEM DATA =====")

df = spark.read.parquet(INPUT)

total = df.count()

print("Total interactions:", total)

print("\n===== SCHEMA =====")
df.printSchema()


# ============================================================
# 4. CHECK TIMESTAMP
# ============================================================

print("\n===== TIMESTAMP CHECK =====")

df = df.filter(
    col("last_timestamp").isNotNull()
)

df.select(
    "last_timestamp"
).summary(
    "count",
    "min",
    "max"
).show()


# ============================================================
# 5. FILTER USERS WITH ENOUGH INTERACTIONS
# ============================================================

print("\n===== FILTER USERS =====")

user_counts = (
    df
    .groupBy("userId")
    .count()
)

eligible_users = (
    user_counts
    .filter(
        col("count") >= MIN_INTERACTIONS
    )
    .select("userId")
)

eligible_count = (
    eligible_users
    .count()
)

print(
    "Users with >=",
    MIN_INTERACTIONS,
    "interactions:",
    eligible_count
)


df = (
    df
    .join(
        eligible_users,
        on="userId",
        how="inner"
    )
)

filtered_interactions = df.count()

print(
    "Interactions after filtering:",
    filtered_interactions
)


# ============================================================
# 6. CHRONOLOGICAL SPLIT 80/20
# ============================================================

print("\n===== CHRONOLOGICAL SPLIT 80/20 =====")

# For each user:
#
# Earlier interactions -> TRAIN
# Later interactions   -> TEST
#
# Example:
#
# 5 interactions:
# 1 2 3 4 | 5
#   TRAIN  | TEST
#
# 10 interactions:
# 1 2 3 4 5 6 7 8 | 9 10
#       TRAIN       | TEST


window = (
    Window
    .partitionBy("userId")
    .orderBy(
        col("last_timestamp").asc(),
        col("itemId").asc()
    )
)


ranked = (
    df
    .withColumn(
        "row_num",
        row_number().over(window)
    )
)


# Count interactions for every user

user_total = (
    ranked
    .groupBy("userId")
    .count()
    .withColumnRenamed(
        "count",
        "total_count"
    )
)


ranked = (
    ranked
    .join(
        user_total,
        on="userId",
        how="inner"
    )
)


# Calculate 80% train size

ranked = (
    ranked
    .withColumn(
        "train_limit_raw",
        floor(
            col("total_count") * 0.8
        )
    )
    .withColumn(
        "train_limit",
        when(
            col("train_limit_raw") >= 1,
            col("train_limit_raw")
        ).otherwise(1)
    )
)


# ============================================================
# 7. CREATE TRAIN DATA
# ============================================================

train = (
    ranked
    .filter(
        col("row_num") <= col("train_limit")
    )
    .select(
        "userId",
        "itemId",
        "rating"
    )
)


# ============================================================
# 8. CREATE TEST DATA
# ============================================================

test = (
    ranked
    .filter(
        col("row_num") > col("train_limit")
    )
    .select(
        "userId",
        "itemId",
        "rating"
    )
)


# ============================================================
# 9. SPLIT STATISTICS
# ============================================================

print("\n===== SPLIT STATISTICS =====")

train_count = train.count()
test_count = test.count()

print("Train interactions:", train_count)
print("Test interactions :", test_count)
print(
    "Total after split :",
    train_count + test_count
)


# ============================================================
# 10. TRAIN USERS / ITEMS
# ============================================================

print("\n===== TRAIN USERS / ITEMS =====")

train_users_count = (
    train
    .select("userId")
    .distinct()
    .count()
)

train_items_count = (
    train
    .select("itemId")
    .distinct()
    .count()
)

print("Train users:", train_users_count)
print("Train items:", train_items_count)


# ============================================================
# 11. TEST USERS / ITEMS
# ============================================================

print("\n===== TEST USERS / ITEMS =====")

test_users_count = (
    test
    .select("userId")
    .distinct()
    .count()
)

test_items_count = (
    test
    .select("itemId")
    .distinct()
    .count()
)

print("Test users:", test_users_count)
print("Test items:", test_items_count)


# ============================================================
# 12. CHECK USER OVERLAP
# ============================================================

print("\n===== USER OVERLAP CHECK =====")

train_users = (
    train
    .select("userId")
    .distinct()
)

test_users = (
    test
    .select("userId")
    .distinct()
)

overlap_users = (
    train_users
    .join(
        test_users,
        on="userId",
        how="inner"
    )
    .count()
)

print(
    "Users appearing in both train and test:",
    overlap_users
)


# ============================================================
# 13. CHECK ITEM OVERLAP
# ============================================================

print("\n===== ITEM OVERLAP CHECK =====")

train_items = (
    train
    .select("itemId")
    .distinct()
)

test_items = (
    test
    .select("itemId")
    .distinct()
)

overlap_items = (
    train_items
    .join(
        test_items,
        on="itemId",
        how="inner"
    )
    .count()
)

print(
    "Items appearing in both train and test:",
    overlap_items
)


# ============================================================
# 14. SAVE TRAIN DATA
# ============================================================

print("\n===== SAVE TRAIN =====")

(
    train
    .write
    .mode("overwrite")
    .parquet(TRAIN_OUTPUT)
)

print("Saved train:")
print(TRAIN_OUTPUT)


# ============================================================
# 15. SAVE TEST DATA
# ============================================================

print("\n===== SAVE TEST =====")

(
    test
    .write
    .mode("overwrite")
    .parquet(TEST_OUTPUT)
)

print("Saved test:")
print(TEST_OUTPUT)


# ============================================================
# 16. FINISH
# ============================================================

spark.stop()

print("\n===== DONE =====")
