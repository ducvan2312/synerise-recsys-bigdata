from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = (
    SparkSession.builder
    .appName("Prepare-ALS-Data")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT = "hdfs:///data/synerise_recsys/interactions/user_item"
OUTPUT = "hdfs:///data/synerise_recsys/interactions/als_ready"

MIN_INTERACTIONS = 5

print("\n===== READ USER-ITEM INTERACTIONS =====")

df = spark.read.parquet(INPUT)

original_interactions = df.count()

print("Original interactions:", original_interactions)

# ==========================================
# STEP 1: FILTER USERS
# Giữ user có ít nhất 5 interactions
# ==========================================

print("\n===== STEP 1: FILTER USERS =====")

user_counts = (
    df.groupBy("userId")
      .agg(count("*").alias("user_interactions"))
)

active_users = (
    user_counts
    .filter(col("user_interactions") >= MIN_INTERACTIONS)
    .select("userId")
)

active_user_count = active_users.count()

print(
    "Users with >= 5 interactions:",
    active_user_count
)

df_user_filtered = df.join(
    active_users,
    on="userId",
    how="inner"
)

user_filtered_interactions = df_user_filtered.count()

print(
    "Interactions after user filtering:",
    user_filtered_interactions
)

# ==========================================
# STEP 2: FILTER ITEMS
# Giữ item có ít nhất 5 interactions
# ==========================================

print("\n===== STEP 2: FILTER ITEMS =====")

item_counts = (
    df_user_filtered
    .groupBy("itemId")
    .agg(count("*").alias("item_interactions"))
)

active_items = (
    item_counts
    .filter(col("item_interactions") >= MIN_INTERACTIONS)
    .select("itemId")
)

active_item_count = active_items.count()

print(
    "Items with >= 5 interactions:",
    active_item_count
)

als_ready = df_user_filtered.join(
    active_items,
    on="itemId",
    how="inner"
)

items_filtered_interactions = als_ready.count()

print(
    "Interactions after item filtering:",
    items_filtered_interactions
)

# ==========================================
# STEP 3: RE-FILTER USERS
#
# Sau khi loại bỏ các item ít interaction,
# một số user có thể còn < 5 interactions.
#
# Vì vậy phải lọc user lần thứ 2.
# ==========================================

print("\n===== STEP 3: RE-FILTER USERS =====")

final_user_counts = (
    als_ready
    .groupBy("userId")
    .agg(count("*").alias("user_interactions"))
)

final_active_users = (
    final_user_counts
    .filter(col("user_interactions") >= MIN_INTERACTIONS)
    .select("userId")
)

final_active_user_count = final_active_users.count()

print(
    "Final users with >= 5 interactions:",
    final_active_user_count
)

als_ready = als_ready.join(
    final_active_users,
    on="userId",
    how="inner"
)

# ==========================================
# STEP 4: FINAL ALS DATA
# ==========================================

print("\n===== FINAL ALS DATA =====")

als_ready = als_ready.select(
    "userId",
    "itemId",
    "rating",
    "last_timestamp"
)

als_ready.printSchema()

final_interactions = als_ready.count()
final_users = als_ready.select("userId").distinct().count()
final_items = als_ready.select("itemId").distinct().count()

print("Interactions:", final_interactions)
print("Users:", final_users)
print("Items:", final_items)

# ==========================================
# VERIFY FINAL USER DISTRIBUTION
# ==========================================

print("\n===== VERIFY USER INTERACTIONS =====")

verify_user_counts = (
    als_ready
    .groupBy("userId")
    .count()
)

users_below_minimum = (
    verify_user_counts
    .filter(col("count") < MIN_INTERACTIONS)
    .count()
)

min_user_interactions = (
    verify_user_counts
    .agg({"count": "min"})
    .collect()[0][0]
)

avg_user_interactions = (
    verify_user_counts
    .agg({"count": "avg"})
    .collect()[0][0]
)

max_user_interactions = (
    verify_user_counts
    .agg({"count": "max"})
    .collect()[0][0]
)

print(
    "Users with < 5 interactions:",
    users_below_minimum
)

print(
    "Minimum interactions/user:",
    min_user_interactions
)

print(
    "Average interactions/user:",
    avg_user_interactions
)

print(
    "Maximum interactions/user:",
    max_user_interactions
)

# ==========================================
# SAMPLE
# ==========================================

print("\n===== SAMPLE =====")

als_ready.show(20, truncate=False)

# ==========================================
# SAVE TO HDFS
# ==========================================

print("\n===== SAVE ALS DATA =====")

(
    als_ready
    .write
    .mode("overwrite")
    .parquet(OUTPUT)
)

print("Saved to:")
print(OUTPUT)

spark.stop()

print("\n===== DONE =====")
