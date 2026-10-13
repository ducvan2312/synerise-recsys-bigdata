from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALSModel
import random


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Synerise-Generate-Recommendations")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.executor.memory", "1g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .config("spark.driver.maxResultSize", "512m")
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

MODEL_PATH = "hdfs:///data/synerise_recsys/model/als"
TRAIN_PATH = "hdfs:///data/synerise_recsys/interactions/train"
OUTPUT_PATH = "hdfs:///data/synerise_recsys/results/recommendations"

TOP_N = 5
DEMO_USERS = 10
RANDOM_SEED = 42


# ============================================================
# 3. LOAD ALS MODEL
# ============================================================

print("=" * 70)
print("LOADING ALS MODEL")
print("=" * 70)

model = ALSModel.load(MODEL_PATH)

print("✓ ALS model loaded successfully")
print(f"  Rank             : {model.rank}")
print(f"  Max Iterations   : 10")
print(f"  Regularization   : 0.1")
print(f"  Implicit Prefs   : True")
print(f"  Alpha            : 5.0")

# ============================================================
# 4. LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train = (
    spark.read
    .parquet(TRAIN_PATH)
    .select("userId", "itemId", "rating")
)

train = train.cache()

train_count = train.count()

print(f"✓ Train interactions : {train_count:,}")

users = (
    train
    .select("userId")
    .distinct()
    .cache()
)

user_count = users.count()

print(f"✓ Train users        : {user_count:,}")


# ============================================================
# 5. GENERATE TOP-N RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print(f"GENERATING TOP-{TOP_N} RECOMMENDATIONS")
print("=" * 70)

print(f"Number of users      : {user_count:,}")
print(f"Recommendations/user : {TOP_N}")
print("Please wait...")

recommendations = model.recommendForUserSubset(
    users,
    TOP_N
)

print("✓ Recommendation generation completed")


# ============================================================
# 6. TRANSFORM RECOMMENDATIONS
# ============================================================

recommendations_flat = (
    recommendations
    .select(
        col("userId"),
        explode("recommendations").alias("recommendation")
    )
    .select(
        col("userId"),
        col("recommendation.itemId").alias("itemId"),
        col("recommendation.rating").alias("score")
    )
)

# ============================================================
# 7. SAVE RECOMMENDATIONS TO HDFS
# ============================================================

print("\n" + "=" * 70)
print("SAVING RECOMMENDATIONS")
print("=" * 70)

(
    recommendations_flat
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

print(f"✓ Saved to:")
print(f"  {OUTPUT_PATH}")


# ============================================================
# 8. VERIFY RESULT
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING RESULT")
print("=" * 70)

saved_recommendations = (
    spark.read
    .parquet(OUTPUT_PATH)
)

result_count = saved_recommendations.count()

result_users = (
    saved_recommendations
    .select("userId")
    .distinct()
    .count()
)

print(f"✓ Recommendation records : {result_count:,}")
print(f"✓ Users                  : {result_users:,}")
print(f"✓ Expected records       : {user_count * TOP_N:,}")


# ============================================================
# 9. SHOW RANDOM DEMO USERS
# ============================================================

print("\n" + "=" * 70)
print(f"RANDOM DEMO - {DEMO_USERS} USERS")
print("=" * 70)

random_users = users.rdd.takeSample(
    False,
    DEMO_USERS,
    RANDOM_SEED
)

selected_user_ids = [
    row["userId"]
    for row in random_users
]

demo_results = (
    saved_recommendations
    .filter(col("userId").isin(selected_user_ids))
    .orderBy("userId", col("score").desc())
)

demo_results.show(
    DEMO_USERS * TOP_N,
    truncate=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION GENERATION COMPLETED")
print("=" * 70)

print(f"ALS Rank            : {model.rank}")
print(f"ALS MaxIter         : 10")
print(f"ALS RegParam        : 0.1")
print(f"ALS ImplicitPrefs   : True")
print(f"ALS Alpha           : 5.0")
print(f"Users               : {user_count:,}")
print(f"Top-N               : {TOP_N}")
print(f"Total recommendations: {result_count:,}")
print(f"Output              : {OUTPUT_PATH}")

print("=" * 70)

spark.stop()
