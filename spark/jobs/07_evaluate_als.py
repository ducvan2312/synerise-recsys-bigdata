from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum as spark_sum,
    log2,
    when,
    row_number,
    collect_list,
    size
)
from pyspark.sql.window import Window


# ============================================================
# 1. SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Synerise-ALS-Evaluation")
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
# 2. PATH
# ============================================================

TEST_PATH = "hdfs:///data/synerise_recsys/interactions/test"

RECOMMENDATION_PATH = (
    "hdfs:///data/synerise_recsys/results/recommendations"
)

TOP_K = 5

# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test = (
    spark.read
    .parquet(TEST_PATH)
    .select("userId", "itemId")
    .dropDuplicates(["userId", "itemId"])
    .cache()
)

test_count = test.count()
test_users = test.select("userId").distinct().count()

print(f"✓ Test interactions : {test_count:,}")
print(f"✓ Test users        : {test_users:,}")


# ============================================================
# 4. LOAD RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("LOADING RECOMMENDATIONS")
print("=" * 70)

recommendations = (
    spark.read
    .parquet(RECOMMENDATION_PATH)
    .select("userId", "itemId", "score")
    .cache()
)

recommendation_count = recommendations.count()
recommendation_users = (
    recommendations
    .select("userId")
    .distinct()
    .count()
)

print(f"✓ Recommendation records : {recommendation_count:,}")
print(f"✓ Recommendation users   : {recommendation_users:,}")


# ============================================================
# 5. ADD RANK TO RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("PREPARING TOP-K RECOMMENDATIONS")
print("=" * 70)

# The recommendation file already contains Top-5.
# Rank again using score so that NDCG can evaluate position.

rank_window = (
    Window
    .partitionBy("userId")
    .orderBy(col("score").desc(), col("itemId").asc())
)

ranked_recommendations = (
    recommendations
    .withColumn("rank", row_number().over(rank_window))
    .filter(col("rank") <= TOP_K)
)

ranked_count = ranked_recommendations.count()

print(f"✓ Top-{TOP_K} recommendation records : {ranked_count:,}")


# ============================================================
# 6. FIND HITS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING HITS")
print("=" * 70)

# A hit occurs when a recommended item is also present
# in the user's test interactions.

hits = (
    ranked_recommendations
    .join(
        test,
        on=["userId", "itemId"],
        how="inner"
    )
    .select(
        "userId",
        "itemId",
        "rank"
    )
    .cache()
)

hit_count = hits.count()
hit_users = hits.select("userId").distinct().count()

print(f"✓ Total hits       : {hit_count:,}")
print(f"✓ Users with hits  : {hit_users:,}")


# ============================================================
# 7. PRECISION@5
# ============================================================

print("\n" + "=" * 70)
print("PRECISION@5")
print("=" * 70)

precision_at_5 = hit_count / (recommendation_users * TOP_K)

print(f"Precision@5 = {precision_at_5:.6f}")


# ============================================================
# 8. RECALL@5
# ============================================================

print("\n" + "=" * 70)
print("RECALL@5")
print("=" * 70)

recall_at_5 = hit_count / test_count

print(f"Recall@5 = {recall_at_5:.6f}")


# ============================================================
# 9. NDCG@5
# ============================================================

print("\n" + "=" * 70)
print("NDCG@5")
print("=" * 70)

# For every hit:
#
# rank 1 -> 1 / log2(2) = 1
# rank 2 -> 1 / log2(3)
# rank 3 -> 1 / log2(4)
# ...
#
# DCG = sum(relevance / log2(rank + 1))

hits_with_gain = (
    hits
    .withColumn(
        "gain",
        1.0 / log2(col("rank") + 1)
    )
)

dcg_per_user = (
    hits_with_gain
    .groupBy("userId")
    .agg(
        spark_sum("gain").alias("dcg")
    )
)

# Ideal DCG:
# If a user has n relevant test items,
# the ideal ranking puts relevant items first.

test_relevance = (
    test
    .groupBy("userId")
    .agg(
        count("*").alias("relevant_count")
    )
)

# Maximum number of relevant items that can appear in Top-5
test_relevance = (
    test_relevance
    .withColumn(
        "ideal_k",
        when(
            col("relevant_count") > TOP_K,
            TOP_K
        ).otherwise(col("relevant_count"))
    )
)

# Create positions 1..5.
# This small DataFrame is only used to calculate IDCG.

positions = spark.range(1, TOP_K + 1).toDF("rank")

ideal_gain = (
    positions
    .withColumn(
        "gain",
        1.0 / log2(col("rank") + 1)
    )
)

ideal_gain = ideal_gain.withColumn(
    "dummy",
    col("rank")
)

ideal_per_user = (
    test_relevance
    .crossJoin(ideal_gain)
    .filter(col("rank") <= col("ideal_k"))
    .groupBy("userId")
    .agg(
        spark_sum("gain").alias("idcg")
    )
)

ndcg_per_user = (
    test_relevance
    .select("userId")
    .join(
        dcg_per_user,
        on="userId",
        how="left"
    )
    .join(
        ideal_per_user,
        on="userId",
        how="left"
    )
    .fillna(0.0, subset=["dcg"])
    .withColumn(
        "ndcg",
        when(
            col("idcg") > 0,
            col("dcg") / col("idcg")
        ).otherwise(0.0)
    )
)

ndcg_at_5 = (
    ndcg_per_user
    .agg({"ndcg": "avg"})
    .collect()[0][0]
)

print(f"NDCG@5 = {ndcg_at_5:.6f}")


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALS EVALUATION SUMMARY")
print("=" * 70)

print(f"Test users              : {test_users:,}")
print(f"Test interactions       : {test_count:,}")
print(f"Recommendation users    : {recommendation_users:,}")
print(f"Top-K                   : {TOP_K}")
print(f"Total hits              : {hit_count:,}")
print(f"Users with hits         : {hit_users:,}")
print(f"Precision@5             : {precision_at_5:.6f}")
print(f"Recall@5                : {recall_at_5:.6f}")
print(f"NDCG@5                  : {ndcg_at_5:.6f}")

print("=" * 70)

spark.stop()
