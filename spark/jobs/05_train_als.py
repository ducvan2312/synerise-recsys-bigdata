from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Synerise-ALS-Recommendation")
    .master("local[3]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. PATH
# ============================================================

TRAIN_PATH = "hdfs:///data/synerise_recsys/interactions/train"
MODEL_PATH = "hdfs:///data/synerise_recsys/model/als"


# ============================================================
# 3. READ TRAIN DATA
# ============================================================

print("\n========================================")
print("READ TRAIN DATA")
print("========================================")

train = spark.read.parquet(TRAIN_PATH)

print("Train data:")
train.printSchema()

print("\nSample:")
train.show(10, truncate=False)

print("\nNumber of training interactions:")
print(train.count())


# ============================================================
# 4. CREATE ALS MODEL
# ============================================================

print("\n========================================")
print("CREATE ALS MODEL")
print("========================================")

als = ALS(
    userCol="userId",
    itemCol="itemId",
    ratingCol="rating",

    # Number of latent factors
    rank=20,

    # Number of training iterations
    maxIter=10,

    # Regularization parameter
    regParam=0.1,

    # Use implicit feedback
    implicitPrefs=True,

    # Confidence scaling
    alpha=5.0,

    # Remove invalid predictions during evaluation
    coldStartStrategy="drop",

    # Reproducibility
    seed=42
)


# ============================================================
# 5. TRAIN ALS
# ============================================================

print("\n========================================")
print("TRAINING ALS MODEL")
print("========================================")

model = als.fit(train)

print("\nALS training completed successfully.")


# ============================================================
# 6. SAVE MODEL TO HDFS
# ============================================================

print("\n========================================")
print("SAVE ALS MODEL")
print("========================================")

model.write().overwrite().save(MODEL_PATH)

print("Model saved to:")
print(MODEL_PATH)


# ============================================================
# 7. FINISH
# ============================================================

spark.stop()

print("\n========================================")
print("DONE")
print("========================================")
