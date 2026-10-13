# E-commerce Product Recommendation System using Big Data

## 1. Project Overview

This project develops an e-commerce product recommendation system using Big Data technologies.

The main dataset is the **Synerise RecSys Challenge 2025 Dataset**. The pipeline uses Apache Flume, Hadoop HDFS, Apache Spark, and Spark MLlib ALS to process user-item interactions, train a collaborative-filtering model, generate recommendations, and evaluate results.

### Main Pipeline

```text
Synerise Dataset
      ↓
Local Parquet-to-JSONL Export
      ↓
Apache Flume
      ↓
Hadoop HDFS
      ↓
Apache Spark Data Cleaning
      ↓
User-Item Interactions
      ↓
ALS-Ready Data
      ↓
Train / Test Split
      ↓
ALS Collaborative Filtering
      ↓
Top-N Recommendations
      ↓
Evaluation
```

## 2. Project Objectives

- Ingest e-commerce event data into HDFS using Apache Flume.
- Clean and process data using Apache Spark.
- Build user-item interaction data.
- Train an ALS collaborative-filtering model.
- Generate Top-N product recommendations.
- Evaluate recommendation quality using Precision@K, Recall@K, and NDCG@K where implemented.
- Build a reproducible Big Data processing pipeline.

## 3. Technologies

| Technology | Version / Purpose |
|---|---|
| Ubuntu | Development environment |
| Hadoop | 3.2.1 |
| HDFS | Distributed storage |
| Apache Flume | 1.11.0 |
| Apache Spark | 3.5.5 |
| Spark MLlib | ALS recommendation |
| Java | 1.8.0_452 |
| Python | 3.x |
| PySpark | Distributed data processing |
| Parquet | Columnar data storage |

## 4. Dataset

### Synerise RecSys Challenge 2025

The dataset contains e-commerce event tables such as:

- `page_visit`
- `add_to_cart`
- `remove_from_cart`
- `product_buy`
- `search_query`
- `product_properties`

Fields vary by event type. Event tables may contain fields such as `client_id`, `sku`, and `timestamp` where available. In this project, `client_id` is used as a user identifier and `sku` as a product identifier in interaction processing.

The original dataset is not included in this repository. Obtain it from the dataset provider and place the required Parquet files in the configured local input directory before running the export job.

## 5. Project Structure

```text
recsys_project/
├── data/
│   ├── raw/
│   │   └── synerise/
│   └── staging/
│       └── flume_export/
├── flume/
│   └── conf/
├── spark/
│   └── jobs/
│       ├── 00_export_parquet_to_jsonl.py
│       ├── 01_data_cleaning.py
│       ├── 02_build_interactions.py
│       ├── 03_prepare_als_data.py
│       ├── 04_train_test_split.py
│       ├── 05_train_als.py
│       ├── 06_generate_recommendations.py
│       └── 07_evaluate_als.py
├── .gitignore
└── README.md
```

This is a logical overview. Use the actual filenames and configuration files present in the repository if they differ.

## 6. HDFS Data Layout

The pipeline is configured around the following HDFS locations:

```text
/data/synerise_recsys/
├── raw/
├── processed/
├── interactions/
│   ├── user_item/
│   ├── als_ready/
│   ├── train/
│   └── test/
├── model/
│   └── als/
└── results/
    └── recommendations/
```

These directories are created or populated as the corresponding jobs run. Exact contents depend on which stages have completed successfully.

## 7. Environment Setup

### Prerequisites

Install and configure Hadoop/HDFS, Apache Spark, Java, Python, and Apache Flume. Start the required Hadoop services before running jobs that read from or write to HDFS.

Check versions in the configured environment:

```bash
java -version
hadoop version
spark-submit --version
flume-ng version
python3 --version
```

### Clone the repository

```bash
git clone https://github.com/ducvan2312/synerise-recsys-bigdata.git
cd synerise-recsys-bigdata
```

Some scripts and Flume configuration files may use absolute paths from the original development machine. Update those paths to match your environment before running the pipeline.

## 8. Data Ingestion and Processing

Run stages in order. Before running each command, verify that the input files, required services, and paths are available.

### Step 1: Export local Parquet files

Review the local input and output paths in `spark/jobs/00_export_parquet_to_jsonl.py`.

```bash
spark-submit spark/jobs/00_export_parquet_to_jsonl.py
```

This job exports the configured local Parquet tables to JSONL files for ingestion. The expected input filenames and directories must match the script.

### Step 2: Run Flume

Review the configuration files under `flume/conf/`. Verify the spool directories, channel settings, HDFS sink paths, and agent name.

Example command (adjust the configuration filename, agent name, and Flume installation paths to match your configuration):

```bash
flume-ng agent \
  --conf "$FLUME_HOME/conf" \
  --conf-file flume/conf/synerise.conf \
  --name a1 \
  -Dflume.root.logger=INFO,console
```

Run Flume after the configured spool directories exist and HDFS is available. This example works only if the selected configuration defines the agent as `a1` and the environment is configured accordingly.

### Step 3: Clean event data

```bash
spark-submit spark/jobs/01_data_cleaning.py
```

This job reads raw event data from HDFS and writes cleaned data to the configured processed-data location.

### Step 4: Build user-item interactions

```bash
spark-submit spark/jobs/02_build_interactions.py
```

This job builds user-item interaction records from the event types selected in the implementation. Consult the script to see which events contribute to the interaction data.

### Step 5: Prepare ALS data

```bash
spark-submit spark/jobs/03_prepare_als_data.py
```

This job prepares the interaction records for model training.

### Step 6: Split train and test data

```bash
spark-submit spark/jobs/04_train_test_split.py
```

This job creates training and test datasets according to the split logic implemented in the script.

### Step 7: Train ALS

```bash
spark-submit spark/jobs/05_train_als.py
```

The expected ALS settings in the project version documented here are:

| Parameter | Value |
|---|---:|
| Rank | 20 |
| Max iterations | 10 |
| Regularization parameter | 0.1 |
| Implicit preferences | True |
| Alpha | 5.0 |
| Random seed | 42 |

Verify the current values in `spark/jobs/05_train_als.py`, because local code changes may make the effective settings different.

### Step 8: Generate recommendations

```bash
spark-submit spark/jobs/06_generate_recommendations.py
```

This job loads the trained model and generates recommendations according to the Top-N value and output path configured in the script.

### Step 9: Evaluate recommendations

```bash
spark-submit spark/jobs/07_evaluate_als.py
```

This job evaluates generated recommendations against the test data. Check the script for the exact value of K and the implemented metric definitions.

## 9. Verify Outputs

List the HDFS project directories:

```bash
hdfs dfs -ls /data/synerise_recsys/
hdfs dfs -ls /data/synerise_recsys/processed/
hdfs dfs -ls /data/synerise_recsys/interactions/
hdfs dfs -ls /data/synerise_recsys/model/als/
hdfs dfs -ls /data/synerise_recsys/results/recommendations/
```

Check available disk space:

```bash
df -h
```

A job should be considered successful when it exits successfully and creates the expected output. Some Hadoop environments print SLF4J multiple-binding warnings; distinguish warnings from fatal errors by checking the final output and exit status.

## 10. Evaluation Metrics

The project may report the following metrics, depending on the implementation in `spark/jobs/07_evaluate_als.py`:

- **Precision@K:** proportion of recommended items that are relevant.
- **Recall@K:** proportion of relevant items retrieved in the top-K recommendations.
- **NDCG@K:** ranking-sensitive measure that gives greater weight to relevant items appearing higher in the ranking.

Results depend on the data, cleaning rules, train/test split, model parameters, and metric implementation. Run the evaluation job to obtain the current results; no fixed benchmark score is claimed in this README.

## 11. Data and Model Files

Large datasets, intermediate files, trained models, and runtime outputs are excluded from Git when covered by `.gitignore`. The original dataset is not bundled with this repository.

Keep data in the configured local directories and regenerate intermediate data and models by running the pipeline. Do not commit access tokens, credentials, or private data.

## 12. Notes

- Run the Spark jobs with `spark-submit` from the project directory.
- Ensure HDFS is running before executing jobs that use HDFS.
- Review absolute paths in the Spark scripts and Flume configuration before running on another machine.
- Run the stages in order because later stages depend on earlier outputs.
- This README describes the intended workflow. The current source code and configuration files determine the actual paths, parameters, and supported commands.
