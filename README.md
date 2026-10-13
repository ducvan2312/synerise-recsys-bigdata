# E-commerce Product Recommendation System using Big Data

## 1. Project Overview

This project develops a product recommendation system for an e-commerce platform using Big Data technologies.

The system uses the Synerise RecSys Challenge 2025 Dataset as the main data source and applies Apache Flume, Hadoop HDFS, Apache Spark, and Spark MLlib ALS to build a collaborative filtering recommendation system.

### Main Pipeline

Synerise Dataset
        ↓
Apache Flume
        ↓
Hadoop HDFS
        ↓
Apache Spark
        ↓
Data Cleaning
        ↓
User-Item Interaction
        ↓
Train / Test Split
        ↓
ALS Collaborative Filtering
        ↓
Top-N Product Recommendation
        ↓
Precision@K / Recall@K Evaluation


---

## 2. Project Objectives

The main objectives of the project are:

- Collect large-scale e-commerce event data.
- Transfer data into Hadoop HDFS using Apache Flume.
- Process and clean data using Apache Spark.
- Build user-item interaction data.
- Train a collaborative filtering recommendation model using ALS.
- Generate Top-N product recommendations for users.
- Evaluate recommendation quality using Precision@K and Recall@K.
- Provide a scalable Big Data recommendation pipeline.


---

## 3. Technologies

| Technology | Version / Purpose |
|------------|------------------|
| Ubuntu | Development environment |
| Hadoop | 3.2.1 |
| HDFS | Distributed storage |
| Apache Flume | 1.11.0 |
| Apache Spark | 3.5.5 |
| Spark MLlib | ALS recommendation |
| Java | 1.8.0_452 |
| Python | 3.x |
| Parquet | Data storage format |
| PySpark | Spark programming |


---

## 4. Dataset

### Synerise RecSys Challenge 2025 Dataset

The project uses the Synerise RecSys Challenge 2025 Dataset.

The dataset contains several types of e-commerce events:

- `page_visit`
- `add_to_cart`
- `remove_from_cart`
- `product_buy`
- `search_query`
- `product_properties`

### Main Fields

#### Event tables

The main event tables contain:

```text
client_id
sku
timestamp
