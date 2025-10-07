import findspark

findspark.init()

import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, BooleanType, ArrayType
from pyspark.sql.window import Window
from pyspark.sql.functions import (col, row_number, struct, lit, to_json, count, current_timestamp,
                                   current_date, date_format, broadcast, first, desc, unix_timestamp, when, sum, collect_list,
                                   max, lit, concat_ws, from_unixtime, format_string, expr, coalesce, dense_rank,
                                   add_months, last_day)
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Reusable imports from userReport structure
from constants.ParquetFileConstants import ParquetFileConstants
from dfutil.assessment import assessmentDFUtil
from dfutil.content import contentDFUtil
from dfutil.dfexport import dfexportutil
from dfutil.utils import utils
from dfutil.utils.redis import Redis
from jobs.default_config import create_config
from jobs.config import get_environment_config


class ODCSRecommendationModel:
    def __init__(self):
        self.class_name = "org.ekstep.analytics.dashboard.ODCSRecommendationModel"

    def name(self):
        return "ODCSRecommendationModel"

    def process_data(self, spark, config):
        try:
            all_enrolments_df = spark.read.parquet(ParquetFileConstants.ENROLMENT_WAREHOUSE_COMPUTED_PARQUET_FILE).withColumnRenamed("userID", "user_id")
            content_df = spark.read.parquet(ParquetFileConstants.CONTENT_WAREHOUSE_COMPUTED_PARQUET_FILE) \
                .filter(col("content_sub_type").isin("Course", "Program", "Moderated Course", "Moderated Program"))
            enrolments_df = all_enrolments_df.join(
                content_df.select("content_id"), ["content_id"], "inner"
            )

            user_df = spark.read.parquet(ParquetFileConstants.USER_WAREHOUSE_COMPUTED_PARQUET_FILE)
            rating_draft_df = spark.read.parquet(ParquetFileConstants.RATING_PARQUET_FILE)

            completion_df = enrolments_df.groupBy("content_id").agg(count("user_id").alias("total_enrolments"),
                                                                    sum(when(col("user_consumption_status") == lit(
                                                                        "completed"), 1).otherwise(0))
                                                                    .alias("completed_enrolments")) \
                .withColumn("completion_percentage",
                            (col("completed_enrolments") / col("total_enrolments")) * lit(100.0))

            rating_df = (
                rating_draft_df
                .filter(col("activitytype") == lit("Course"))
                .groupBy("activityid")
                .agg(count(lit(1)).alias("rating_count"), sum(col("rating")).alias("total_rating"))
                .withColumn("avg_rating", col("total_rating") / col("rating_count"))
                .select(
                    col("activityid").alias("content_id"),
                    col("rating_count"),
                    col("avg_rating"),
                )
            )

            enrolments_with_mdo = enrolments_df.join(user_df, ["user_id"], "inner").select(
                "mdo_id", "content_id"
            )

            content_stats_df = (
                enrolments_with_mdo
                .join(completion_df, ["content_id"], "left")
                .join(rating_df, ["content_id"], "left")
                .groupBy("mdo_id", "content_id")
                .agg(first("completion_percentage").alias("completion_percentage"), first("avg_rating")
                     .alias("avg_rating"), count("content_id").alias("total_enrolments"),)
                .na.fill(0, subset=["completion_percentage", "avg_rating", "total_enrolments"])
            )

            window_spec = Window.partitionBy("mdo_id").orderBy(
                desc("completion_percentage"), desc("avg_rating"), desc("total_enrolments")
            )

            ranked_content_df = (
                content_stats_df
                .withColumn("rank", row_number().over(window_spec))
                .filter(col("rank") <= 15)
            )

            final_df = (
                ranked_content_df
                .groupBy("mdo_id")
                .agg(collect_list("content_id").alias("top_content_ids"))
                .withColumn("top_15_content_ids", concat_ws(",", col("top_content_ids")))
                .select("mdo_id", "top_15_content_ids")
            )

            final_df.show(10, truncate=False)
            Redis.dispatchDataFrame("odcs_course_recomendation", final_df, "mdo_id", "top_15_content_ids", conf=config)
           # Redis.dispatchDataFrame("odcs_course_recomendation_pyspark_test", final_df, "mdo_id", "top_15_content_ids", conf=config)
        except Exception as e:
            print(f"Error occurred during ODCS Recommendation processing: {str(e)}")
            raise

def create_spark_session_with_packages(config):
    spark = SparkSession.builder \
        .appName("ODCS Recommendation Model - Cached") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.memory", "18g") \
        .config("spark.driver.memory", "18g") \
        .config("spark.executor.memoryFraction", "0.7") \
        .config("spark.storage.memoryFraction", "0.2") \
        .config("spark.storage.unrollFraction", "0.1") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.cassandra.connection.host", config.sparkCassandraConnectionHost) \
        .config("spark.cassandra.connection.port", '9042') \
        .config("spark.cassandra.output.batch.size.rows", '10000') \
        .config("spark.cassandra.connection.keepAliveMS", "60000") \
        .config("spark.cassandra.connection.timeoutMS", '30000') \
        .config("spark.cassandra.read.timeoutMS", '30000') \
        .getOrCreate()
    return spark


def main():
    # Create model instance
    start_time = datetime.now()
    print(f"[START] ODCS Recommendation processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    config_dict = get_environment_config()
    config = create_config(config_dict)
    spark = create_spark_session_with_packages(config)
    model = ODCSRecommendationModel()
    model.process_data(spark, config)
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"[END] ODCS Recommendation completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total duration: {duration}")
    spark.stop()


if __name__ == "__main__":
    main()