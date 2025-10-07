import findspark

findspark.init()

import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, ArrayType
from pyspark.sql.window import Window
from pyspark.sql.functions import (col, explode, desc, row_number, countDistinct, current_timestamp, date_format,
                                   broadcast, unix_timestamp, when, lit, concat_ws, from_unixtime, format_string, expr)
from datetime import datetime, timedelta, time as dtime, timezone
from pyspark.sql import functions as F
import sys
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Reusable imports from userReport structure (kept for consistency; not used below)
from constants.ParquetFileConstants import ParquetFileConstants
from dfutil.assessment import assessmentDFUtil
from dfutil.content import contentDFUtil
from dfutil.dfexport import dfexportutil
from jobs.default_config import create_config
from jobs.config import get_environment_config
from dfutil.utils.redis import Redis
from dfutil.utils import utils


class NationalLearningWeekModel:
    def __init__(self):
        self.class_name = "org.ekstep.analytics.dashboard.NationalLearningWeekModel"

    def name(self):
        return "NationalLearningWeekModel"

    @staticmethod
    def get_date():
        return datetime.now().strftime("%Y-%m-%d")

    def process_data(self, spark, config):
        try:
            start_time = time.time()
            today = self.get_date()
            currentDateTime = date_format(current_timestamp(), "yyyy-MM-dd HH:mm:ss")
            app_postgres_url = f"jdbc:postgresql://{config.appPostgresHost}/{config.appPostgresSchema}"

            output_path = getattr(config, 'baseCachePath', '/home/analytics/pyspark/data-res/pq_files/cache_pq/')
            ORG_PARQUET = f"{output_path}/orgHierarchy"
            EVENT_ENROLMENT_PARQUET = f"{output_path}/eventEnrolmentDetails"
            # -----------------------------
            # HARDCODED NLW CONFIG (edit)
            # -----------------------------
            NLW_CONFIG = {
                "default": {
                    "start": config.nationalLearningWeekStart,
                    "end": config.nationalLearningWeekEnd,
                },
                "overrides": config.overridesForSlw,  # dict: {mdo_id: {"start": "...", "end": "..."}}
                "rolled_up_required": config.rollupRequiredOrgs,  # list of mdo_ids
            }

            orgDF = spark.read.parquet(ORG_PARQUET)
            userDF = spark.read.parquet(ParquetFileConstants.USER_ORG_COMPUTED_FILE) \
                .withColumnRenamed("userID", "user_id") \
                .withColumnRenamed("userOrgID", "mdo_id")

            contentEnrolmentDataDF = spark.read.parquet(ParquetFileConstants.ENROLMENT_WAREHOUSE_COMPUTED_PARQUET_FILE) \
                .withColumnRenamed("userID", "user_id") \
                .withColumnRenamed("certificateID", "certificate_id")
            eventsEnrolmentDataDF = spark.read.parquet(EVENT_ENROLMENT_PARQUET)
            org_map_df = orgDF.select(
                F.col("mdo_id"),
                F.coalesce(F.col("ministry_id"), F.col("mdo_id")).alias("ministry_id")
            ).dropDuplicates(["mdo_id"])

            def apply_window(df, ts_col, start_str, end_str):
                return df.filter((F.col(ts_col) >= F.lit(start_str)) & (F.col(ts_col) <= F.lit(end_str)))

            def override_windows(base_df, value_agg_fn, overrides_dict, key_col="mdo_id"):
                """Recompute metric per-org for overrides and replace base rows for those keys."""
                if not overrides_dict:
                    return base_df
                parts = []
                for org_id, win in overrides_dict.items():
                    ov = value_agg_fn(win["start"], win["end"]).filter(F.col(key_col) == org_id)
                    parts.append(ov)
                if not parts:
                    return base_df
                merged = parts[0]
                for p in parts[1:]:
                    merged = merged.unionByName(p)
                return (
                    base_df.join(merged.select(key_col).dropDuplicates(), [key_col], "left_anti")
                    .unionByName(merged)
                )

            def apply_rollup(mdo_metric_df, ministry_metric_df, value_col="value"):
                """
                Replace values for mdo_ids listed in rolled_up_required with their ministry totals.
                Output remains keyed by mdo_id.
                """
                rolled = NLW_CONFIG.get("rolled_up_required", [])
                if not rolled:
                    return mdo_metric_df
                rolled_df = spark.createDataFrame([(x,) for x in rolled], "mdo_id string")
                rolled_with_min = rolled_df.join(org_map_df, "mdo_id", "left")
                min_vals = ministry_metric_df.select(
                    F.col("ministry_id").alias("ministry_id_for_roll"),
                    F.col(value_col).alias("ministry_value_for_roll")
                )
                rolled_vals = rolled_with_min.join(
                    min_vals, rolled_with_min.ministry_id == min_vals.ministry_id_for_roll, "left"
                ).select("mdo_id", F.col("ministry_value_for_roll").alias(value_col))
                return (
                    mdo_metric_df.join(rolled_df, "mdo_id", "left_anti")
                    .unionByName(rolled_vals)
                )

            default_start = NLW_CONFIG["default"]["start"]
            default_end = NLW_CONFIG["default"]["end"]
            overrides = NLW_CONFIG.get("overrides", {})

            # yesterday IST (for daily pulse)
            ist = timezone(timedelta(hours=5, minutes=30))
            today_ist = datetime.now(ist).date()
            y_start = datetime.combine(today_ist - timedelta(days=1), dtime.min, tzinfo=ist).strftime(
                "%Y-%m-%d %H:%M:%S")
            y_end = (datetime.combine(today_ist, dtime.min, tzinfo=ist) - timedelta(seconds=1)).strftime(
                "%Y-%m-%d %H:%M:%S")

            # -----------------------------
            # METRIC #1: TOTAL ENROLMENTS (content + event)
            # -----------------------------
            def enrolments_mdo(start, end):
                c = (apply_window(contentEnrolmentDataDF, "enrolled_on", start, end)
                     .join(userDF, ["user_id"], "left")
                     .join(org_map_df, ["mdo_id"], "left")
                     .groupBy("mdo_id").agg(F.count(F.lit(1)).alias("c_enrol")))
                e = (apply_window(eventsEnrolmentDataDF, "enrolled_on_datetime", start, end)
                     .join(userDF, ["user_id"], "left")
                     .join(org_map_df, ["mdo_id"], "left")
                     .groupBy("mdo_id").agg(F.count(F.lit(1)).alias("e_enrol")))
                return (c.join(e, ["mdo_id"], "full_outer")
                        .select("mdo_id", (F.coalesce(F.col("c_enrol"), F.lit(0))
                                           + F.coalesce(F.col("e_enrol"), F.lit(0))).alias("value")))

            def enrolments_ministry(start, end):
                c = (apply_window(contentEnrolmentDataDF, "enrolled_on", start, end)
                     .join(userDF, ["user_id"], "left")
                     .join(org_map_df, ["mdo_id"], "left")
                     .groupBy("ministry_id").agg(F.count(F.lit(1)).alias("c_enrol")))
                e = (apply_window(eventsEnrolmentDataDF, "enrolled_on_datetime", start, end)
                     .join(userDF, ["user_id"], "left")
                     .join(org_map_df, ["mdo_id"], "left")
                     .groupBy("ministry_id").agg(F.count(F.lit(1)).alias("e_enrol")))
                return (c.join(e, ["ministry_id"], "full_outer")
                        .select("ministry_id", (F.coalesce(F.col("c_enrol"), F.lit(0))
                                                + F.coalesce(F.col("e_enrol"), F.lit(0))).alias("value")))

            enrol_mdo_base = enrolments_mdo(default_start, default_end)
            enrol_mdo = override_windows(enrol_mdo_base, enrolments_mdo, overrides, key_col="mdo_id")
            enrol_ministry = enrolments_ministry(default_start, default_end)
            enrolments_df = apply_rollup(enrol_mdo, enrol_ministry, "value")
            Redis.dispatchDataFrame("dashboard_total_enrolment_by_ministry_slw_count", enrolments_df, "ministry_id",
                                    "value", conf=config)
            # -----------------------------
            # METRIC #2: TOTAL CERTIFICATES (content + event)
            # -----------------------------
            def certificates_mdo(start, end):
                c = (apply_window(contentEnrolmentDataDF.filter(F.col("certificate_id").isNotNull()),
                                  "first_completed_on", start, end)
                     .join(userDF, ["user_id"], "left")
                     .join(org_map_df, ["mdo_id"], "left")
                     .groupBy("mdo_id").agg(F.count(F.lit(1)).alias("c_cert")))
                e = (apply_window(eventsEnrolmentDataDF.filter(F.col("certificate_id").isNotNull()),
                                  "completed_on_datetime", start, end)
                     .join(userDF, ["user_id"], "left")
                     .join(org_map_df, ["mdo_id"], "left")
                     .groupBy("mdo_id").agg(F.countDistinct("certificate_id").alias("e_cert")))
                return (c.join(e, ["mdo_id"], "full_outer")
                        .select("mdo_id", (F.coalesce(F.col("c_cert"), F.lit(0))
                                           + F.coalesce(F.col("e_cert"), F.lit(0))).alias("value")))

            def certificates_ministry(start, end):
                c = (apply_window(contentEnrolmentDataDF.filter(F.col("certificate_id").isNotNull()),
                                  "first_completed_on", start, end)
                     .join(userDF, ["user_id"], "left")
                     .join(org_map_df, ["mdo_id"], "left")
                     .groupBy("ministry_id").agg(F.count(F.lit(1)).alias("c_cert")))
                e = (apply_window(eventsEnrolmentDataDF.filter(F.col("certificate_id").isNotNull()),
                                  "completed_on_datetime", start, end)
                     .join(userDF, ["user_id"], "left")
                     .join(org_map_df, ["mdo_id"], "left")
                     .groupBy("ministry_id").agg(F.countDistinct("certificate_id").alias("e_cert")))
                return (c.join(e, ["ministry_id"], "full_outer")
                        .select("ministry_id", (F.coalesce(F.col("c_cert"), F.lit(0))
                                                + F.coalesce(F.col("e_cert"), F.lit(0))).alias("value")))

            cert_mdo_base = certificates_mdo(default_start, default_end)
            cert_mdo = override_windows(cert_mdo_base, certificates_mdo, overrides, key_col="mdo_id")
            cert_ministry = certificates_ministry(default_start, default_end)
            certificates_df = apply_rollup(cert_mdo, cert_ministry, "value")
            Redis.dispatchDataFrame("dashboard_certificates_generated_by_ministry_slw_count", certificates_df, "ministry_id",
                                    "value", conf=config)
            # -----------------------------
            # METRIC #3: YESTERDAY CERTIFICATES (daily pulse)
            # -----------------------------
            y_mdo = certificates_mdo(y_start, y_end)
            y_ministry = certificates_ministry(y_start, y_end)
            yday_certificates_df = apply_rollup(y_mdo, y_ministry, "value")
            Redis.dispatchDataFrame("dashboard_certificate_generated_yday_by_ministry_slw_count", yday_certificates_df,
                                    "ministry_id", "value", conf=config)
            # -------------------------
            # METRIC #4: EVENTS PUBLISHED (RKS)
            # -------------------------

            slwDateConditions = f'{{"range": {{"startDate": {{"gte": "{default_start.split(" ")[0]}", "lte": "{default_end.split(" ")[0]}"}}}}}}'

            objectType = ["Event"]
            shouldClauseRequired = ",".join([f'{{"match":{{"objectType.raw":"{pc}"}}}}' for pc in objectType])

            fieldsRequired = ["identifier", "name", "objectType", "resourceType", "status", "startDate", "startTime",
                              "duration", "registrationLink", "createdFor", "recordedLinks", "resourceTypeDetails"]
            arrayFieldsRequired = ["createdFor", "recordedLinks"]
            fieldsClauseRequired = ",".join([f'"{f}"' for f in fieldsRequired])

            eventQuery = (
                f'{{"_source":[{fieldsClauseRequired}],'f'"query":{{"bool":{{"must":[{slwDateConditions}],"should":[{shouldClauseRequired}]}}}}}}')

            eventDataDF = utils.read_elasticsearch_data(spark, config.sparkElasticsearchConnectionHost,
                                                        config.sparkElasticsearchConnectionPort, "compositesearch",
                                                        eventQuery, fieldsRequired, arrayFieldsRequired)

            filteredDF = eventDataDF.filter(col("resourceType") == "Rajya Karmayogi Saptah")

            # resourceTypeDetails.stateOrMinistryId is an array → explode it
            explodedDF = filteredDF.select(col("identifier"),
                                           explode(col("resourceTypeDetails.stateOrMinistryId")).alias(
                                               "stateOrMinistryId"))

            publishedEventsCountByCreatedFor = (
                explodedDF.groupBy("stateOrMinistryId").agg(countDistinct("identifier").alias("event_count")).orderBy(
                    desc("event_count")))

            publishedEventsCountByCreatedFor.show(truncate=False)

            events_published_df = publishedEventsCountByCreatedFor.select(col("stateOrMinistryId").alias("mdo_id"),
                                                                          col("event_count"))
            Redis.dispatchDataFrame("dashboard_events_published_by_ministry_count", events_published_df,
                                    "mdo_id", "event_count", conf=config)

            base_leaderboard = self.build_user_leaderboard_for_window(spark, default_start, default_end,
                                                                      ParquetFileConstants, userDF,
                                                                      contentEnrolmentDataDF, eventsEnrolmentDataDF)

            if overrides:
                override_org_ids = list(overrides.keys())
                ovDF = spark.createDataFrame([(x,) for x in override_org_ids], "org_id string")
                base_without_overrides = base_leaderboard.join(ovDF, on="org_id", how="left_anti")

                parts = [base_without_overrides]
                for org_id, win in overrides.items():
                    ov_part = self.build_user_leaderboard_for_window(spark, win["start"], win["end"],
                                                                     ParquetFileConstants, userDF,
                                                                     contentEnrolmentDataDF,
                                                                     eventsEnrolmentDataDF).filter(
                        F.col("org_id") == org_id)
                    parts.append(ov_part)

                final_leaderboard = parts[0]
                for p in parts[1:]:
                    final_leaderboard = final_leaderboard.unionByName(p)
            else:
                final_leaderboard = base_leaderboard
            self.write_postgres_table(final_leaderboard, app_postgres_url, "nlw_user_leaderboard_pyspark_test",
                                      config.appPostgresUsername, config.appPostgresCredential)
            #self.write_postgres_table(final_leaderboard, app_postgres_url, config.dwNLWUserLeaderboardTable, config.appPostgresUsername, config.appPostgresCredential)
            total_time = time.time() - start_time
            print(f"\n✅ NLW metrics completed in {total_time:.2f}s ({total_time / 60:.1f} min)")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise
    def write_postgres_table(self, df, url: str, table: str, username: str, password: str, mode: str = "overwrite"):
        df.write \
            .format("jdbc") \
            .option("url", url) \
            .option("dbtable", table) \
            .option("user", username) \
            .option("password", password) \
            .option("driver", "org.postgresql.Driver") \
            .option("truncate", "true") \
            .mode("overwrite") \
            .save()
    def duration_to_hours(self, colname: str):
        parts = F.split(F.col(colname), ":")
        return F.when(F.col(colname).rlike(r"^\d{1,2}:\d{2}:\d{2}$"), \
                      parts.getItem(0).cast("double") \
                      + parts.getItem(1).cast("double") / 60.0 \
                      + parts.getItem(2).cast("double") / 3600.0).otherwise(F.lit(0.0))

    def build_user_leaderboard_for_window(self, spark, default_start, default_end, ParquetFileConstants, userDF,
                                          contentEnrolmentDataDF, eventsEnrolmentDataDF):
        userEventCertificatesDF = eventsEnrolmentDataDF \
            .filter((F.col("completed_on_datetime") >= F.lit(default_start)) & (
                    F.col("completed_on_datetime") <= F.lit(default_end))) \
            .filter(F.col("certificate_id").isNotNull()).groupBy("user_id") \
            .agg(F.countDistinct("certificate_id").alias("event_certificate_count"))
        userContentCertificatesDF = contentEnrolmentDataDF \
            .filter(
            (F.col("first_completed_on") >= F.lit(default_start)) & (F.col("first_completed_on") <= F.lit(default_end))) \
            .filter(F.col("certificate_id").isNotNull()).groupBy("user_id") \
            .agg(F.count(F.lit(1)).alias("content_certificate_count"))

        eventsDF = spark.read.parquet(ParquetFileConstants.EVENT_PARQUET_FILE).withColumnRenamed("duration",
                                                                                                 "event_complete_duration")

        userEventLearningHoursDF = eventsEnrolmentDataDF \
            .filter((F.col("completed_on_datetime") >= F.lit(default_start)) & (
                    F.col("completed_on_datetime") <= F.lit(default_end))) \
            .filter(F.col("certificate_id").isNotNull()).join(eventsDF, on=["event_id"], how="left") \
            .withColumn("event_duration_hours", self.duration_to_hours("event_complete_duration")) \
            .groupBy("user_id") \
            .agg(F.sum(F.coalesce(F.col("event_duration_hours"), F.lit(0.0))).alias("event_learning_hours"))

        contentDF = spark.read.parquet(ParquetFileConstants.CONTENT_WAREHOUSE_COMPUTED_PARQUET_FILE) \
            .filter(F.col("content_sub_type").isin("Course", "Moderated Course"))

        userContentLearningHoursDF = contentEnrolmentDataDF \
            .filter(
            (F.col("first_completed_on") >= F.lit(default_start)) & (F.col("first_completed_on") <= F.lit(default_end))) \
            .filter(F.col("certificate_id").isNotNull()).join(contentDF, on=["content_id"], how="inner") \
            .withColumn("content_duration_hours", self.duration_to_hours("content_duration")) \
            .groupBy("user_id") \
            .agg(F.sum(F.coalesce(F.col("content_duration_hours"), F.lit(0.0))).alias("content_learning_hours"))

        userTotalCertificatesDF = userEventCertificatesDF.join(userContentCertificatesDF, on=["user_id"], how="full_outer") \
            .select(F.col("user_id"),
                    (F.coalesce(F.col("event_certificate_count"), F.lit(0)) +
                     F.coalesce(F.col("content_certificate_count"), F.lit(0))).alias("total_certificates"))

        userTotalLearningHoursDF = userEventLearningHoursDF.join(userContentLearningHoursDF, on=["user_id"],
                                                                 how="full_outer") \
            .select( \
            F.col("user_id"), \
            F.coalesce(F.col("event_learning_hours"), F.lit(0.0)).alias("event_learning_hours"), \
            F.coalesce(F.col("content_learning_hours"), F.lit(0.0)).alias("content_learning_hours"), \
            F.round( \
                F.coalesce(F.col("event_learning_hours"), F.lit(0.0)) + \
                F.coalesce(F.col("content_learning_hours"), F.lit(0.0)), 2 \
                ).alias("total_learning_hours"))

        karmaPointsDataDF = spark.read.parquet(ParquetFileConstants.USER_KARMA_POINTS_PARQUET_FILE) \
            .filter((F.col("credit_date") >= F.lit(default_start)) & (F.col("credit_date") <= F.lit(default_end))) \
            .groupBy(F.col("userid")) \
            .agg( \
            F.sum(F.col("points")).alias("total_points"), \
            F.max(F.col("credit_date")).alias("last_credit_date"))

        userOrgData = userDF.select(F.col("user_id").alias("userid"), \
                                    F.col("mdo_id").alias("org_id"), \
                                    F.col("fullName").alias("fullname"), \
                                    F.col("userOrgName").alias("org_name"), \
                                    F.col("designation"), \
                                    F.col("userProfileImgUrl").alias("profile_image"))

        userLeaderBoardDataDF = userOrgData.join(karmaPointsDataDF, on=["userid"], how="left") \
            .filter(F.col("org_id") != "") \
            .select( \
            userOrgData["userid"].alias("user_id"), \
            userOrgData["org_id"], \
            userOrgData["fullname"], \
            userOrgData["designation"], \
            userOrgData["org_name"], \
            userOrgData["profile_image"], \
            karmaPointsDataDF["total_points"], \
            karmaPointsDataDF["last_credit_date"])

        wRank = Window.partitionBy("org_id").orderBy(F.desc("total_points"))
        userLeaderBoardOrderedDataDF = userLeaderBoardDataDF.withColumn("rank", F.dense_rank().over(wRank))

        wRow = Window.partitionBy("org_id").orderBy(F.col("rank"), F.col("last_credit_date").asc())
        finalUserLeaderBoardDataDF = userLeaderBoardOrderedDataDF.withColumn("row_num", F.row_number().over(wRow))

        userStatsDF = userTotalCertificatesDF.join(userTotalLearningHoursDF, on=["user_id"], how="full_outer") \
            .select( \
            F.col("user_id"), \
            F.coalesce(F.col("total_certificates"), F.lit(0)).alias("count"), \
            F.coalesce(F.col("total_learning_hours"), F.lit(0.0)).alias("total_learning_hours"))

        userStatsDetailedDF = userStatsDF.join(finalUserLeaderBoardDataDF, on=["user_id"], how="right")

        selectedColUserLeaderboardDF = userStatsDetailedDF \
            .select( \
            F.col("user_id").alias("userid"), \
            F.col("org_id"), \
            F.col("fullname"), \
            F.col("designation"), \
            F.col("profile_image"), \
            F.coalesce(F.col("total_points"), F.lit(0)).alias("total_points"), \
            F.coalesce(F.col("rank"), F.lit(0)).alias("rank"), \
            F.col("row_num"), \
            F.col("count"), \
            F.col("total_learning_hours"), \
            F.col("last_credit_date")).dropDuplicates(["userid"])

        return selectedColUserLeaderboardDF


def create_spark_session_with_packages(config):
    # Set environment variables for PySpark to find packages
    os.environ[
        'PYSPARK_SUBMIT_ARGS'] = '--packages com.datastax.spark:spark-cassandra-connector_2.12:3.4.1,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0,org.postgresql:postgresql:42.6.0 pyspark-shell'

    spark = SparkSession.builder \
        .appName('NationalLearningWeekModel') \
        .master("local[*]") \
        .config("spark.executor.memory", '20g') \
        .config("spark.driver.memory", '18g') \
        .config("spark.executor.memoryFraction", '0.7') \
        .config("spark.storage.memoryFraction", '0.2') \
        .config("spark.storage.unrollFraction", "0.1") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.parquet.compression.codec", 'snappy') \
        .config("spark.sql.legacy.json.allowEmptyString.enabled", "true") \
        .config("spark.sql.caseSensitive", "true") \
        .config("spark.cassandra.connection.host", config.sparkCassandraConnectionHost) \
        .config("spark.cassandra.connection.port", '9042') \
        .config("spark.cassandra.output.batch.size.rows", '10000') \
        .config("spark.cassandra.connection.keepAliveMS", "60000") \
        .config("spark.cassandra.connection.timeoutMS", '30000') \
        .config("spark.cassandra.read.timeoutMS", '30000') \
        .config("es.nodes", config.sparkElasticsearchConnectionHost) \
        .config("es.port", config.sparkElasticsearchConnectionPort) \
        .config("es.index.auto.create", "false") \
        .config("es.nodes.wan.only", "true") \
        .config("es.nodes.discovery", "false") \
        .getOrCreate()

    return spark


def main():
    # Initialize Spark Session with optimized settings for caching
    config_dict = get_environment_config()
    config = create_config(config_dict)

    # Initialize Spark Session with optimizations from config
    spark = create_spark_session_with_packages(config)
    start_time = datetime.now()
    print(f"[START] NLW processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # You said no config available — we still pass one to match your signature
    config_dict = get_environment_config()
    config = create_config(config_dict)

    model = NationalLearningWeekModel()
    model.process_data(spark, config)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"[END] NLW completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total duration: {duration}")
    spark.stop()


if __name__ == "__main__":
    main()
