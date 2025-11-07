from pyspark.sql import SparkSession, functions as F
import os, shutil, glob

# ===========================================================
# 1️⃣ Initialize Spark
# ===========================================================
spark = (
    SparkSession.builder
    .appName("MediaAppAnalytics")
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("✅ Spark session started successfully")

# ===========================================================
# 2️⃣ Define Paths
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================
# 3️⃣ Load all CSVs
# ===========================================================
def load_csv(name):
    path = os.path.join(BASE_DIR, f"{name}.csv")
    df = spark.read.option("header", True).csv(path, inferSchema=True)
    print(f"✅ Loaded {name}.csv with {df.count()} rows and {len(df.columns)} columns")
    return df

users_df = load_csv("users")
subs_df = load_csv("subscriptions")
transactions_df = load_csv("transactions")
priceplans_df = load_csv("price_plans")

# ===========================================================
# 4️⃣ Parse timestamps and numeric types
# ===========================================================
def cast_timestamps(df):
    for c in df.columns:
        if "timestamp" in c.lower():
            df = df.withColumn(c, F.to_timestamp(F.col(c)))
    return df

subs_df = cast_timestamps(subs_df)
transactions_df = cast_timestamps(transactions_df)
users_df = cast_timestamps(users_df)

transactions_df = transactions_df.withColumn("transaction_amount", F.col("transaction_amount").cast("double"))
priceplans_df = priceplans_df.withColumn("price", F.col("price").cast("double"))

print("✅ Data types normalized and timestamps parsed")

# ===========================================================
# 5️⃣ Create user_subscription_summary
# ===========================================================
user_subscription_summary = (
    subs_df
    .join(users_df, ["user_id"], "left")
    .join(priceplans_df, ["price_plan_id"], "left")
    .withColumn(
        "is_active",
        F.when(F.col("subs_cancel_timestamp").isNull(), F.lit(True)).otherwise(F.lit(False))
    )
    .select(
        "user_id", "city", "price_plan_name", "price_plan_type",
        "subs_id", "subs_start_timestamp", "subs_end_timestamp",
        "subs_cancel_timestamp", "is_active"
    )
)
print("✅ Created user_subscription_summary")


# ===========================================================
# 6️⃣  Create subscription_transactions_summary (robust)
# ===========================================================
txn_cols = {c.lower(): c for c in transactions_df.columns}

def find_col(candidates):
    for c in candidates:
        if c.lower() in txn_cols:
            return txn_cols[c.lower()]
    return None

transaction_id_col = find_col(["transaction_id", "txn_id"])
transaction_status_col = find_col(["transaction_status", "txn_status", "status"])
transaction_amount_col = find_col(["transaction_amount", "amount", "txn_amount"])
txn_created_col = find_col(["txn_created_timestamp", "created_timestamp", "txn_created_at"])
transaction_ts_col = find_col(["transaction_timestamp", "txn_timestamp", "timestamp"])

print("\n🔍 Transaction Column Mapping:")
print(f"  id={transaction_id_col}, status={transaction_status_col}, amount={transaction_amount_col}, created={txn_created_col}, ts={transaction_ts_col}")

# ✅ Build select list dynamically — skip any missing columns
select_cols = ["user_id", "subs_id", "price_plan_name"]

if transaction_id_col:
    select_cols.append(F.col(transaction_id_col).alias("transaction_id"))
if transaction_status_col:
    select_cols.append(F.col(transaction_status_col).alias("transaction_status"))
if transaction_amount_col:
    select_cols.append(F.col(transaction_amount_col).alias("transaction_amount"))
if txn_created_col:
    select_cols.append(F.col(txn_created_col).alias("txn_created_timestamp"))
if transaction_ts_col:
    select_cols.append(F.col(transaction_ts_col).alias("transaction_timestamp"))

subscription_transactions_summary = (
    transactions_df
    .join(subs_df, ["user_id", "subs_id"], "left")
    .join(priceplans_df, ["price_plan_id"], "left")
    .select(*select_cols)
)

print(f"✅ Created subscription_transactions_summary with {subscription_transactions_summary.count()} rows")


# ===========================================================
# 7️⃣ Write Outputs
# ===========================================================
def write_csv(df, folder, filename):
    out_path = os.path.join(OUTPUT_DIR, folder)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_path)
    part = glob.glob(f"{out_path}/part-*.csv")[0]
    os.rename(part, os.path.join(out_path, filename))
    print(f"📝 Saved {filename}")

write_csv(user_subscription_summary, "user_subscription_summary", "user_subscription_summary.csv")
write_csv(subscription_transactions_summary, "subscription_transactions_summary", "subscription_transactions_summary.csv")

# ===========================================================
# 8️⃣ Additional Analytics (same as Databricks)
# ===========================================================
print("\n📈 Running summary analytics...")

# a) Daily new users
daily_new_users = (
    users_df.groupBy(F.to_date("created_timestamp").alias("signup_date"))
    .agg(F.countDistinct("user_id").alias("new_users"))
    .orderBy("signup_date")
)
daily_new_users.show(5)

# b) Daily free trials
daily_free_trials = (
    subs_df.join(priceplans_df, "price_plan_id")
    .filter(F.col("price_plan_type") == "one_time")
    .groupBy(F.to_date("subs_start_timestamp").alias("date"))
    .agg(F.countDistinct("user_id").alias("free_trial_users"))
    .orderBy("date")
)
daily_free_trials.show(5)

# c) Daily paid conversions
daily_paid_conversions = (
    subs_df.join(priceplans_df, "price_plan_id")
    .filter(F.col("price_plan_type") == "recurring")
    .groupBy(F.to_date("subs_start_timestamp").alias("date"))
    .agg(F.countDistinct("user_id").alias("paid_conversions"))
    .orderBy("date")
)
daily_paid_conversions.show(5)

# d) Daily cancellations
daily_cancellations = (
    subs_df.filter(F.col("subs_cancel_timestamp").isNotNull())
    .groupBy(F.to_date("subs_cancel_timestamp").alias("cancel_date"))
    .agg(F.countDistinct("user_id").alias("cancelled_users"))
    .orderBy("cancel_date")
)
daily_cancellations.show(5)

# ===========================================================
# 9️⃣ Stop Spark
# ===========================================================
spark.stop()
print("\n🏁 Spark job completed successfully!")
