from pyspark.sql import SparkSession, functions as F  # Import SparkSession to create Spark app, and functions as F to use Spark SQL functions
import os, shutil, glob  # OS for paths, shutil for directory operations, glob for finding files by pattern


# 1️ Initialize Spark

spark = (
    SparkSession.builder     # Start building a new Spark session
    .appName("MediaAppAnalytics")  # Name of the Spark application (appears in Spark UI)
    .master("local[*]")    # Run Spark locally using all available CPU cores ('*')
    .getOrCreate()      # Create a new Spark session or reuse if one exists
)
spark.sparkContext.setLogLevel("WARN")    # Reduce Spark log verbosity to show only warnings or above
print(" Spark session started successfully")   # Confirmation message that Spark is ready


# 2️ Define Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # Get absolute path of current script's directory
OUTPUT_DIR = os.path.join(BASE_DIR, "output")      # Define an output folder path under the same directory
os.makedirs(OUTPUT_DIR, exist_ok=True)       # Create the output folder if it doesn’t exist already


# 3️ Load all CSVs

def load_csv(name):
    path = os.path.join(BASE_DIR, f"{name}.csv")   # Build absolute path for the given CSV name
    df = spark.read.option("header", True).csv(path, inferSchema=True)     # Read CSV into a DataFrame, with header=True and automatic type inference
    print(f" Loaded {name}.csv with {df.count()} rows and {len(df.columns)} columns")  # Print how many rows and columns were read (count() triggers Spark job)
    return df    # Return the loaded DataFrame

#  Load all datasets into Spark DataFrames

users_df = load_csv("users")      # Read users.csv into users_df
subs_df = load_csv("subscriptions")   # Read subscriptions.csv into subs_df
transactions_df = load_csv("transactions")    # Read transactions.csv into transactions_df
priceplans_df = load_csv("price_plans")    # Read price_plans.csv into priceplans_df


# 4️ Parse timestamps and numeric types

def cast_timestamps(df):
     # Loop through all columns of the DataFrame
    for c in df.columns:
        # If column name contains "timestamp"
        if "timestamp" in c.lower():
            # Convert it to proper TimestampType using to_timestamp()
            df = df.withColumn(c, F.to_timestamp(F.col(c)))
    return df    # Return updated DataFrame

# Apply timestamp casting to all datasets that may have timestamps

subs_df = cast_timestamps(subs_df)
transactions_df = cast_timestamps(transactions_df)
users_df = cast_timestamps(users_df)

# Convert numeric string columns into double type for arithmetic operations

transactions_df = transactions_df.withColumn("transaction_amount", F.col("transaction_amount").cast("double"))
priceplans_df = priceplans_df.withColumn("price", F.col("price").cast("double"))

print(" Data types normalized and timestamps parsed")


# 5️ Create user_subscription_summary

user_subscription_summary = (
    subs_df
    .join(users_df, ["user_id"], "left")   # Join subscriptions with users by user_id (left join keeps all subs)
    .join(priceplans_df, ["price_plan_id"], "left")   # Join price plans to add plan info to subscriptions
    .withColumn(                            # Create a derived column called 'is_active'
        "is_active",
        F.when(F.col("subs_cancel_timestamp").isNull(), F.lit(True)).otherwise(F.lit(False))  # True if not cancelled False if cancelled
    )
    .select(                                                    # Choose only relevant columns for summary
        "user_id", "city", "price_plan_name", "price_plan_type",
        "subs_id", "subs_start_timestamp", "subs_end_timestamp",
        "subs_cancel_timestamp", "is_active"
    )
)
print(" Created user_subscription_summary")       # Log info that summary DataFrame is ready



# 6️  Create subscription_transactions_summary (robust)

# Create lowercase-to-original mapping for transaction DataFrame columns

txn_cols = {c.lower(): c for c in transactions_df.columns}

# Helper function to find existing column names from list of candidate names

def find_col(candidates):
    for c in candidates:          # Iterate through possible column name variations
        if c.lower() in txn_cols:   # If found in DataFrame
            return txn_cols[c.lower()]  # Return the actual column name
    return None                # Return None if none matched
# Dynamically detect real column names 
transaction_id_col = find_col(["transaction_id", "txn_id"])
transaction_status_col = find_col(["transaction_status", "txn_status", "status"])
transaction_amount_col = find_col(["transaction_amount", "amount", "txn_amount"])
txn_created_col = find_col(["txn_created_timestamp", "created_timestamp", "txn_created_at"])
transaction_ts_col = find_col(["transaction_timestamp", "txn_timestamp", "timestamp"])
# Print detected mappings to console for verification
print("\n🔍 Transaction Column Mapping:")
print(f"  id={transaction_id_col}, status={transaction_status_col}, amount={transaction_amount_col}, created={txn_created_col}, ts={transaction_ts_col}")

# Build list of columns to select dynamically
select_cols = ["user_id", "subs_id", "price_plan_name"]

# Append transaction-related columns only if they exist
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

   # Join DataFrames to build 'subscription_transactions_summary'

subscription_transactions_summary = (
    transactions_df
    .join(subs_df, ["user_id", "subs_id"], "left")
    .join(priceplans_df, ["price_plan_id"], "left")
    .select(*select_cols)
)
# Count total rows (action) to confirm join success
print(f" Created subscription_transactions_summary with {subscription_transactions_summary.count()} rows")



# 7️ Write Outputs

def write_csv(df, folder, filename):
    out_path = os.path.join(OUTPUT_DIR, folder)  # Create output folder path
    if os.path.exists(out_path):    # If already exists
        shutil.rmtree(out_path)   # Delete it for a clean overwrite
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_path) # Combine partitions into a single CSV file
    part = glob.glob(f"{out_path}/part-*.csv")[0]  # Locate the actual part-*.csv file generated by Spark
    os.rename(part, os.path.join(out_path, filename)) # Rename it to the output filename
    print(f" Saved {filename}")  # Log info after saving

write_csv(user_subscription_summary, "user_subscription_summary", "user_subscription_summary.csv")
write_csv(subscription_transactions_summary, "subscription_transactions_summary", "subscription_transactions_summary.csv")

# 8️ Additional Analytics (same as Databricks)

print("\n Running summary analytics...")   # Log start of analytics section

# a) Daily new users
daily_new_users = (
    users_df.groupBy(F.to_date("created_timestamp").alias("signup_date")) # Group by signup date
    .agg(F.countDistinct("user_id").alias("new_users"))  # Count distinct users per day
    .orderBy("signup_date")   # Sort by date
)
daily_new_users.show(5)   # Display top 5 results

# b) Daily free trials
daily_free_trials = (
    subs_df.join(priceplans_df, "price_plan_id")  # Join subs with price plans
    .filter(F.col("price_plan_type") == "one_time")  # Filter for one-time plans
    .groupBy(F.to_date("subs_start_timestamp").alias("date"))  # Group by start date
    .agg(F.countDistinct("user_id").alias("free_trial_users"))  # Count unique users per day
    .orderBy("date")
)
daily_free_trials.show(5)  # Display first 5 rows

# c) Daily paid conversions
daily_paid_conversions = (
    subs_df.join(priceplans_df, "price_plan_id")  # Join with plans 
    .filter(F.col("price_plan_type") == "recurring") # Filter recurring plans (paid conversions)
    .groupBy(F.to_date("subs_start_timestamp").alias("date")) # Group by subscription start date
    .agg(F.countDistinct("user_id").alias("paid_conversions")) # Count distinct users converted to paid
    .orderBy("date")          # Sort ascending by date
)
daily_paid_conversions.show(5)

# d) Daily cancellations
daily_cancellations = (
    subs_df.filter(F.col("subs_cancel_timestamp").isNotNull()) # Filter only cancelled subscriptions
    .groupBy(F.to_date("subs_cancel_timestamp").alias("cancel_date")) # Group by cancellation date
    .agg(F.countDistinct("user_id").alias("cancelled_users"))  # Count distinct cancelled users
    .orderBy("cancel_date")    # Sort ascending
)
daily_cancellations.show(5)

# 9️ Stop Spark

spark.stop()  # Gracefully shuts down Spark and releases resources
print("\n Spark job completed successfully!")  # Log final completion message
