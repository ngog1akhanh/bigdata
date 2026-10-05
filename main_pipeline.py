from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# ==============================================================================
# 1. KHỞI TẠO SPARK SESSION & CẤU HÌNH TỐI ƯU HIỆU NĂNG
# ==============================================================================
def create_spark_session(app_name="NOAA_SEA_Weather_Cleaning_Pipeline"):
    """
    Khởi tạo SparkSession tối ưu hóa bộ nhớ và giảm overhead do Shuffle.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

# Khai báo Schema cố định nhằm tránh dùng inferSchema (quét file tốn thời gian)
NOAA_SCHEMA = StructType([
    StructField("STATION_ID", StringType(), True),
    StructField("DATE_STR", StringType(), True),
    StructField("ELEMENT", StringType(), True),
    StructField("DATA_VALUE", FloatType(), True),
    StructField("M_FLAG", StringType(), True),
    StructField("Q_FLAG", StringType(), True),
    StructField("S_FLAG", StringType(), True),
    StructField("OBS_TIME", StringType(), True)
])

# ==============================================================================
# 2. CÁC MODULE LÀM SẠCH (ĐỒNG BỘ CÙNG THÀNH VIÊN 2, 3, 4)
# ==============================================================================

def module1_schema_and_dedup(df):
    """
    Thành viên 2 đảm nhận:
    - Ép kiểu chuỗi ngày YYYYMMDD sang Date chuẩn.
    - Xóa trùng lặp dựa trên khóa chính (STATION_ID, DATE, ELEMENT).
    """
    df_formatted = df.withColumn("DATE", F.to_date(F.col("DATE_STR"), "yyyyMMdd"))
    df_dedup = df_formatted.dropDuplicates(["STATION_ID", "DATE", "ELEMENT"])
    return df_dedup


def module2_flags_and_missing(df):
    """
    Thành viên 3 đảm nhận:
    - Lọc bỏ bản ghi vi phạm cờ chất lượng Q_FLAG.
    - Loại bỏ các dòng khuyết thiếu (Null) ở cột giá trị.
    """
    df_valid_flags = df.filter(
        (F.col("Q_FLAG").isNull()) | 
        (F.col("Q_FLAG") == "") | 
        (F.col("Q_FLAG") == " ")
    )
    df_clean_missing = df_valid_flags.filter(F.col("DATA_VALUE").isNotNull())
    return df_clean_missing


def module3_units_and_outliers(df):
    """
    Thành viên 4 đảm nhận:
    - Quy đổi đơn vị (chia 10 để ra °C, mm).
    - Lọc bỏ số liệu bất thường (Outliers) cho vùng Đông Nam Á.
    - Trích xuất năm (YEAR) làm khóa phân mảnh storage.
    """
    df_scaled = df.withColumn("VALUE_CLEAN", F.col("DATA_VALUE") / 10.0)
    
    # Giới hạn ngưỡng khí hậu Đông Nam Á (Nhiệt độ: 0 - 50°C, Lượng mưa: >= 0mm)
    df_no_outliers = df_scaled.filter(
        ((F.col("ELEMENT").isin("TMAX", "TMIN", "TAVG")) & 
         (F.col("VALUE_CLEAN") >= 0.0) & 
         (F.col("VALUE_CLEAN") <= 50.0)) |
        ((F.col("ELEMENT") == "PRCP") & 
         (F.col("VALUE_CLEAN") >= 0.0)) |
        (~F.col("ELEMENT").isin("TMAX", "TMIN", "TAVG", "PRCP"))
    )
    
    df_final = df_no_outliers.withColumn("YEAR", F.year(F.col("DATE")))
    return df_final

# ==============================================================================
# 3. LUỒNG ĐIỀU HÀNH CHÍNH (MASTER PIPELINE EXECUTION)
# ==============================================================================

def execute_pipeline(input_path: str, output_path: str):
    """
    Luồng chạy tự động tích hợp 3 module và ghi nhận bộ dữ liệu sạch.
    """
    print("🚀 [1/5] Đang khởi tạo Spark Session...")
    spark = create_spark_session()
    
    print(f"📥 [2/5] Đọc dữ liệu thô từ: {input_path}")
    raw_df = spark.read.csv(input_path, schema=NOAA_SCHEMA, header=False)
    
    print("⚙️ [3/5] Thực thi Module 1: Chuẩn hóa ngày & Xóa trùng lặp...")
    m1_df = module1_schema_and_dedup(raw_df)
    
    # Tối ưu Bottleneck: Cache dữ liệu đã loại trùng lặp vào RAM
    m1_df.cache()
    
    print("⚙️ [4/5] Thực thi Module 2 & 3: Lọc cờ lỗi, xử lý Null & Outliers...")
    m2_df = module2_flags_and_missing(m1_df)
    final_df = module3_units_and_outliers(m2_df)
    
    print(f"💾 [5/5] Xuất bộ dữ liệu sạch ra Parquet (PartitionBy YEAR) tại: {output_path}")
    final_df.write \
        .mode("overwrite") \
        .partitionBy("YEAR") \
        .parquet(output_path)
        
    print("✅ HOÀN THÀNH PIPELINE TỰ ĐỘNG!")
    return final_df

# ==============================================================================
# 4. ĐIỂM CHẠY CHƯƠNG TRÌNH (ENTRY POINT)
# ==============================================================================
if __name__ == "__main__":
    INPUT_CSV = "sample_noaa_raw.csv"
    OUTPUT_PARQUET = "cleaned_weather_data.parquet"
    
    cleaned_df = execute_pipeline(INPUT_CSV, OUTPUT_PARQUET)
    
    print("\n--- MẪU 5 DÒNG DỮ LIỆU SẠCH ĐẦU RA ---")
    cleaned_df.select("STATION_ID", "DATE", "ELEMENT", "VALUE_CLEAN", "YEAR").show(5)