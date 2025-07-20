import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.storage.StorageLevel

object BusLineProcessor {
  def main(args: Array[String]): Unit = {
    // 参数解析
    val inputPath = if (args.length > 0) args(0) else "hdfs://10.29.74.178:9000/gps/*"
    val outputBase = if (args.length > 1) args(1) else "hdfs://10.29.74.178:9000/wgr/lines95"
    val targetLines = if (args.length > 2) args(2).split(",").toSeq else Seq(
      "00E42", "03060", "03160", "06150", "M2623", "M3103", "M3153", "M3963", "M3983", "M5203",
      "0E382", "00640", "00620", "00660", "00680", "00850", "01010", "01030", "01130", "00385",
      "02030", "02040", "02050", "00E72", "02260", "02290", "03030", "03080", "08330", "03310",
      "03220", "03240", "03260", "03340", "03370", "03380", "03760", "03930", "07690", "B9614"
    )

    val spark = SparkSession.builder()
      .appName("BusLineProcessing")
      .getOrCreate()
      
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

    try {
      // 注册Haversine UDF
      spark.udf.register("fastHaversine", (lat1: Double, lon1: Double, lat2: Double, lon2: Double) => {
        val R = 6371000
        val dLat = math.toRadians(lat2 - lat1)
        val dLon = math.toRadians(lon2 - lon1)
        val a = math.sin(dLat/2) * math.sin(dLat/2) + 
                math.cos(math.toRadians(lat1)) * math.cos(math.toRadians(lat2)) * 
                math.sin(dLon/2) * math.sin(dLon/2)
        val c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        R * c
      })
      
      // 加载并缓存中间数据
      val intermediateDF = spark.read.format("csv")
        .option("header", true)
        .load(inputPath)
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
      
      // 顺序处理每条线路
      targetLines.foreach { lineNumber =>
        processLine(lineNumber, intermediateDF, outputBase)
      }
      
      // 清理缓存
      intermediateDF.unpersist()
      
    } finally {
      spark.stop()
    }
  }

  // 线路处理函数
  def processLine(linenumber: String, intermediateDF: DataFrame, outputBase: String): Unit = {
    println(s"Starting processing line: $linenumber")
    
    try {
      // 过滤当前线路数据并缓存
      val lineDF = intermediateDF.filter(col("linenumber") === linenumber)
        .persist(StorageLevel.MEMORY_AND_DISK_SER)

      try {
        val windowSpec = Window.partitionBy("id").orderBy("t")

        // 处理逻辑
        val resultDF = lineDF
          .withColumn("time_diff", col("time_diff").cast("double"))
          .withColumn("threshold", col("threshold").cast("double"))
          .withColumn("lag", expr("CASE WHEN time_diff >= threshold THEN 1 ELSE 0 END"))
          .withColumn("t", to_timestamp(col("t"), "yy-MM-dd HH:mm:ss"))
          .orderBy("id", "t")
          .withColumn("next_lag", lead("lag", 1).over(windowSpec))
          .withColumn("last_lag", lag("lag", 1).over(windowSpec))
          .withColumn("next_lat", lead("lat", 1).over(windowSpec))
          .withColumn("next_lng", lead("lng", 1).over(windowSpec))
          .withColumn("last_lat", lag("lat", 1).over(windowSpec))
          .withColumn("last_lng", lag("lng", 1).over(windowSpec))
          .withColumn("distance", expr("""
            CASE 
              WHEN lag = 2 AND next_lag = 1 THEN fastHaversine(lat, lng, next_lat, next_lng)
              WHEN lag = 1 AND last_lag = 2 THEN fastHaversine(last_lat, last_lng, lat, lng)
              WHEN lag = 1 AND next_lag = 1 THEN fastHaversine(last_lat, last_lng, lat, lng)
              WHEN lag = 1 AND last_lag = 1 THEN fastHaversine(last_lat, last_lng, lat, lng)
              ELSE NULL
            END
          """))
          .withColumn("new_lag", expr("""
            CASE 
              WHEN distance IS NOT NULL AND distance < (11.11 * (threshold - 1)) THEN 0
              ELSE lag
            END
          """))
          .withColumn("lag", expr("""
            CASE 
              WHEN (lag = 2 AND new_lag = 0) OR (lag = 1 AND new_lag = 0) THEN 0
              ELSE new_lag
            END
          """))
          // 筛选符合条件的ID
          .transform { df =>
            val filteredIDs = df
              .groupBy("id")
              .agg(
                max(when(
                  (col("direction") === 1 && col("position") === 1) || 
                  (col("direction") === 1 && col("position") === 0 && col("lag") === 1), 1)
                  .otherwise(0)).as("condition1"),
                max(when(
                  (col("direction") === 2 && col("position") === 2) || 
                  (col("direction") === 2 && col("position") === 0 && col("lag") === 1), 1)
                  .otherwise(0)).as("condition2")
              )
              .filter("condition1 = 1 AND condition2 = 1")
              .select("id")
              .persist(StorageLevel.MEMORY_AND_DISK_SER)
            
            val result = df.join(filteredIDs, Seq("id"), "inner")
            filteredIDs.unpersist()
            result
          }
          // 模式识别处理
          .withColumn("threshold", col("threshold").cast("int"))
          .withColumn("next_lag", lead("lag", 1).over(windowSpec))
          .withColumn("pattern_start", when(col("lag") === 2, 1).otherwise(0))
          .withColumn("pattern_id", sum("pattern_start").over(windowSpec))
          .withColumn("is_end_of_pattern", 
            when(col("lag") === 1 && (col("next_lag") === 0 || col("next_lag") === 2), 1).otherwise(0))
          .withColumn("pattern_end", when(col("is_end_of_pattern") === 1, col("t")).otherwise(null))
          .withColumn("pattern_lower_bound", 
            when(col("lag") === 2, from_unixtime(unix_timestamp($"t") - $"threshold")).otherwise(null))
          .withColumn("pattern_upper_bound", 
            when(col("pattern_end").isNotNull, from_unixtime(unix_timestamp($"pattern_end") + $"threshold")).otherwise(null))
          // 填充边界值
          .transform { df =>
            val windowForUpper = Window.partitionBy("id", "pattern_id").orderBy("t")
            val windowForLower = Window.partitionBy("id").orderBy("pattern_id").rowsBetween(1, Window.unboundedFollowing)
            
            df
              .withColumn("first_upper", first("pattern_upper_bound", ignoreNulls = true).over(windowForUpper))
              .withColumn("pattern_upper_bound", coalesce(col("pattern_upper_bound"), col("first_upper")))
              .drop("first_upper")
              .withColumn("next_lower", first("pattern_lower_bound", ignoreNulls = true).over(windowForLower))
              .withColumn("pattern_lower_bound", coalesce(col("pattern_lower_bound"), col("next_lower")))
              .drop("next_lower")
          }
          // 最终处理
          .withColumn("next_upper", lead("pattern_upper_bound", 1).over(windowSpec))
          .withColumn("pattern_upper_bound", 
            when(col("pattern_upper_bound").isNull && col("pattern_id") =!= 0, col("next_upper"))
              .otherwise(col("pattern_upper_bound")))
          .drop("next_upper")
          .withColumn("prev_pattern_id", lag("pattern_id", 1).over(windowSpec))
          .filter(
            (col("pattern_id") === 0 && col("t").gt(col("pattern_lower_bound"))) || 
            (col("pattern_id") > 0 && col("t").lt(col("pattern_upper_bound"))) ||
            (col("prev_pattern_id") > 0 && col("t").gt(col("pattern_lower_bound")))
          )
          .withColumn("test", expr("""
            CASE 
              WHEN pattern_upper_bound IS NULL OR lag = 2 THEN 1
              WHEN t > pattern_lower_bound AND t >= pattern_upper_bound THEN 1
              WHEN t < pattern_upper_bound AND t <= pattern_lower_bound THEN 0
              ELSE NULL
            END
          """))
          .withColumn("change_marker", 
            when(lag("test", 1).over(windowSpec) === 0 && col("test") === 1, 1).otherwise(0))
          .withColumn("patternID", sum("change_marker").over(windowSpec) + 1)
          .drop("change_marker", "pattern_id", "pattern_lower_bound", "pattern_upper_bound", 
                "prev_pattern_id", "test", "threshold", "lag", "is_end_of_pattern", "pattern_end")
          .withColumn("t", col("t").cast("string"))

        // 保存结果
        val savePath = s"$outputBase/line_$linenumber"
        resultDF.repartition(1).write.format("csv").option("header", true).save(savePath)
        
      } finally {
        lineDF.unpersist()
      }
    } catch {
      case e: Exception =>
        println(s"Error processing line $linenumber: ${e.getMessage}")
        e.printStackTrace()
    }
  }
}
