import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import scala.math._

object BusSignalProcessor {
  def main(args: Array[String]): Unit = {
    // 参数解析
    val inputPath = if (args.length > 0) args(0) else "hdfs://10.29.74.178:9000/lines95_clusters/*"
    val outputPath = if (args.length > 1) args(1) else "hdfs://10.29.74.178:9000/lines95_signal"

    val spark = SparkSession.builder()
      .appName("BusSignalProcessing")
      .getOrCreate()
      
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

    try {
      // 处理数据
      val resultDF = processData(spark, inputPath)
      
      // 保存结果
      resultDF.repartition(1).write
        .option("header", "true")
        .csv(outputPath)
    } finally {
      spark.stop()
    }
  }

  // Haversine距离计算函数
  def haversineDistance(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double = {
    val R = 6371e3
    val φ1 = lat1.toRadians
    val φ2 = lat2.toRadians
    val Δφ = (lat2 - lat1).toRadians
    val Δλ = (lon2 - lon1).toRadians
    val a = pow(sin(Δφ/2), 2) + cos(φ1) * cos(φ2) * pow(sin(Δλ/2), 2)
    val c = 2 * atan2(sqrt(a), sqrt(1-a))
    R * c
  }

  // 数据处理逻辑
  def processData(spark: SparkSession, inputPath: String): DataFrame = {
    import spark.implicits._
    
    // 注册Haversine UDF
    val haversineUDF = udf(haversineDistance _)
    
    // 1. 读取数据
    val df = spark.read.option("header", "true").csv(inputPath)
    
    // 2. 过滤掉cluster为-1的记录
    val filteredDF = df.filter(col("cluster") =!= "-1")
    
    // 3. 计算总的id出现次数和阈值
    val totalCount = filteredDF.select("id").distinct().count()
    val threshold = (0.75 * totalCount).toInt
    
    // 4. 筛选符合条件的cluster
    val qualifiedClusters = filteredDF.groupBy("cluster")
      .agg(countDistinct("id").alias("id_count"))
      .filter(col("id_count") >= threshold)
      .select("cluster")
    
    // 5. 保留这些cluster对应的所有数据
    val qualifiedDF = filteredDF.join(qualifiedClusters, Seq("cluster"), "inner")
    
    // 6. 转换为时间戳并排序
    val dfWithTimestamp = qualifiedDF
      .withColumn("timestamp", to_timestamp(col("t"), "yyyy-MM-dd HH:mm:ss"))
      .orderBy("id", "patternID", "timestamp")
    
    // 8. 计算时间差和距离
    val windowSpec = Window.partitionBy("id", "patternID").orderBy("timestamp")
    
    val dfWithDiff = dfWithTimestamp
      .withColumn("next_timestamp", lead(col("timestamp"), 1).over(windowSpec))
      .withColumn("next_lat", lead(col("lat"), 1).over(windowSpec))
      .withColumn("next_lng", lead(col("lng"), 1).over(windowSpec))
      .withColumn("time_diff", 
        when(col("next_timestamp").isNotNull, 
          unix_timestamp(col("next_timestamp")) - unix_timestamp(col("timestamp")))
          .otherwise(lit(0)))
      .withColumn("distance",
        when(col("next_lat").isNotNull && col("next_lng").isNotNull,
          haversineUDF(col("lat"), col("lng"), col("next_lat"), col("next_lng")))
          .otherwise(lit(0.0)))
    
    // 9. 计算速度
    val dfWithSpeed = dfWithDiff
      .withColumn("speed",
        when(col("time_diff") === 0, lit(0.0))
          .otherwise(col("distance") / col("time_diff")))
      .drop("next_timestamp", "next_lat", "next_lng", "time_diff", "distance")
    
    // 10. 信号标记处理
    processSignal(dfWithSpeed)
  }
  
  // 信号处理逻辑
  def processSignal(df: DataFrame): DataFrame = {
    // Step 1: 标记每组中speed最大的记录为0.5
    val windowSpeed = Window.partitionBy("cluster", "id", "patternID").orderBy(col("speed").desc)
    val dfWithSignal1 = df
      .withColumn("rn", row_number().over(windowSpeed))
      .withColumn("signal", when(col("rn") === 1, 0.5).otherwise(lit(null)))
      .drop("rn")
    
    // Step 2: 检查每个cluster,id组是否存在signal为0.5且speed>10的数据
    val checkCondition1 = dfWithSignal1
      .groupBy("cluster", "id")
      .agg(
        max(
          when(col("signal") === 0.5 && col("speed") > 10, 1).otherwise(0)
        ).alias("has_valid_signal")
      )
      .groupBy("cluster")
      .agg(
        min(col("has_valid_signal")).alias("all_have_valid_signal")
      )
    
    val dfWithSignal2 = dfWithSignal1
      .join(checkCondition1, Seq("cluster"), "left")
      .withColumn("signal",
        when(col("signal") === 0.5 && col("all_have_valid_signal") === 1, 0.75)
          .otherwise(col("signal")))
      .drop("all_have_valid_signal")
    
    // Step 3: 检查是否每组数据都存在signal为0.75的数据
    val checkCondition2 = dfWithSignal2
      .groupBy("cluster", "id", "patternID")
      .agg(
        max(
          when(col("signal") === 0.75, 1)
          .otherwise(0)
        ).alias("has_075_signal")
      )
      .groupBy("cluster")
      .agg(
        min("has_075_signal").alias("all_have_075_signal")
      )
    
    val dfWithSignal3 = dfWithSignal2
      .join(checkCondition2, Seq("cluster"), "left")
      .withColumn("signal",
        when(col("signal") === 0.75 && col("all_have_075_signal") === 1, 1)
          .otherwise(col("signal")))
      .drop("all_have_075_signal")
    
    // Step 4: 传播signal标记
    val windowTime = Window.partitionBy("cluster", "id", "patternID").orderBy("timestamp")
    val finalDF = dfWithSignal3
      .withColumn("prev_signal", lag(col("signal"), 1).over(windowTime))
      .withColumn("signal",
        when(col("prev_signal") === 0.5 && col("signal").isNull, 0.5)
          .when(col("prev_signal") === 0.75 && col("signal").isNull, 0.75)
          .when(col("prev_signal") === 1 && col("signal").isNull, 1)
          .otherwise(col("signal")))
      .drop("prev_signal", "timestamp")
      .orderBy("id", "t")
    
    finalDF
  }
}
