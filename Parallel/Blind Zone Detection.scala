import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import scala.math._

object BusSignalProcessor {
  def main(args: Array[String]): Unit = {
    // 参数解析
    val inputPath = if (args.length > 0) args(0) else "hdfs://10.29.74.178:9000/95lines_clusters/*"
    val outputPath = if (args.length > 1) args(1) else "hdfs://10.29.74.178:9000/95lines_signal"

    val spark = SparkSession.builder()
      .appName("BusSignalProcessing")
      .getOrCreate()
      
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

    try {
      // 处理数据
      val resultDF = processData(spark, inputPath)
      
      // 保存结果
      resultDF.repartition(1).write.format("csv")
        .option("header", true)
        .save(outputPath)
    } finally {
      spark.stop()
    }
  }

  // 定义Haversine距离计算函数
  def haversineDistance(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double = {
    val R = 6371e3 // 地球半径，单位：米
    val φ1 = lat1.toRadians
    val φ2 = lat2.toRadians
    val Δφ = (lat2 - lat1).toRadians
    val Δλ = (lon2 - lon1).toRadians

    val a = pow(sin(Δφ / 2), 2) + cos(φ1) * cos(φ2) * pow(sin(Δλ / 2), 2)
    val c = 2 * atan2(sqrt(a), sqrt(1 - a))

    R * c // 距离，单位：米
  }

  // 数据处理逻辑
  def processData(spark: SparkSession, inputPath: String): DataFrame = {
    import spark.implicits._
    
    // 注册Haversine UDF
    val haversine = udf(haversineDistance _)
    
    // 读取数据
    val df = spark.read.format("csv")
      .option("header", true)
      .load(inputPath)
    
    // 计算车辆数量
    val vehicleCountByCluster = df.groupBy("cluster")
      .agg(countDistinct("id").alias("vehicle_count"))
    
    // 转换时间戳
    val dfWithTimestamp = df.withColumn("t", to_timestamp($"t", "yyyy-MM-dd HH:mm:ss"))
    
    // 创建窗口规格
    val windowSpec = Window.partitionBy("id", "patternID").orderBy("t")
    
    // 计算时间差
    val dfWithTimeDiff = dfWithTimestamp
      .withColumn("next_t", lead($"t", 1).over(windowSpec))
      .withColumn("time_diff",
        when(col("next_t").isNotNull,
             unix_timestamp($"next_t") - unix_timestamp($"t")
        ).otherwise(lit(0))
      )
    
    // 排序并添加下一位置信息
    val dfWithNextLocation = dfWithTimeDiff
      .select("id", "lng", "lat", "t", "patternID", "cluster", "time_diff")
      .orderBy(col("id"), col("t").asc)
      .withColumn("next_lng", lead($"lng", 1).over(windowSpec))
      .withColumn("next_lat", lead($"lat", 1).over(windowSpec))
    
    // 计算距离
    val dfWithDistance = dfWithNextLocation.withColumn("distance",
      when(col("next_lng").isNotNull,
           haversine(col("lat"), col("lng"), col("next_lat"), col("next_lng"))
      ).otherwise(lit(0.0))
    )
    
    // 过滤无效集群
    val result = dfWithDistance
      .select("id", "lng", "lat", "t", "patternID", "cluster", "time_diff", "distance")
      .orderBy(col("id"), col("t").asc)
      .filter($"cluster" !== -1)
    
    // 计算阈值
    val totalCount = result.select("id").distinct().count()
    val threshold = 0.75 * totalCount
    
    // 筛选符合条件的集群
    val qualifiedCluster = result.groupBy("cluster")
      .agg(countDistinct("id").alias("cluster_idcount"))
      .filter($"cluster_idcount" >= lit(threshold))
      .select("cluster")
    
    // 保留符合条件的集群数据
    val filteredResult = result.join(qualifiedCluster, Seq("cluster"), "inner")
    
    // 计算速度
    val finalDF = filteredResult
      .withColumn("speed", 
        when(col("time_diff") === 0, 0)
          .otherwise(col("distance") / col("time_diff"))
      )
      .drop("distance", "time_diff")
    
    // 信号处理步骤
    processSignal(finalDF)
  }
  
  // 信号处理逻辑
  def processSignal(df: DataFrame): DataFrame = {
    // Step 1: 找到每组中 speed 最大的记录，并设置 signal 列为 0.5
    val windowSpec1 = Window.partitionBy("cluster", "id", "patternID").orderBy(col("speed").desc)
    
    val withSignalDF = df
      .withColumn("rn", row_number().over(windowSpec1))
      .withColumn("signal", when(col("rn") === 1, 0.5).otherwise(lit(null)))
      .drop("rn")
    
    // Step 2: 检查每个 cluster, id 组是否存在 signal 为 0.5 且 speed > 10 的数据，并更新为 0.75
    val checkConditionDF = withSignalDF
      .groupBy("cluster", "id")
      .agg(max(when(col("signal") === 0.5 && col("speed") > 10, 1).otherwise(0)).alias("has_valid_signal"))
      .groupBy("cluster")
      .agg(min("has_valid_signal").alias("all_have_valid_signal"))
    
    val updatedTo075DF = withSignalDF
      .join(checkConditionDF, Seq("cluster"), "left")
      .withColumn("signal",
        when(col("signal") === 0.5 && col("all_have_valid_signal") === 1, 0.75)
          .otherwise(col("signal"))
      )
      .drop("all_have_valid_signal")
    
    // Step 3: 再按 cluster, id, patternID 分组，检查是否每组数据都存在 signal 为 0.75 的数据，并更新为 1
    val checkConditiondf = updatedTo075DF
      .groupBy("cluster", "id", "patternID")
      .agg(max(when(col("signal") === 0.75, 1).otherwise(0)).alias("has_075_signal"))
      .groupBy("cluster")
      .agg(min("has_075_signal").alias("all_have_075_signal"))
    
    val updatedTo1DF = updatedTo075DF
      .join(checkConditiondf, Seq("cluster"), "left")
      .withColumn("signal",
        when(col("signal") === 0.75 && col("all_have_075_signal") === 1, 1)
          .otherwise(col("signal"))
      )
      .drop("all_have_075_signal")
    
    // Step 4: 更新紧跟在 signal 为 0.5, 0.75, 1 后的记录的 signal 也为对应的值
    val windowSpecTime = Window.partitionBy("cluster", "id", "patternID").orderBy("t")
    
    val withLagDF = updatedTo1DF
      .withColumn("prev_signal", lag(col("signal"), 1).over(windowSpecTime))
    
    val finalUpdatedDF = withLagDF
      .withColumn("signal",
        when(col("prev_signal") === 0.5 && col("signal").isNull, 0.5)
          .when(col("prev_signal") === 0.75 && col("signal").isNull, 0.75)
          .when(col("prev_signal") === 1 && col("signal").isNull, 1)
          .otherwise(col("signal"))
      )
      .drop("prev_signal")
      .orderBy("id", "t")
    
    finalUpdatedDF
  }
}
