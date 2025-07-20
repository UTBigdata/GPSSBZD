import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.storage.StorageLevel
import java.util.concurrent.Executors
import scala.concurrent._
import scala.concurrent.duration._

object BusDataProcessor {
  def main(args: Array[String]): Unit = {
    val gpsPath = if (args.length > 0) args(0) else "hdfs://10.103.104.102:9000/wgr/bus_19-02-01/gps/*"
    val busLinePath = if (args.length > 1) args(1) else "hdfs://10.103.104.102:9000/wgr/bus_line.csv"
    val intermediateOutput = if (args.length > 2) args(2) else "hdfs://10.103.104.102:9000/wgr/bus_19-02-01/lines95_processed"
    val finalOutputBase = if (args.length > 3) args(3) else "hdfs://10.103.104.102:9000/wgr/bus_19-02-01/lines95"

    val spark = SparkSession.builder()
      .appName("BusDataProcessing")
      .getOrCreate()
      
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

    try {
      processFirstPart(spark, gpsPath, busLinePath, intermediateOutput)
      
      processLinesInParallel(spark, intermediateOutput, finalOutputBase)
      
    } finally {
      spark.stop()
    }
  }

  def processFirstPart(spark: SparkSession, gpsPath: String, busLinePath: String, outputPath: String): Unit = {
    import spark.implicits._
    
    val df = spark.read.format("csv")
      .option("header", true)
      .load(gpsPath)
      
    val busLine = spark.read.format("csv")
      .option("header", true)
      .option("delimiter", ";")
      .load(busLinePath)

    val uniqueLineNumbers = df.select("linenumber").distinct().as[String].collect()
    val broadcastUniqueLineNumbers = spark.sparkContext.broadcast(uniqueLineNumbers)

    val dfBroadcastFiltered = df.filter(col("linenumber").isin(broadcastUniqueLineNumbers.value: _*)).cache()

    val timeDifference = udf((time1: Timestamp, time2: Timestamp) => {
      Math.abs(time1.getTime - time2.getTime) / 1000 
    })
    
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
    
    val dfWithTimestamp = dfBroadcastFiltered
      .withColumn("time", to_timestamp($"t", "yy-MM-dd HH:mm:ss"))
      .orderBy(asc("id"), asc("time"))
    
    val windowSpec = Window.partitionBy($"id").orderBy($"time")
    
    val withTimes = dfWithTimestamp
      .withColumn("next_timestamp", lag($"time", 1).over(windowSpec))
      .withColumn("time_diff", 
        when($"next_timestamp".isNotNull && $"time".isNotNull, 
          timeDifference($"time", $"next_timestamp"))
        .otherwise(0L)
      )
    
    val times = withTimes.dropDuplicates(Seq("id", "linenumber", "lng", "lat"))
    val result = times.orderBy(col("id"), col("time").asc)
      .select("linenumber", "id", "lng", "lat", "t", "time_diff")
    
    val processedBusLine = busLine.withColumn("direction", 
      when(col("direction").isNull, 2).otherwise(col("direction"))
    
    val windowSpecId = Window.partitionBy($"id").orderBy($"t".cast("timestamp"))
    val dfWithRowNumber = result.withColumn("row_num", row_number().over(windowSpecId))
    
    val dfWithDirection = dfWithRowNumber
      .withColumn("direction", when($"row_num" === 1, 1).otherwise(0))
      .withColumn("position", when($"row_num" === 1, 1).otherwise(0))
      .drop("row_num")

    val busline1 = processedBusLine.filter($"direction" === 1 && $"position" === 1)
    val busline2 = processedBusLine.filter($"direction" === 2 && $"position" === 1)
    val busline2Ids = busline2.select("existLine_id").distinct()
    val matchedBusline1 = busline1.join(busline2Ids, Seq("existLine_id"), "inner")

    val joinedDF1 = dfWithDirection.as("u")
      .join(matchedBusline1.as("b"), $"u.linenumber" === $"b.existLine_id", "left_outer")
      .withColumn("distance", expr("fastHaversine(u.lat, u.lng, b.x, b.y)"))
      .withColumn("new_direction", when($"distance" < 100, 1).otherwise($"u.direction"))
      .withColumn("new_position", when($"distance" < 100, 1).otherwise($"u.position"))
      .select($"u.linenumber", $"u.id", $"u.lng", $"u.lat", $"u.t", $"u.time_diff", 
              $"new_direction".as("direction"), $"new_position".as("position"))

    val joinedDF2 = joinedDF1.as("u")
      .join(busline2.as("b"), $"u.linenumber" === $"b.existLine_id", "left_outer")
      .withColumn("distance", expr("fastHaversine(u.lat, u.lng, b.x, b.y)"))
      .withColumn("new_direction", when($"distance" < 100, 2).otherwise($"u.direction"))
      .withColumn("new_position", when($"distance" < 100, 2).otherwise($"u.position"))
      .select($"u.linenumber", $"u.id", $"u.lng", $"u.lat", $"u.t", $"u.time_diff", 
              $"new_direction".as("direction"), $"new_position".as("position"))

    val windowSpec3 = Window.partitionBy($"id").orderBy($"t".cast("timestamp"))
      .rowsBetween(Window.unboundedPreceding, Window.currentRow)

    val filledDF = joinedDF2.withColumn("filled_direction", 
      last(when($"direction" =!= 0, $"direction"), ignoreNulls = true).over(windowSpec3)
    )

    val percentileThresholds = filledDF.groupBy("id").agg(expr("percentile_approx(time_diff, 0.95)").as("threshold"))

    val intermediateDF = filledDF.join(percentileThresholds, Seq("id"), "left")
        intermediateDF.repartition(200).write.format("csv")
      .option("header", true)
      .save(outputPath)
      
    dfBroadcastFiltered.unpersist()
    broadcastUniqueLineNumbers.destroy()
  }

  def processLinesInParallel(spark: SparkSession, intermediatePath: String, outputBase: String): Unit = {
    import spark.implicits._
    
    val intermediateDF = spark.read.format("csv")
      .option("header", true)
      .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
      .load(intermediatePath)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    val lineNumbers = intermediateDF.select("linenumber").distinct().as[String].collect()
    val broadcastLineNumbers = spark.sparkContext.broadcast(lineNumbers)

    def processLine(linenumber: String): Unit = {
      try {
        val lineDF = intermediateDF.filter(col("linenumber") === linenumber)
          .persist(StorageLevel.MEMORY_AND_DISK_SER)

        try {
          val windowSpec = Window.partitionBy("id").orderBy("t")

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
            .transform { df =>
              val filteredIDs = df
                .groupBy("id")
                .agg(
                  max(when(
                    (col("direction") === 1 && col("position") === 1) || 
                    (col("direction") === 1 && col("position") === 0 && col("lag") === 1, 1)
                    .otherwise(0)).as("condition1"),
                  max(when(
                    (col("direction") === 2 && col("position") === 2) || 
                    (col("direction") === 2 && col("position") === 0 && col("lag") === 1, 1)
                    .otherwise(0)).as("condition2")
                )
                .filter("condition1 = 1 AND condition2 = 1")
                .select("id")
              
              df.join(filteredIDs, Seq("id"), "inner")
            }
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

          val savePath = s"$outputBase/line_${linenumber}"
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

    val executorService = Executors.newFixedThreadPool(math.max(4, Runtime.getRuntime.availableProcessors() * 3 / 4))
    implicit val executionContext = ExecutionContext.fromExecutorService(executorService)

    try {
      val futures = broadcastLineNumbers.value.map { lineNumber =>
        Future {
          processLine(lineNumber)
        }.recover {
          case e: Exception =>
            println(s"Failed to process line $lineNumber: ${e.getMessage}")
        }
      }

      Await.result(Future.sequence(futures.toSeq), Duration.Inf)
    } finally {
      executorService.shutdown()
      intermediateDF.unpersist()
      broadcastLineNumbers.destroy()
    }
  }
}
