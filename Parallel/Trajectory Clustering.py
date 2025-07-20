from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, broadcast, collect_list, struct, lit
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, FloatType, TimestampType
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from math import radians, sin, cos, sqrt, atan2
import time
from sklearn.metrics import silhouette_score
from pyspark import StorageLevel
from pyspark.sql import functions as F

# 初始化Spark会话
spark = SparkSession.builder \
    .master("spark://10.29.74.178:7077") \
    .appName("TrajectoryClustering") \
    .config("spark.executor.cores", "12") \
    .config("spark.executor.memory", "30g") \
    .config("spark.driver.memory", "10g") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.default.parallelism", "400") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 定义schema
input_schema = StructType([
    StructField("id", StringType(), True),
    StructField("lng", FloatType(), True),
    StructField("lat", FloatType(), True),
    StructField("t", StringType(), True),
    StructField("patternID", StringType(), True),
    StructField("linenumber", StringType(), True)
])

params_schema = StructType([
    StructField("linenumber", StringType(), True),
    StructField("eps", FloatType(), True),
    StructField("min_samples", IntegerType(), True)
])

# 加载参数
params_df = spark.read.csv('hdfs://10.29.74.178:9000/lines95_parameter.csv', 
                         header=True, schema=params_schema)
params_dict = {row['linenumber']: (row['eps'], row['min_samples']) for row in params_df.collect()}

# 读取数据
df = spark.read.csv("hdfs://10.103.104.102:9000//wgr/bus_19-02-01/95lines/95lines.csv", 
                   header=True, schema=input_schema)

# 数据预处理
df = df.withColumn('t', col('t').cast(TimestampType())) \
       .withColumn('lng', col('lng').cast(FloatType())) \
       .withColumn('lat', col('lat').cast(FloatType()))

# ==============================================
# 新增：优化分区策略（基于计算复杂度的动态分区）
# ==============================================

# 1. 预计算各线路的计算量 (N×S)
print("开始计算各线路复杂度...")
line_complexity = df.groupBy("linenumber").agg(
    F.countDistinct("id").alias("bus_count"),
    F.countDistinct("patternID").alias("pattern_count")
).withColumn(
    "complexity", 
    F.col("bus_count") * F.col("pattern_count")
).orderBy("complexity", ascending=False)

# 收集到Driver端并打印统计信息
line_complexity_list = line_complexity.collect()
total_complexity = sum(row['complexity'] for row in line_complexity_list)
avg_complexity = total_complexity / len(line_complexity_list) if len(line_complexity_list) > 0 else 0

print(f"\n线路复杂度统计:")
print(f"总线路数: {len(line_complexity_list)}")
print(f"总复杂度: {total_complexity}")
print(f"平均复杂度: {avg_complexity:.2f}")
print(f"最大复杂度: {max(row['complexity'] for row in line_complexity_list)}")
print(f"最小复杂度: {min(row['complexity'] for row in line_complexity_list)}")

# 2. 动态分区算法（装箱算法）
total_partitions = 60  # 保持与你原设置一致
target_complexity_per_partition = total_complexity / total_partitions
overflow_threshold = 1.2  # 允许20%的溢出

print(f"\n开始动态分区计算(目标分区数={total_partitions})...")
partitions = []
current_partition = []
current_sum = 0
big_lines = []

for line in line_complexity_list:
    line_complexity_value = line['complexity']
    line_name = line['linenumber']
    
    # 处理超大线路（单独分区）
    if line_complexity_value > 3 * avg_complexity:
        big_lines.append(line_name)
        partitions.append([line_name])
        continue
        
    if current_sum + line_complexity_value <= target_complexity_per_partition * overflow_threshold:
        current_partition.append(line_name)
        current_sum += line_complexity_value
    else:
        if current_partition:  # 避免空分区
            partitions.append(current_partition)
        current_partition = [line_name]
        current_sum = line_complexity_value

if current_partition:
    partitions.append(current_partition)

# 创建分区映射字典
partition_map = {}
for partition_id, line_group in enumerate(partitions):
    for line in line_group:
        partition_map[line] = partition_id

# 为超大线路分配额外分区
for i, line in enumerate(big_lines):
    partition_map[line] = total_partitions + i

final_partition_count = total_partitions + len(big_lines)

print(f"\n分区结果:")
print(f"实际创建分区数: {final_partition_count} (基础{total_partitions} + 超大线路{len(big_lines)})")
print(f"常规分区数: {len(partitions)}")
print(f"包含超大线路: {big_lines if big_lines else '无'}")

# 3. 应用分区策略
print("\n应用分区策略到原始数据...")
partition_udf = F.udf(lambda x: partition_map.get(x, 0), IntegerType())
df = df.withColumn("computed_partition", partition_udf("linenumber")) \
       .repartition(final_partition_count, "computed_partition")

# ==============================================
# 后续处理（保持你的原始逻辑，仅调整分区相关部分）
# ==============================================

# 将轨迹数据收集为数组（使用优化后的分区）
trajectories_df = df.groupBy("linenumber", "id", "patternID") \
                  .agg(collect_list(struct("lng", "lat")).alias("points")) \
                  .persist(StorageLevel.MEMORY_AND_DISK)

# 自定义距离计算函数（保持不变）
def calculate_trajectory_distance(traj1, traj2):
    """计算两条轨迹间的综合距离"""
    try:
         # 转换为坐标列表
        coords1 = [(p.lng, p.lat) for p in traj1]
        coords2 = [(p.lng, p.lat) for p in traj2]
        
        # 1. 计算水平距离
        def horizontal_distance(p1, p2, tr2):
            def project_point_on_line(p, line_start, line_end):
                if np.allclose(line_start, line_end):
                    return line_start
                p = np.array(p)
                line_start = np.array(line_start)
                line_end = np.array(line_end)
                v = line_end - line_start
                u = p - line_start
                t = np.dot(u, v) / np.dot(v, v)
                if t < 0: return line_start
                elif t > 1: return line_end
                else: return line_start + t * v
            
            proj_p1 = project_point_on_line(p1, tr2[0], tr2[-1])
            proj_p2 = project_point_on_line(p2, tr2[0], tr2[-1])
            dist1 = great_circle((p1[1], p1[0]), (proj_p1[1], proj_p1[0])).km
            dist2 = great_circle((p2[1], p2[0]), (proj_p2[1], proj_p2[0])).km
            return (dist1 + dist2) / 2.0
        
        d_horizontal = horizontal_distance(coords1[0], coords1[-1], coords2)
        
        # 2. 计算垂直距离
        def vertical_distance(p1, p2, tr2):
            # 使用与horizontal_distance相同的投影方法
            proj_p1 = project_point_on_line(p1, tr2[0], tr2[-1])
            proj_p2 = project_point_on_line(p2, tr2[0], tr2[-1])
            dist1 = great_circle((p1[1], p1[0]), (proj_p1[1], proj_p1[0])).km
            dist2 = great_circle((p2[1], p2[0]), (proj_p2[1], proj_p2[0])).km
            return (dist1 + dist2) / 2.0
        
        d_vertical = vertical_distance(coords1[0], coords1[-1], coords2)
        
        # 3. 计算角度距离
        def angular_distance(tr1, tr2):
            def angle_between_vectors(v1, v2):
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                return np.degrees(theta)
            
            v1 = np.array(tr1[-1]) - np.array(tr1[0])
            v2 = np.array(tr2[-1]) - np.array(tr2[0])
            
            if np.allclose(v1, 0) or np.allclose(v2, 0):
                theta = 90.0
            else:
                theta = angle_between_vectors(v1, v2)
            
            len_tr1 = sum(great_circle((tr1[i-1][1], tr1[i-1][0]), 
                          (tr1[i][1], tr1[i][0])).km for i in range(1, len(tr1)))
            len_tr2 = sum(great_circle((tr2[i-1][1], tr2[i-1][0]), 
                          (tr2[i][1], tr2[i][0])).km for i in range(1, len(tr2)))
            
            if 0 < theta <= 90:
                return min(len_tr1, len_tr2) * np.sin(np.radians(theta))
            elif 90 < theta <= 180:
                return max(len_tr1, len_tr2)
            else:
                return 0
        
        d_angular = angular_distance(coords1, coords2)
        
        # 综合距离
        return float(d_horizontal + d_vertical + d_angular)
    except:
        return float('inf')

# 注册UDF
spark.udf.register("trajectory_distance_udf", calculate_trajectory_distance, FloatType())

# 为每个linenumber创建轨迹对（使用优化后的分区）
trajectories_with_id = trajectories_df.withColumn("traj_id", 
    F.concat_ws("_", "id", "patternID"))

# 自连接创建轨迹对（分区优化版）
print("\n创建轨迹对（使用优化分区）...")
trajectory_pairs = trajectories_with_id.alias("t1").join(
    trajectories_with_id.alias("t2"),
    (F.col("t1.linenumber") == F.col("t2.linenumber")) & 
    (F.col("t1.traj_id") < F.col("t2.traj_id")),
    "inner"
).select(
    F.col("t1.linenumber").alias("linenumber"),
    F.col("t1.id").alias("id1"),
    F.col("t1.patternID").alias("patternID1"),
    F.col("t1.points").alias("points1"),
    F.col("t2.id").alias("id2"),
    F.col("t2.patternID").alias("patternID2"),
    F.col("t2.points").alias("points2")
)

# 计算距离（分布式执行）
distances_df = trajectory_pairs.withColumn(
    "distance",
    F.expr("trajectory_distance_udf(points1, points2)")
).select(
    "linenumber", "id1", "patternID1", "id2", "patternID2", "distance"
).persist(StorageLevel.MEMORY_AND_DISK)

# 聚类处理（保持不变）
@pandas_udf(
    StructType([
        StructField("id", StringType()),
        StructField("patternID", StringType()),
        StructField("cluster", IntegerType())
    ]),
    PandasUDFType.GROUPED_MAP
)
def cluster_trajectories_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    linenumber = pdf['linenumber'].iloc[0]
    
    if linenumber not in params_dict:
       return pd.DataFrame(columns=['id', 'patternID', 'cluster'])

    eps, min_samples = params_dict[linenumber]
    
    # 获取所有轨迹ID
    ids1 = pdf[['id1', 'patternID1']].rename(columns={'id1':'id', 'patternID1':'patternID'})
    ids2 = pdf[['id2', 'patternID2']].rename(columns={'id2':'id', 'patternID2':'patternID'})
    all_ids = pd.concat([ids1, ids2]).drop_duplicates()
    id_list = all_ids.apply(tuple, axis=1).tolist()
    num_trajectories = len(id_list)
    
    # 创建距离矩阵
    dist_matrix = np.full((num_trajectories, num_trajectories), float('inf'))
    np.fill_diagonal(dist_matrix, 0)
    
    # 填充距离矩阵
    id_to_idx = {id_tuple: idx for idx, id_tuple in enumerate(id_list)}
    
    for _, row in pdf.iterrows():
        id1 = (row['id1'], row['patternID1'])
        id2 = (row['id2'], row['patternID2'])
        if id1 in id_to_idx and id2 in id_to_idx:
            i = id_to_idx[id1]
            j = id_to_idx[id2]
            # 确保距离不是无限大
            distance = row['distance']
            if not np.isinf(distance):
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance
    
    # 处理无限大值 - 替换为一个大的有限值
    max_distance = np.nanmax(dist_matrix[~np.isinf(dist_matrix)])
    if np.isnan(max_distance) or max_distance <= 0:
        max_distance = eps * 10  # 默认值
    
    dist_matrix[np.isinf(dist_matrix)] = max_distance * 2
    
    # DBSCAN聚类
    try:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        clusters = db.fit_predict(dist_matrix)
    except Exception as e:
        print(f"聚类失败于线路 {linenumber}: {str(e)}")
        # 返回默认聚类结果(所有点为一个簇)
        clusters = np.zeros(num_trajectories, dtype=int)
    
    # 返回结果
    result = pd.DataFrame({
        'id': [x[0] for x in id_list],
        'patternID': [x[1] for x in id_list],
        'cluster': clusters
    })
    return result

# 执行聚类
cluster_results = distances_df.groupBy("linenumber").apply(cluster_trajectories_udf)

# 合并回原始数据
df_final = df.join(cluster_results, on=['id', 'patternID'], how='left')

# 保存结果
output_file = 'hdfs://10.29.74.178:9000/95lines_clusters'
df_final.write.csv(output_file, header=True, mode='overwrite')
print(f"\n文件已保存到: {output_file}")

# 清理缓存
trajectories_df.unpersist()
distances_df.unpersist()

# 停止Spark会话
spark.stop()
