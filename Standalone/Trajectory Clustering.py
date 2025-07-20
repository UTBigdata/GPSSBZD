from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, broadcast
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, FloatType, TimestampType
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import silhouette_score
from itertools import combinations

# 初始化Spark会话，并配置资源
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("TrajectoryClustering") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

#定义 schema，确保 linenumber 是字符串类型
input_schema = StructType([
    StructField("id", StringType(), True),
    StructField("lng", FloatType(), True),
    StructField("lat", FloatType(), True),
    StructField("t", StringType(), True),
    StructField("patternID", StringType(), True),
    StructField("linenumber", StringType(), True)  # 确保 linenumber 为字符串类型
])

params_schema = StructType([
    StructField("linenumber", StringType(), True),
    StructField("eps", FloatType(), True),
    StructField("min_samples", IntegerType(), True)
])

# 加载参数配置文件
params_df = spark.read.csv('/home/lrr/wgr_data/lines99_parameter.csv', header=True, schema=params_schema)

# 创建一个字典来存储每个 linenumber 对应的 eps 和 min_samples 参数
params_dict = {row['linenumber']: (row['eps'], row['min_samples']) for row in params_df.collect()}

# 读取CSV文件到DataFrame
df = spark.read.csv('/home/data/40lines/lines95.csv', header=True, schema=input_schema)

# 确保有 'linenumber' 列用于分区
if 'linenumber' not in df.columns:
    raise KeyError("The DataFrame does not contain a 'linenumber' column.")

# 将时间字符串转换为datetime对象，并确保经纬度字段是浮点数类型
df = df.withColumn('t', col('t').cast(TimestampType())) \
       .withColumn('lng', col('lng').cast(FloatType())) \
       .withColumn('lat', col('lat').cast(FloatType()))

# Repartition the data based on 'linenumber'
num_partitions = len(params_dict)  # 假设每个 linenumber 对应一个分区
df = df.repartition(num_partitions, 'linenumber')

#计算最大圆距离
def great_circle_distance(a, b):
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    return great_circle((a[1], a[0]), (b[1], b[0])).km

# 计算投影点
def project_point_on_line(p, line_start, line_end):
    # 如果线段的起点和终点相同，则返回线段的任意一个端点
    if np.allclose(line_start, line_end):
        return line_start

    # 转换为笛卡尔坐标系
    p = np.array([p[0], p[1]])
    line_start = np.array([line_start[0], line_start[1]])
    line_end = np.array([line_end[0], line_end[1]])

    # 计算向量
    v = line_end - line_start
    u = p - line_start

    # 投影点的参数
    t = np.dot(u, v) / np.dot(v, v)

    # 如果投影点在直线外，则返回线段的端点
    if t < 0:
        return line_start
    elif t > 1:
        return line_end
    else:
        return line_start + t * v

# 计算水平距离
def horizontal_distance(p1, p2, tr2):
    proj_p1 = project_point_on_line(p1, tr2[0], tr2[-1])
    proj_p2 = project_point_on_line(p2, tr2[0], tr2[-1])
    dist1 = great_circle_distance(p1, proj_p1)
    dist2 = great_circle_distance(p2, proj_p2)
    if np.isnan(dist1) or np.isnan(dist2):
        return np.nan
    return (dist1 + dist2) / 2.0

# 计算垂直距离
def vertical_distance(p1, p2, tr2):
    proj_p1 = project_point_on_line(p1, tr2[0], tr2[-1])
    proj_p2 = project_point_on_line(p2, tr2[0], tr2[-1])
    dist1 = great_circle_distance(p1, proj_p1)
    dist2 = great_circle_distance(p2, proj_p2)
    if np.isnan(dist1) or np.isnan(dist2):
        return np.nan
    return (dist1 + dist2) / 2.0

# 计算角度距离
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

    len_tr1 = sum(great_circle_distance(tr1[i-1], tr1[i]) for i in range(1, len(tr1)))
    len_tr2 = sum(great_circle_distance(tr2[i-1], tr2[i]) for i in range(1, len(tr2)))

    if 0 < theta <= 90:
        return min(len_tr1, len_tr2) * np.sin(np.radians(theta))
    elif 90 < theta <= 180:
        return max(len_tr1, len_tr2)
    else:
        return 0

# 计算轨迹间的距离
def trajectory_distance(tr1, tr2):
    d_horizontal = horizontal_distance(tr1[0], tr1[-1], tr2)
    d_vertical = vertical_distance(tr1[0], tr1[-1], tr2)
    d_angular = angular_distance(tr1, tr2)
    if np.isnan(d_horizontal) or np.isnan(d_vertical):
        return np.nan
    return d_horizontal + d_vertical + d_angular
# 聚类逻辑的 Pandas UDF
@pandas_udf(
    StructType([
        StructField("id", StringType()),
        StructField("patternID", StringType()),
        StructField("cluster", IntegerType())
    ]),
    PandasUDFType.GROUPED_MAP
)
def cluster_trajectories_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    # 获取当前分区的 linenumber
    linenumber = pdf['linenumber'].iloc[0]
    
    # 检查是否有对应的 DBSCAN 参数，如果没有则跳过
    if linenumber not in params_dict:
        print(f"Warning: Skipping linenumber {linenumber} due to missing parameters.")
        return pd.DataFrame(columns=['id', 'patternID', 'cluster'])  # 返回空 DataFrame

    # 获取当前 linenumber 对应的 DBSCAN 参数
    eps, min_samples = params_dict[linenumber]  # 直接从字典中获取参数

    # 按id和patternID分组，形成轨迹列表
    trajectories_df = pdf.groupby(['id', 'patternID']).apply(
        lambda g: pd.Series({'points': list(zip(g['lng'], g['lat']))})
    ).reset_index()

    # 提取轨迹数据到一个Python列表中，并同时生成 group_keys
    trajectories_list = [row['points'] for _, row in trajectories_df.iterrows()]
    group_keys = [(row['id'], row['patternID']) for _, row in trajectories_df.iterrows()]

    # 计算距离矩阵
    num_trajectories = len(trajectories_list)
    distances = np.zeros((num_trajectories, num_trajectories))

    for i, j in combinations(range(num_trajectories), 2):
        distance = trajectory_distance(trajectories_list[i], trajectories_list[j])
        if not np.isnan(distance):
            distances[i, j] = distance
            distances[j, i] = distance

    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusters = db.fit_predict(distances)

    # 创建一个字典，用于存储每个(id, patternID)组合对应的聚类标签
    cluster_dict = dict(zip(group_keys, clusters))

    # 返回带有聚类标签的新 DataFrame
    return pd.DataFrame([(str(key[0]), str(key[1]), int(value)) for key, value in cluster_dict.items()],
                        columns=['id', 'patternID', 'cluster'])

# 只对有参数配置的linenumber进行分组并应用聚类逻辑
valid_linenumbers = set(params_dict.keys())
df_with_clusters = df.filter(col('linenumber').isin(valid_linenumbers)).groupBy('linenumber').apply(cluster_trajectories_udf)

# 缓存中间结果以提高性能
df_with_clusters.cache()

# 合并回原始数据框
df_final = df.join(df_with_clusters.select('id', 'patternID', 'cluster'), on=['id', 'patternID'], how='left')

# 将所有字段转换为字符串类型
for field in df_final.schema.fields:
    df_final = df_final.withColumn(field.name, col(field.name).cast(StringType()))

# 保存带有聚类标签的数据框到新的CSV文件
output_file = '/home/data/40lines/lines95_clusters'
df_final.repartition(1).write.csv(output_file, header=True, mode='overwrite')
print(f"文件已保存到: {output_file}")

# 停止Spark会话
spark.stop()
