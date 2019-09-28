package machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.SparkSession

object KmeansModel {

  Logger.getLogger("org").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {

    // 0.构建 Spark 对象
    val spark = SparkSession
      .builder()
      .master("local") // 本地测试，否则报错 A master URL must be set in your configuration at org.apache.spark.SparkContext.
      .appName("KmeansModel")
      .enableHiveSupport()
      .getOrCreate() // 有就获取无则创建

    spark.sparkContext.setCheckpointDir("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/KmeansModel") //设置文件读取、存储的目录，HDFS最佳
    import spark.implicits._

    // 读取样本
    val dataset = spark.read.format("libsvm").load("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/sample_kmeans_data.txt")
    dataset.show()

    // 训练 a k-means model.
    val kmeans = new KMeans()
      .setK(2)
      .setSeed(1L)
    val model = kmeans.fit(dataset)

    // 模型指标计算
    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // 结果显示
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

    // 模型保存与加载
    model.write.overwrite().save("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/KmeansModel")
    val load_kmeansmodel = KMeansModel.load("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/KmeansModel")
    println(load_kmeansmodel)
    spark.stop()
  }
}