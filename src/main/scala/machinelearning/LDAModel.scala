package machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.sql.SparkSession

object LDAModel {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    // 0.构建 Spark 对象
    val spark = SparkSession
      .builder()
      .master("local") // 本地测试，否则报错 A master URL must be set in your configuration at org.apache.spark.SparkContext.
      .appName("LDAModel")
      .enableHiveSupport()
      .getOrCreate() // 有就获取无则创建

    // 1.读取样本
    val dataset = spark.read.format("libsvm").load("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/sample_lda_libsvm_data.txt")
    dataset.show()

    // 2.训练 LDA model
    val lda = new LDA()
      .setK(10)
      .setMaxIter(10)
    val model = lda.fit(dataset)

    val ll = model.logLikelihood(dataset)
    val lp = model.logPerplexity(dataset)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")

    // 3.主题 topics
    val topics = model.describeTopics(3)
    println("The topics described by their top-weighted terms:")
    topics.show(false)

    // 4.测试结果
    val transformed = model.transform(dataset)
    transformed.show(false)

    // 5.模型保存与加载
    model.save("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/LDAModel")

    spark.stop()
  }
}