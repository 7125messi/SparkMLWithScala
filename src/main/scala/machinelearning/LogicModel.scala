package machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession


object LogicModel {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]): Unit = {
    // 0 构建Spark对象
    val spark = SparkSession
      .builder()
      .master("local") // 本地测试，否则报错 A master URL must be set in your configuration at org.apache.spark.SparkContext
      .appName("LogicModel")
      .enableHiveSupport()
      .getOrCreate() // 有就获取，无就创建

    //设置文件读取、存储的目录，HDFS最佳
    spark.sparkContext.setCheckpointDir("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/LogicModel")
    import spark.implicits._

    // 1 训练样本准备
    val training = spark.createDataFrame(
      Seq(
      (1.0, Vectors.sparse(692, Array(10, 20, 30), Array(-1.0, 1.5, 1.3))),
      (0.0, Vectors.sparse(692, Array(45, 175, 500), Array(-1.0, 1.5, 1.3))),
      (1.0, Vectors.sparse(692, Array(100, 200, 300), Array(-1.0, 1.5, 1.3)))
      )
    ).toDF("label", "features")
    training.show(false)

    val test = spark.createDataFrame(
      Seq(
      (1.0, Vectors.sparse(692, Array(10, 20, 30), Array(-1.0, 1.5, 1.3))),
      (0.0, Vectors.sparse(692, Array(45, 175, 500), Array(-1.0, 1.5, 1.3))),
      (1.0, Vectors.sparse(692, Array(100, 200, 300), Array(-1.0, 1.5, 1.3)))
      )
    ).toDF("label", "features")
    test.show(false)

    // 2 建立逻辑回归模型
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // 根据训练样本进行模型训练
    val lrModel = lr.fit(training)
    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")

//    // 3 建立多元回归模型
//    val mlr = new LogisticRegression()
//      .setMaxIter(10)
//      .setRegParam(0.3)
//      .setElasticNetParam(0.8)
//      .setFamily("multinomial")
//
//    // 根据训练样本进行模型训练
//    val mlrModel = mlr.fit(training)
//    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
//    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")

    // 4 对模型进行测试
    val test_predict = lrModel.transform(test)
    test_predict
      .select("label", "prediction", "probability", "features")
      .show(false)

    // 5 模型摘要
    val trainingSummary = lrModel.summary
    // 每次迭代目标值
    val objectiveHistory = trainingSummary.objectiveHistory

    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))

    // 6 计算模型指标数据
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
    // AUC指标
    val roc = binarySummary.roc
    roc.show(false)
    val AUC = binarySummary.areaUnderROC
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")

    // 设置模型阈值
    // 不同的阈值，计算不同的F1，然后通过最大的F1找出并重设模型的最佳阈值。
    val fMeasure = binarySummary.fMeasureByThreshold
    fMeasure.show(false)
    // 获得最大的F1值
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    // 找出最大F1值对应的阈值（最佳阈值）
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
    //并将模型的Threshold设置为选择出来的最佳分类阈值
    lrModel.setThreshold(bestThreshold)

    //7 模型保存与加载
    lrModel.write.overwrite().save("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/LogicModel")
    val load_lrModel = LogisticRegressionModel.load("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/LogicModel")
    println(load_lrModel)
  }
}
