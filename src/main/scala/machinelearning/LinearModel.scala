package machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession

/**
 * 线性回归
 */
object LinearModel {

  Logger.getLogger("org").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {

    // 0 构建Spark对象
    val spark = SparkSession
      .builder()
      .master("local") // 本地测试，否则报错 A master URL must be set in your configuration at org.apache.spark.SparkContext
      .appName("LinearModel")
      .enableHiveSupport()
      .getOrCreate() // 有就获取，无就创建

    //设置文件读取、存储的目录，HDFS最佳
    spark.sparkContext.setCheckpointDir("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/LinearModel")
    import spark.implicits._

    // 1 训练样本数据和测试样本数据准备
    val training = spark.createDataFrame(
      Seq(
        (5.601801561245534, Vectors.sparse(10, Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), Array(0.6949189734965766, -0.32697929564739403, -0.15359663581829275, -0.8951865090520432, 0.2057889391931318, -0.6676656789571533, -0.03553655732400762, 0.14550349954571096, 0.034600542078191854, 0.4223352065067103))),
        (0.2577820163584905, Vectors.sparse(10, Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), Array(0.8386555657374337, -0.1270180511534269, 0.499812362510895, -0.22686625128130267, -0.6452430441812433, 0.18869982177936828, -0.5804648622673358, 0.651931743775642, -0.6555641246242951, 0.17485476357259122))),
        (1.5299675726687754, Vectors.sparse(10, Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), Array(-0.13079299081883855, 0.0983382230287082, 0.15347083875928424, 0.45507300685816965, 0.1921083467305864, 0.6361110540492223, 0.7675261182370992, -0.2543488202081907, 0.2927051050236915, 0.680182444769418)))
      )
    ).toDF("label","features")
    training.show(false)

    val test = spark.createDataFrame(
      Seq(
      (5.601801561245534, Vectors.sparse(10, Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), Array(0.6949189734965766, -0.32697929564739403, -0.15359663581829275, -0.8951865090520432, 0.2057889391931318, -0.6676656789571533, -0.03553655732400762, 0.14550349954571096, 0.034600542078191854, 0.4223352065067103))),
      (0.2577820163584905, Vectors.sparse(10, Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), Array(0.8386555657374337, -0.1270180511534269, 0.499812362510895, -0.22686625128130267, -0.6452430441812433, 0.18869982177936828, -0.5804648622673358, 0.651931743775642, -0.6555641246242951, 0.17485476357259122))),
      (1.5299675726687754, Vectors.sparse(10, Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), Array(-0.13079299081883855, 0.0983382230287082, 0.15347083875928424, 0.45507300685816965, 0.1921083467305864, 0.6361110540492223, 0.7675261182370992, -0.2543488202081907, 0.2927051050236915, 0.680182444769418)))
      )
    ).toDF("label", "features")
    test.show(false)


    // 2 建立逻辑回归模型
    val lr = new LinearRegression()
      .setMaxIter(100)
      .setRegParam(0.1)
      .setElasticNetParam(0.5)

    // 3 根据训练样本进行模型训练
    val lrModel = lr.fit(training)
    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")

    // 4 对模型进行测试
    val test_predict = lrModel.transform(test)
    test_predict
      .select("label","prediction","features")
      .show(false)

    // 5 模型摘要
    val trainingSummary = lrModel.summary

    // 每次迭代目标值
    val objectiveHistory = trainingSummary.objectiveHistory
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show(false)
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    // 6 模型保存与加载
    lrModel.write.overwrite().save("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/LinearModel")
    val load_lrModel = LinearRegressionModel.load("file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/LinearModel")

  }
}
