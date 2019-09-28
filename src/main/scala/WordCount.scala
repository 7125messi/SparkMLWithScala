import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * 用户scala开发本地测试的spark wordcount程序
 */
object WordCount {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("My First Spark App!").setMaster("local")
    val sc = new SparkContext(conf)
    val lines = sc.textFile(
      "file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/word.txt",
      1
    )
    val wordCounts = lines.flatMap { line => line.split("\n") }
      .map{word =>(word,1)}
      .reduceByKey(_+_)
    //将计算结果保存到HDFS
    // wordCounts.saveAsTextFile("/user/result")
    //将计算结果保存到本地
    wordCounts.saveAsTextFile(
      "file:///Users/zhaoyadong/opt/git/sparklearning/src/main/resources/wordcount"
    )
    //打印输出
    wordCounts.foreach(pair => println(pair._1+":"+pair._2))
    sc.stop()
  }
}
