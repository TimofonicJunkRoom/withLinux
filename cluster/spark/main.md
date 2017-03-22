[Apache Spark](https://spark.apache.org)
===

Keyword: large-scale data processing.

Language: Scala, Python, R, Java

# downloading

download the prebuilt binary blob tarball and extract to a proper directory,
e.g. `spark-2.1.0-bin-hadoop2.7.tgz`

# [quick-start](http://spark.apache.org/docs/latest/quick-start.html)

spark shell `./bin/spark-shell`

Basics in Scala

```scala
val textFile = sc.textFile("READMD.md")
textFile.count()
textFile.first()
val linesWithSpark = textFile.filter(line => line.contains("Spark"))
textFile.filter(line => line.contains("Spark")).count()
```

Basics in Python

```python
textFile = sc.textFile("README.md")
textFile.count()
textFile.first()
linesWithSpark = textFile.filter(lambda line: "Spark" in line)
textFile.filter(lambda line: "Spark" in line).count()
```

More RDD operations in Scala

```scala
textFile.map(line => line.split(" ").size).reduce((a, b) => if (a > b) a else b)
import java.lang.Math
textFile.map(line => line.split(" ").size).reduce((a, b) => Math.max(a, b))
val wordCounts = textFile.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey((a, b) => a + b)
wordCounts.collect()
```

More RDD operations in Python

```python
textFile.map(lambda line: len(line.split())).reduce(lambda a, b: a if (a > b) else b)
def max(a,b):
  if a>b:
    return a
  else:
    return b

textFile.map(lambda line: len(line.split())).reduce(max)
wordCounts = textFile.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
wordCounts.collect()
```

Caching in Scala

```scala
linesWithSpark.cache()
```

Caching in Python

```scala
linesWithSpark.cache()
```

Self-contained application in Scala and the submitting process is more complex
than that in Python.

Self-contained application in Python

```python
"""SimpleApp.py"""
from pyspark import SparkContext

logFile = "YOUR_SPARK_HOME/README.md"  # Should be some file on your system
sc = SparkContext("local", "Simple App")
logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

sc.stop()
```

Submit your application to spark

```
$ YOUR_SPARK_HOME/bin/spark-submit \
  --master local[4] \
  SimpleApp.py
...
Lines with a: 46, Lines with b: 23
```

Run more examples

```
./bin/spark-submit examples/src/main/python/pi.py
```

# [cluster installation](http://spark.apache.org/docs/latest/cluster-overview.html)

TODO

# [Spark Programming](http://spark.apache.org/docs/latest/programming-guide.html)

TODO
