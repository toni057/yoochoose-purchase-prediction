import os
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, LongType
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer
from functools import reduce


class PreProcessing:
    buy_schema = (
        StructType()
        .add("sessionid", LongType(), True)
        .add("timestamp", TimestampType(), True)
        .add("itemid", LongType(), True)
        .add("price", LongType(), True)
        .add("quantity", LongType(), True)
    )

    click_schema = (
        StructType()
        .add("sessionid", LongType(), True)
        .add("timestamp", TimestampType(), True)
        .add("itemid", LongType(), True)
        .add("category", StringType(), True)
    )

    def __init__(self, spark, path):
        print("Reading csv..........")
        self.read_csv_data(spark, path)

        print("Performing ETL.......")
        self.create_transformers()

        print("Saving tfrecods......")
        self.transform().repartition(1000).write.format("tfrecords").mode("overwrite").save(os.path.join(path, "train.tfrecords"))

    def read_csv_data(self, spark, path):
        self.buys = (
            spark.read
            .csv(os.path.join(path, "yoochoose-buys.dat"), schema=self.buy_schema)
            .groupBy("sessionid", "timestamp", "itemid")
            .agg(F.mean("price").alias("price"), F.sum("quantity").alias("quantity"))
            .repartition(1000)
            .cache()
        )

        self.clicks = (
            spark.read
            .csv(os.path.join(path, "yoochoose-clicks.dat"), schema=self.click_schema)
            .withColumnRenamed("sessionid", "sessionid_cl")
            .withColumnRenamed("timestamp", "timestamp_cl")
            .withColumnRenamed("itemid", "itemid_cl")
            .withColumnRenamed("category", "category_cl")
            .distinct()
            .repartition(1000)
            .cache()
        )

    def create_transformers(self):
        itemIndexer = (
            StringIndexer()
            .setInputCol("itemid_")
            .setOutputCol("itemid_indexed")
            .setStringOrderType("alphabetAsc")
            .setHandleInvalid("keep")
        )

        categoryIndexer = (
            StringIndexer()
            .setInputCol("category_")
            .setOutputCol("category_indexed")
            .setStringOrderType("alphabetAsc")
            .setHandleInvalid("keep")
        )

        self.itemIndexerModel = itemIndexer.fit(
            self.clicks.select(F.col("itemid_cl").alias("itemid_"))
                .union(self.buys.select(F.col("itemid").alias("itemid_")))
                .distinct()
        )

        self.categoryIndexerModel = categoryIndexer.fit(
            self.clicks.select(F.col("category_cl").alias("category_")).distinct()
        )

    def pivot_udf(df, *cols):
        mydf = df.select("sessionid", "timestamp", "price", "quantity").drop_duplicates().cache()
        for c in cols:
            mydf = mydf.join(df.withColumn('combcol', F.concat(F.lit('{}_'.format(c)), df['cum'])) \
                             .groupby("sessionid", "timestamp", "price", "quantity") \
                             .pivot('combcol') \
                             .agg(F.first(c)),
                             ["sessionid", "timestamp", "price", "quantity"])
        return mydf

    def pivot_data(self, buys, clicks):
        df0 = (
            buys
            .join(clicks,
                  (buys["sessionid"] == clicks["sessionid_cl"]) & (buys["itemid"] == clicks["itemid_cl"]), "inner")
            .filter("timestamp >= timestamp_cl")
            .withColumn("cum",
                        F.sum(F.lit(1)) \
                        .over(Window.partitionBy("sessionid", "timestamp", "price", "quantity") \
                              .orderBy(F.col("timestamp_cl").desc()) \
                              .rangeBetween(Window.unboundedPreceding, Window.currentRow)))
            .filter(F.col("cum") <= n_bins)
            .withColumn("cum", leadingzero_udf("cum"))
            .withColumn("dt_mins",
                        (F.col("timestamp").cast(LongType()) - F.col("timestamp_cl").cast(LongType())) / 60)
            .withColumn("timestamp", F.col("timestamp").cast("integer")
                        .drop("timestamp_cl"))
        )
        return pivot_udf(df0, "itemid_cl", "category_cl", "dt_mins")

    def stringIndexCols(self, df, column):
        def stringIndex(df, column, original_column, indexer):
            return (
                indexer.transform(
                    df.withColumnRenamed(column, original_column + "_"))
                    .drop(original_column + "_")
                    .withColumnRenamed(original_column + "_indexed", column)
            )

        if "itemid" in column:
            return stringIndex(df, column, "itemid", itemIndexerModel)
        if "category" in column:
            return stringIndex(df, column, "category", categoryIndexerModel)
        else:
            return df.withColumn(column,
                                 F.coalesce(F.col(column), F.lit(0).cast(
                                     list(filter(lambda c: c.name == column, df.schema))[0].dataType)))

    def transform(self):
        df = pivot_data(self, self.buys, self.clicks)
        return reduce(lambda df, c: stringIndexCols(df, c), df.columns, df)
