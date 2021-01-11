#Name: Matias Morales
# zid: z5216410

from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml import Pipeline, Transformer
from pyspark.sql.types import DoubleType, IntegerType
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    #Build the pipeline
    # white space expression tokenizer
    word_tokenizer = Tokenizer(inputCol="descript", outputCol="words")

    # bag of words count
    count_vectors = CountVectorizer(inputCol="words", outputCol="features")

    # label indexer
    label_maker = StringIndexer(inputCol = "category", outputCol = "label")

    class Selector(Transformer):
        def __init__(self, outputCols=['id','features', 'label']):
            self.outputCols=outputCols

        def _transform(self, df: DataFrame) -> DataFrame:
            return df.select(*self.outputCols)

    selector = Selector(outputCols = ['id','features', 'label'])

    # build the pipeline
    pipeline = Pipeline(stages=[word_tokenizer, count_vectors, label_maker, selector])

    return pipeline


def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):

    training_df = training_df.select("id","group","features","label", "label_0", "label_1", "label_2")

    last_nb_pred_0 = None
    last_nb_pred_1 = None
    last_nb_pred_2 = None
    last_svm_pred_0 = None
    last_svm_pred_1 = None
    last_svm_pred_2 = None

    # Training via Group
    for i in range(5):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()

        nb_model_0 = nb_0.fit(c_train)
        svm_model_0 = svm_0.fit(c_train)

        nb_model_1 = nb_1.fit(c_train)
        svm_model_1 = svm_1.fit(c_train)

        nb_model_2 = nb_2.fit(c_train)
        svm_model_2 = svm_2.fit(c_train)

        nb_pred_0 = nb_model_0.transform(c_test)
        svm_pred_0 = svm_model_0.transform(c_test)

        nb_pred_1 = nb_model_1.transform(c_test)
        svm_pred_1 = svm_model_1.transform(c_test)

        nb_pred_2 = nb_model_2.transform(c_test)
        svm_pred_2 = svm_model_2.transform(c_test)

        # Get ['id', 'prediction', 'group']
        nb_pred_0=nb_pred_0.select(nb_pred_0['id'],nb_pred_0['nb_pred_0'],nb_pred_0['group'])
        svm_pred_0=svm_pred_0.select(svm_pred_0['id'],svm_pred_0['svm_pred_0'],svm_pred_0['group'])

        nb_pred_1=nb_pred_1.select(nb_pred_1['id'],nb_pred_1['nb_pred_1'],nb_pred_1['group'])
        svm_pred_1=svm_pred_1.select(svm_pred_1['id'],svm_pred_1['svm_pred_1'],svm_pred_1['group'])

        nb_pred_2=nb_pred_2.select(nb_pred_2['id'],nb_pred_2['nb_pred_2'],nb_pred_2['group'])
        svm_pred_2=svm_pred_2.select(svm_pred_2['id'],svm_pred_2['svm_pred_2'],svm_pred_2['group'])

        if last_nb_pred_0 == None:
            last_nb_pred_0 = nb_pred_0
        else:
            last_nb_pred_0 = last_nb_pred_0.union(nb_pred_0)

        if last_nb_pred_1 == None:
            last_nb_pred_1 = nb_pred_1
        else:
            last_nb_pred_1 = last_nb_pred_1.union(nb_pred_1)

        if  last_nb_pred_2 == None:
             last_nb_pred_2 = nb_pred_2
        else:
            last_nb_pred_2 = last_nb_pred_2.union(nb_pred_2)

        if last_svm_pred_0== None:
            last_svm_pred_0= svm_pred_0
        else:
            last_svm_pred_0 = last_svm_pred_0.union(svm_pred_0)

        if last_svm_pred_1 == None:
            last_svm_pred_1 = svm_pred_1
        else:
            last_svm_pred_1 = last_svm_pred_1.union(svm_pred_1)

        if last_svm_pred_2 == None:
            last_svm_pred_2 = svm_pred_2
        else:
            last_svm_pred_2 = last_svm_pred_2.union(svm_pred_2)


    df = training_df

    df = df.join(last_nb_pred_0, on=['group','id'], how='left').join(last_svm_pred_0, on=['group','id'], how='left')
    df = df.join(last_nb_pred_1, on=['group','id'], how='left').join(last_svm_pred_1, on=['group','id'], how='left')
    df = df.join(last_nb_pred_2, on=['group','id'], how='left').join(last_svm_pred_2, on=['group','id'], how='left')

    df=df.withColumn('joint_pred_0', F.when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 0), 0).\
                     when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 1), 1).\
                     when((F.col("nb_pred_0") == 1) & (F.col("svm_pred_0") == 0), 2).\
                     when((F.col("nb_pred_0") == 1) & (F.col("svm_pred_0") == 1), 3))

    df=df.withColumn('joint_pred_1',F.when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 0), 0).\
                     when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 1), 1).\
                      when((F.col("nb_pred_1") == 1) & (F.col("svm_pred_1") == 0), 2).\
                      when((F.col("nb_pred_1") == 1) & (F.col("svm_pred_1") == 1), 3))


    df=df.withColumn('joint_pred_2',F.when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 0), 0).\
                     when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 1), 1).\
                     when((F.col("nb_pred_2") == 1) & (F.col("svm_pred_2") == 0), 2).\
                     when((F.col("nb_pred_2") == 1) & (F.col("svm_pred_2") == 1), 3))

    T=df.select(col("id"),col("group"),col("features"),col("label"),col("label_0"),col("label_1"),col("label_2")
                ,col("nb_pred_0"),col("nb_pred_1"),col("nb_pred_2"),col("svm_pred_0"),col("svm_pred_1")
                ,col("svm_pred_2"),col("joint_pred_0").cast(DoubleType()),col("joint_pred_1").cast(DoubleType())\
               ,col("joint_pred_2").cast(DoubleType()))

    return T


def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
      # Tranfrom Test Data
    df_1 = base_features_pipeline_model.transform(test_df)

    # Generate Meta Features based on Test Data
    df_2 = gen_base_pred_pipeline_model.transform(df_1)

    # Joint Model
    df_2=df_2.withColumn('joint_pred_0', F.when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 0), 0).\
                     when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 1), 1).\
                     when((F.col("nb_pred_0") == 1) & (F.col("svm_pred_0") == 0), 2).\
                     when((F.col("nb_pred_0") == 1) & (F.col("svm_pred_0") == 1), 3))

    df_2=df_2.withColumn('joint_pred_1',F.when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 0), 0).\
                     when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 1), 1).\
                      when((F.col("nb_pred_1") == 1) & (F.col("svm_pred_1") == 0), 2).\
                      when((F.col("nb_pred_1") == 1) & (F.col("svm_pred_1") == 1), 3))


    df_2=df_2.withColumn('joint_pred_2',F.when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 0), 0).\
                     when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 1), 1).\
                     when((F.col("nb_pred_2") == 1) & (F.col("svm_pred_2") == 0), 2).\
                     when((F.col("nb_pred_2") == 1) & (F.col("svm_pred_2") == 1), 3))

    # Prediction based on Meta Features
    meta_df = gen_meta_feature_pipeline_model.transform(df_2)
    meta_prediction = meta_classifier.transform(meta_df)

    return meta_prediction.select('id', 'label','final_prediction')
