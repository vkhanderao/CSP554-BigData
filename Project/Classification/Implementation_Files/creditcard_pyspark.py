from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.tuning import TrainValidationSplitModel
from pyspark.ml.feature import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)

credit_df = spark.read.csv('creditcard.csv',header = True,inferSchema = True)

#Renaming the target variable from class to label
credit_df = credit_df.withColumnRenamed("Class","label")

#Defining the dataframe Schema, data types
print("Defining the dataframe Schema,data types")
credit_df.printSchema()

#Summary statistics on every column
print("Summary Statistics on every column of dataframe")
describe_stats = credit_df.describe()
print(describe_stats.toPandas())

print("Null Values Count of every column:\n",({col:credit_df.filter(credit_df[col].isNull()).count() for col in credit_df.columns}))

#calculating the correlation values
vect_cols = ['V1', 'V2', 'V3', 'V4', 'V5','V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','label']
combined_vector = VectorAssembler(inputCols = vect_cols,outputCol = 'vector_features')
vector_df = combined_vector.transform(credit_df)
corr_mat = Correlation.corr(vector_df,'vector_features')
corrmatrix = corr_mat.collect()[0][0].toArray().tolist()
mat_df = spark.createDataFrame(corrmatrix,vect_cols)
#filtering out the correlation values of every numeric column with label.
mat_df.select('label').show()

#Creating the heatmap of correlation matrix.
print("Creating the heatmap of correlation matrix.")
sns.heatmap(mat_df.toPandas())
plt.title("Heat Map of correlation Matrix")
plt.savefig('heatmap.pdf')
plt.clf()



cols =  ['V1', 'V2', 'V3', 'V4', 'V5','V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

print("boxplots of numeric columns before applying transformations")
with PdfPages('boxplots_before.pdf') as pdf:
    for i in cols:
        l = credit_df.select(i).collect()
        l = [r[0] for r in l]
        plt.figure()
        b = sns.boxplot(x = l)
        plt.title(i)
        pdf.savefig(b.get_figure())
        plt.clf()

#Target variable values
target_counts = credit_df.groupby('label').count()
print("Target value counts of each category:")
target_counts.show()

#creating the barplot target value counts of each category:
print("creating the barplot target value counts of each category:")
l = target_counts.collect()
x = [r[0] for r in l]
y = [r[1] for r in l]
plt.bar(x,y)
plt.xticks(x)
plt.xlabel('target variable')
plt.ylabel('Counts')
plt.savefig('barplot.pdf')
plt.clf()

#standardscaler transformations on the numerica columns with std,mean
combined_vector = VectorAssembler(inputCols = cols,outputCol = 'vector_features')
vector_df = combined_vector.transform(credit_df)
scaled = StandardScaler(inputCol = 'vector_features',outputCol = 'scaled_features',withStd = True,withMean = True)
final_df = scaled.fit(vector_df).transform(vector_df)
final_df.select('vector_features','scaled_features').show(1)




#Creating the train test split for the prediction after cross validations
train, test = final_df.select(['scaled_features','label']).randomSplit([0.7, 0.3], seed = 42)
# Calculate a balancing ratio to account for the class imbalance
balancing_ratio = train.filter(train['label']==0).count()/train.count()
train=train.withColumn("classWeights", F.when(train.label == 1,balancing_ratio).otherwise(1-balancing_ratio))

# Create a logistic regression object
lr = LogisticRegression(featuresCol = 'scaled_features', labelCol = 'label', weightCol="classWeights")

print("Model Building")
evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderPR')
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5,seed = 42)

# Run cross validations
cvModel = cv.fit(train)
print("Area under Precision- Recall Curve On test:",(evaluator.evaluate(cvModel.transform(test))))


