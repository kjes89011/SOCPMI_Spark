hadoop fs -put $1 $1
spark-submit SOCPMI.py $1
hadoop fs -cat Word_SOCPMI_Similarity/part* > $2
rm $1
hadoop fs -rm $1
hadoop fs -rm -r Word_SOCPMI_Similarity
