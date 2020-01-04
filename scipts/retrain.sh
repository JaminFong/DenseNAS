#!/usr/bin/env sh

CWD=`pwd`
HDFS=hdfs://hobot-bigdata/
#set this to enable reading from hdfs
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms512m -Xmx10000m"
export NCCL_DEBUG=WARN

cd ${WORKING_PATH}
cp -r ${WORKING_PATH}/* /job_data

hadoop fs -get hdfs://hobot-bigdata/user/jiemin.fang/envs/torch12_cu10.tar && tar xf torch12_cu10.tar && mv torch12_cu10 pytorch

mkdir imagenet && cd imagenet
hadoop fs -get hdfs://hobot-bigdata/user/jiemin.fang/data/imagenet-lmdb/train_origin
mv train_origin train
hadoop fs -get hdfs://hobot-bigdata/user/jiemin.fang/data/imagenet-lmdb/train_datalist
hadoop fs -get hdfs://hobot-bigdata/user/jiemin.fang/data/imagenet_val.tar
tar xf imagenet_val.tar 
cd ..

./pytorch/bin/python -m run_apis.retrain \
    --data_path ./imagenet \
    --dataset imagenet \
    --report_freq 400 \
    --save /job_data \
    --tb_path /job_tboard \
    --config train_config_imagenet # train_config_imagenet_mbv3
