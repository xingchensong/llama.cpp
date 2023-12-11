# 模型量化损失评估
对比ggml不同大小（1b4, 7b）和不同block（32/128/512/2048）以及不同量化方式（q8_0/q5_0/q5_1/q4_0/q4_1）的量化损失。

## 测试集说明
共有两个验证集{data/inner_100.txt | data/common_100.txt}

## 依赖环境
如果出现以下错误：
```sh
cmake: error while loading shared libraries: librhash.so.0: cannot open shared object file: No such file or directory
`````````
请使用conda 安装相关软件
```sh
conda install cmake=3.14.0
`````````

## 脚本使用
1. 运行脚本
```
bash run.sh
```

## 现有模型的路径
| 模型 | v1 |
| :---: | :--- |
| ggml-fp16 | /jfs-hdfs/user/nlp/LLM/model_files/llama.cpp |
