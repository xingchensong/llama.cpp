#!/bin/bash
# Copyright [2023-10-12] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>


block=("512" "1024" "2048" "128" "32")

# model=("Q8_0" "Q4_0" "Q4_1" "Q5_0" "Q5_1" "Q2_K" "Q3_K" "Q4_K" "Q5_K" "Q6_K")
model=("Q8_0" "Q4_0" "Q4_1" "Q5_0" "Q5_1")
# name="llama2-7bchat"
name="bloom1b4.v2"

for b in "${block[@]}"; do
  echo "block ${b}"
  git checkout ggml.c
  sed -i "s/#define QK4_0 32/#define QK4_0 ${b}/g" ggml.c
  sed -i "s/#define QK4_1 32/#define QK4_1 ${b}/g" ggml.c
  sed -i "s/#define QK5_0 32/#define QK5_0 ${b}/g" ggml.c
  sed -i "s/#define QK5_1 32/#define QK5_1 ${b}/g" ggml.c
  sed -i "s/#define QK8_0 32/#define QK8_0 ${b}/g" ggml.c
  sed -i "s/#define QK8_1 32/#define QK8_1 ${b}/g" ggml.c
  grep "#define QK4_0" ggml.c
  grep "#define QK8_0" ggml.c
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j 6
  suffix="qk${b}"
  log_dir="exp/log-${name}-block${b}"
  mkdir -p exp
  mkdir -p ${log_dir}
  for m in "${model[@]}"; do
      ./build/bin/quantize \
        /jfs-hdfs/user/xingchen.song/workspace/github/llama.cpp/models/ggml-${name}.fp16.gguf \
        models/ggml-${name}.${m}.${suffix}.gguf ${m} 10

      echo "$b, $m, inner"

      ./build/bin/perplexity_1b4_inner -m models/ggml-${name}.${m}.${suffix}.gguf \
        -f /jfs-hdfs/user/xingchen.song/workspace/github/ggml/exp/torch/inner_100.txt \
        --ppl-stride 50 -c 100 -b 512 -s 2023 -t 8 >& ${log_dir}/inner.$m.log

      echo "$b, $m, common"

      ./build/bin/perplexity_1b4_common -m models/ggml-${name}.${m}.${suffix}.gguf \
        -f /jfs-hdfs/user/xingchen.song/workspace/github/ggml/exp/torch/common_100.txt \
        --ppl-stride 50 -c 100 -b 512 -s 2023 -t 8 >& ${log_dir}/common.$m.log
  done
done
