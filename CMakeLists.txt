cmake_minimum_required(VERSION 3.10)
# 将项目名称更改为更具体的名字
project(MLPInferenceCpp)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定 ONNX Runtime 库的绝对路径
set(ONNXRUNTIME_DIR "/home/jd/t/git/onnxruntime-linux-x64-1.22.0")

# 检查路径以确保其有效
if(NOT EXISTS ${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "ONNX Runtime 头文件未在 ${ONNXRUNTIME_DIR}/include 找到。请检查路径。")
endif()
if(NOT EXISTS ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
    message(FATAL_ERROR "ONNX Runtime 库文件未在 ${ONNXRUNTIME_DIR}/lib 找到。请检查路径。")
endif()

# 添加 ONNX Runtime 的头文件目录
include_directories(${ONNXRUNTIME_DIR}/include)

# 添加 ONNX Runtime 的库文件目录
link_directories(${ONNXRUNTIME_DIR}/lib)

# --- 修改部分 ---
# 添加可执行文件，使用您提供的正确源文件名 'mlp_inference.cpp'
# 并将生成的可执行文件命名为 'mlp_inference'
add_executable(mlp_inference mlp_inference.cpp)

# 链接 ONNX Runtime 库到您的可执行文件
target_link_libraries(mlp_inference onnxruntime)

