cmake_minimum_required(VERSION 3.10)
project(RealTimeAudioDenoise)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. 寻找 ONNX Runtime ---
# 这段代码会自动寻找 ONNX Runtime，如果找不到会提示你
# 如果你不知道路径，先留空，稍后我会教你如何找到它
set(ONNXRUNTIME_ROOT "C:/Users/Leville/Desktop/RealTime-Audio-Denoise-CPP/3rdparty/onnxruntime-win-x64-1.25.1")

if(NOT ONNXRUNTIME_ROOT STREQUAL "")
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
else()
    # 如果你不知道路径，可以尝试用 vcpkg 或 conan 安装
    # 或者手动指定路径
    message(WARNING "ONNXRUNTIME_ROOT is not set. Please set it to the path of your ONNX Runtime installation.")
endif()

# --- 2. 寻找 PortAudio ---
if(NOT DEFINED VCPKG_ROOT)
    set(VCPKG_ROOT "C:/Users/Leville/Desktop/vcpkg")
endif()

if(NOT DEFINED VCPKG_INSTALLED_DIR)
    set(VCPKG_INSTALLED_DIR "${VCPKG_ROOT}/installed")
endif()

if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-windows")
endif()

find_path(PORTAUDIO_INCLUDE_DIRS
    NAMES portaudio.h
    HINTS
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include"
)

find_library(PORTAUDIO_LIBRARIES
    NAMES portaudio
    HINTS
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib"
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib"
)

if(NOT PORTAUDIO_INCLUDE_DIRS OR NOT PORTAUDIO_LIBRARIES)
    message(FATAL_ERROR "Could not find PortAudio. Please check your vcpkg installation and VCPKG_ROOT/VCPKG_TARGET_TRIPLET settings.")
endif()

# --- 3. 添加可执行文件 ---
add_executable(denoise_app src/main.cpp)
target_include_directories(denoise_app PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
target_compile_definitions(denoise_app PRIVATE MODEL_PATH="${CMAKE_SOURCE_DIR}/models/rnnoise.onnx")

# --- 4. 链接库 ---
target_link_libraries(denoise_app PRIVATE ${PORTAUDIO_LIBRARIES})

target_link_libraries(denoise_app PRIVATE winmm dsound ole32 uuid setupapi mmdevapi)

# 如果你手动指定了 ONNX Runtime 路径，取消下面的注释
if(NOT ONNXRUNTIME_ROOT STREQUAL "")
    target_link_libraries(denoise_app PRIVATE onnxruntime)
endif()

add_custom_command(TARGET denoise_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:denoise_app>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/portaudio.dll"
        $<TARGET_FILE_DIR:denoise_app>
)