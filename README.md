# RealTime-Audio-Denoise-CPP

基于 C++、ONNX Runtime 和 PortAudio 的实时音频降噪系统模拟项目。

这个项目的目标不是直接提供生产级降噪器，而是演示如何在 Windows 上把一个开源的 RNNoise ONNX 模型接入实时音频流，并完成以下流程：

1. 从麦克风读取实时音频
2. 按模型要求切分音频帧并送入 ONNX Runtime
3. 维护模型所需的 GRU 状态
4. 将推理结果回写到输出音频流

## 项目结构

- `src/main.cpp`：实时音频采集、模型推理和输出控制
- `models/rnnoise.onnx`：开源 RNNoise ONNX 模型
- `3rdparty/onnxruntime-win-x64-1.25.1/`：项目内置的 ONNX Runtime 运行时
- `build/`：CMake 生成的构建目录

## 环境要求

- Windows 11
- Visual Studio 2022
- CMake 3.10 或更高
- vcpkg
- 已安装 `portaudio`
- `ffmpeg`（离线模式处理 MP3 前，需要把 `ffmpeg/bin` 加入 PATH）

下面的说明以当前仓库的默认目录结构为例，实际路径请替换为你自己的本地环境：

- ONNX Runtime：`3rdparty/onnxruntime-win-x64-1.25.1`
- PortAudio：你的 vcpkg 根目录，例如 `<VCPKG_ROOT>`

## 构建

在项目根目录下生成构建文件，然后编译 Release 版本：

```powershell
cd .\build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release --target denoise_app
```

如果你已经配置过 CMake，也可以直接重新构建：

```powershell
cd .\build
cmake --build . --config Release --target denoise_app
```

## 运行

程序默认会从仓库中的 `models/rnnoise.onnx` 加载模型。构建后可直接运行：

```powershell
cd .\build\Release
.\denoise_app.exe
```

如果你想手动临时补充运行时 DLL，也可以在启动前设置 `PATH`：

```powershell
$env:PATH = '<PROJECT_ROOT>\3rdparty\onnxruntime-win-x64-1.25.1\lib;<VCPKG_ROOT>\installed\x64-windows\bin;' + $env:PATH
.\denoise_app.exe
```

程序启动后会打开默认音频输入输出流，按回车键退出。

## 验证结果

下面这些结果是项目当前状态的目标与已完成事项：

- CMake 配置成功
- `denoise_app` 成功编译
- 模型能够正常加载
- 音频流能够启动并正常退出

## 常见问题

### 1. 提示找不到模型文件

确认 `models/rnnoise.onnx` 存在于仓库根目录，并且运行目录是 `build/Release`。

### 2. 提示找不到 DLL

确认 `onnxruntime.dll` 和 `portaudio.dll` 可被运行时找到。优先检查下面两个目录：

- `3rdparty/onnxruntime-win-x64-1.25.1/lib`
- `<VCPKG_ROOT>/installed/x64-windows/bin`

### 3. 声音输入输出异常

这是一个实时模拟项目，强依赖麦克风、扬声器和系统默认音频设备。建议先用耳机或虚拟音频设备验证。

## 说明

当前实现重点是“实时音频降噪系统模拟”的工程流程打通。如果后续要提升降噪质量，可以继续补充：

- 更严格的特征提取
- 更准确的模型后处理
- 音频块缓存与延迟控制
- 日志与运行时状态面板
