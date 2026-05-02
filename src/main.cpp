#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <portaudio.h>

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#include <propvarutil.h>
#endif

#ifndef MODEL_PATH
#define MODEL_PATH "models/rnnoise.onnx"
#endif

// --- 配置参数 ---
#define SAMPLE_RATE 48000
#define FRAME_SIZE 42   // 该 RNNoise ONNX 模型的固定帧长
#define CHANNELS 1      // 单声道

constexpr int VAD_STATE_SIZE = 24;
constexpr int NOISE_STATE_SIZE = 48;
constexpr int DENOISE_STATE_SIZE = 96;

// --- 错误检查宏 ---
#define CHECK_ERROR(err) if (err != paNoError) { \
    std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl; \
    return 1; \
}

static void EnableUtf8Console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

static std::wstring GetDefaultAudioDeviceName(EDataFlow flow) {
#ifdef _WIN32
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool should_uninitialize = SUCCEEDED(hr);

    IMMDeviceEnumerator* enumerator = nullptr;
    IMMDevice* device = nullptr;
    IPropertyStore* property_store = nullptr;
    std::wstring device_name = L"<unknown>";

    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator));
    if (SUCCEEDED(hr) && enumerator != nullptr) {
        hr = enumerator->GetDefaultAudioEndpoint(flow, eConsole, &device);
        if (SUCCEEDED(hr) && device != nullptr) {
            hr = device->OpenPropertyStore(STGM_READ, &property_store);
            if (SUCCEEDED(hr) && property_store != nullptr) {
                PROPVARIANT value;
                PropVariantInit(&value);
                hr = property_store->GetValue(PKEY_Device_FriendlyName, &value);
                if (SUCCEEDED(hr) && value.vt == VT_LPWSTR && value.pwszVal != nullptr) {
                    device_name = value.pwszVal;
                }
                PropVariantClear(&value);
            }
        }
    }

    if (property_store != nullptr) {
        property_store->Release();
    }
    if (device != nullptr) {
        device->Release();
    }
    if (enumerator != nullptr) {
        enumerator->Release();
    }
    if (should_uninitialize) {
        CoUninitialize();
    }

    return device_name;
#else
    return L"<unknown>";
#endif
}

static void PrintWideLine(const std::wstring& text) {
#ifdef _WIN32
    HANDLE output_handle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (output_handle != INVALID_HANDLE_VALUE && output_handle != nullptr) {
        DWORD written = 0;
        WriteConsoleW(output_handle, text.c_str(), static_cast<DWORD>(text.size()), &written, nullptr);
        WriteConsoleW(output_handle, L"\n", 1, &written, nullptr);
        return;
    }
#endif
    std::wcout << text << std::endl;
}

static void PrintDefaultAudioDevices() {
    PaDeviceIndex input_device = Pa_GetDefaultInputDevice();
    PaDeviceIndex output_device = Pa_GetDefaultOutputDevice();

    if (input_device == paNoDevice) {
        PrintWideLine(L"Default input device: <none>");
    } else {
        std::wstringstream line;
        line << L"Default input device: "
             << GetDefaultAudioDeviceName(eCapture)
             << L" (index " << input_device << L")";
        PrintWideLine(line.str());
    }

    if (output_device == paNoDevice) {
        PrintWideLine(L"Default output device: <none>");
    } else {
        std::wstringstream line;
        line << L"Default output device: "
             << GetDefaultAudioDeviceName(eRender)
             << L" (index " << output_device << L")";
        PrintWideLine(line.str());
    }
}

struct AudioDeviceSelection {
    PaDeviceIndex input_device = paNoDevice;
    PaDeviceIndex output_device = paNoDevice;
};

static void PrintAvailableAudioDevices() {
    int device_count = Pa_GetDeviceCount();
    if (device_count < 0) {
        std::cout << "Failed to enumerate PortAudio devices: " << Pa_GetErrorText(device_count) << std::endl;
        return;
    }

    std::cout << "\nAvailable audio devices:" << std::endl;
    for (PaDeviceIndex device_index = 0; device_index < device_count; ++device_index) {
        const PaDeviceInfo* device_info = Pa_GetDeviceInfo(device_index);
        if (device_info == nullptr) {
            continue;
        }

        const PaHostApiInfo* host_api_info = Pa_GetHostApiInfo(device_info->hostApi);

        std::cout << "  [" << device_index << "] "
                  << (device_info->name != nullptr ? device_info->name : "<unknown>")
                  << " | in: " << device_info->maxInputChannels
                  << " | out: " << device_info->maxOutputChannels
                  << " | api: "
                  << (host_api_info && host_api_info->name != nullptr ? host_api_info->name : "<unknown>");

        if (device_index == Pa_GetDefaultInputDevice()) {
            std::cout << " | default input";
        }
        if (device_index == Pa_GetDefaultOutputDevice()) {
            std::cout << " | default output";
        }

        std::cout << std::endl;
    }
}

static PaDeviceIndex PromptForDeviceIndex(const std::string& prompt, bool need_input, bool need_output) {
    while (true) {
        std::cout << prompt;
        std::string line;
        std::getline(std::cin, line);

        if (line.empty()) {
            return paNoDevice;
        }

        try {
            int selected = std::stoi(line);
            PaDeviceIndex device_index = static_cast<PaDeviceIndex>(selected);
            const PaDeviceInfo* device_info = Pa_GetDeviceInfo(device_index);
            if (device_info == nullptr) {
                std::cout << "Invalid device index." << std::endl;
                continue;
            }

            if (need_input && device_info->maxInputChannels < 1) {
                std::cout << "That device has no input channels." << std::endl;
                continue;
            }

            if (need_output && device_info->maxOutputChannels < 1) {
                std::cout << "That device has no output channels." << std::endl;
                continue;
            }

            return device_index;
        } catch (...) {
            std::cout << "Please enter a valid number, or press Enter to use the default device." << std::endl;
        }
    }
}

static AudioDeviceSelection SelectAudioDevices() {
    PrintAvailableAudioDevices();

    AudioDeviceSelection selection;
    selection.input_device = PromptForDeviceIndex(
        "Select input device index (Enter = default input): ", true, false);
    selection.output_device = PromptForDeviceIndex(
        "Select output device index (Enter = default output): ", false, true);

    if (selection.input_device == paNoDevice) {
        selection.input_device = Pa_GetDefaultInputDevice();
    }

    if (selection.output_device == paNoDevice) {
        selection.output_device = Pa_GetDefaultOutputDevice();
    }

    return selection;
}

static void PrintSupportedAudioPairs(double sample_rate) {
    int device_count = Pa_GetDeviceCount();
    if (device_count < 0) {
        std::cout << "Failed to enumerate PortAudio devices for pair probing: "
                  << Pa_GetErrorText(device_count) << std::endl;
        return;
    }

    std::cout << "\nSupported input/output device pairs at " << sample_rate << " Hz:" << std::endl;
    int supported_pair_count = 0;

    for (PaDeviceIndex input_device = 0; input_device < device_count; ++input_device) {
        const PaDeviceInfo* input_info = Pa_GetDeviceInfo(input_device);
        if (input_info == nullptr || input_info->maxInputChannels < 1) {
            continue;
        }

        PaStreamParameters input_parameters{};
        input_parameters.device = input_device;
        input_parameters.channelCount = CHANNELS;
        input_parameters.sampleFormat = paFloat32;
        input_parameters.suggestedLatency = input_info->defaultLowInputLatency;
        input_parameters.hostApiSpecificStreamInfo = nullptr;

        for (PaDeviceIndex output_device = 0; output_device < device_count; ++output_device) {
            const PaDeviceInfo* output_info = Pa_GetDeviceInfo(output_device);
            if (output_info == nullptr || output_info->maxOutputChannels < 1) {
                continue;
            }

            PaStreamParameters output_parameters{};
            output_parameters.device = output_device;
            output_parameters.channelCount = CHANNELS;
            output_parameters.sampleFormat = paFloat32;
            output_parameters.suggestedLatency = output_info->defaultLowOutputLatency;
            output_parameters.hostApiSpecificStreamInfo = nullptr;

            PaError supported = Pa_IsFormatSupported(&input_parameters, &output_parameters, sample_rate);
            if (supported == paFormatIsSupported) {
                ++supported_pair_count;
                std::cout << "  [" << input_device << " -> " << output_device << "] "
                          << (input_info->name != nullptr ? input_info->name : "<unknown>")
                          << "  ==>  "
                          << (output_info->name != nullptr ? output_info->name : "<unknown>")
                          << std::endl;
            }
        }
    }

    if (supported_pair_count == 0) {
        std::cout << "  <no supported pairs found>" << std::endl;
    } else {
        std::cout << "Total supported pairs: " << supported_pair_count << std::endl;
    }
}

static bool ValidateAudioFormat(const AudioDeviceSelection& audio_devices, double sample_rate) {
    const PaDeviceInfo* input_device_info = Pa_GetDeviceInfo(audio_devices.input_device);
    const PaDeviceInfo* output_device_info = Pa_GetDeviceInfo(audio_devices.output_device);

    PaStreamParameters input_parameters{};
    input_parameters.device = audio_devices.input_device;
    input_parameters.channelCount = CHANNELS;
    input_parameters.sampleFormat = paFloat32;
    input_parameters.suggestedLatency = input_device_info ? input_device_info->defaultLowInputLatency : 0.0;
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = audio_devices.output_device;
    output_parameters.channelCount = CHANNELS;
    output_parameters.sampleFormat = paFloat32;
    output_parameters.suggestedLatency = output_device_info ? output_device_info->defaultLowOutputLatency : 0.0;
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    PaError supported = Pa_IsFormatSupported(&input_parameters, &output_parameters, sample_rate);
    if (supported != paFormatIsSupported) {
        std::cerr << "Selected device combination is not supported at "
                  << sample_rate << " Hz: " << Pa_GetErrorText(supported) << std::endl;
        return false;
    }

    return true;
}

// --- 全局变量 ---
Ort::Env* g_ort_env = nullptr;
Ort::Session* g_ort_session = nullptr;
std::unique_ptr<Ort::MemoryInfo> g_memory_info;

// 用于存储 ONNX Runtime 的输入/输出名称
std::vector<Ort::AllocatedStringPtr> input_names_ptr;
std::vector<Ort::AllocatedStringPtr> output_names_ptr;
std::vector<const char*> input_node_names;
std::vector<const char*> output_node_names;

// --- 降噪处理器类 ---
class Denoiser {
public:
    Denoiser()
        : vad_gru_state_(VAD_STATE_SIZE, 0.0f),
          noise_gru_state_(NOISE_STATE_SIZE, 0.0f),
          denoise_gru_state_(DENOISE_STATE_SIZE, 0.0f),
          last_vad_score_(0.0f),
          last_gain_(1.0f) {
    }

    bool Process(const float* input_audio, float* output_audio) {
        try {
            std::array<int64_t, 3> main_shape{1, 1, FRAME_SIZE};
            std::array<int64_t, 2> vad_shape{1, VAD_STATE_SIZE};
            std::array<int64_t, 2> noise_shape{1, NOISE_STATE_SIZE};
            std::array<int64_t, 2> denoise_shape{1, DENOISE_STATE_SIZE};

            auto main_input_tensor = Ort::Value::CreateTensor<float>(
                *g_memory_info, 
                const_cast<float*>(input_audio), 
                FRAME_SIZE, 
                main_shape.data(), 
                main_shape.size()
            );

            auto vad_state_tensor = Ort::Value::CreateTensor<float>(
                *g_memory_info,
                vad_gru_state_.data(),
                vad_gru_state_.size(),
                vad_shape.data(),
                vad_shape.size()
            );

            auto noise_state_tensor = Ort::Value::CreateTensor<float>(
                *g_memory_info,
                noise_gru_state_.data(),
                noise_gru_state_.size(),
                noise_shape.data(),
                noise_shape.size()
            );

            auto denoise_state_tensor = Ort::Value::CreateTensor<float>(
                *g_memory_info,
                denoise_gru_state_.data(),
                denoise_gru_state_.size(),
                denoise_shape.data(),
                denoise_shape.size()
            );

            std::array<Ort::Value, 4> input_tensors = {
                std::move(main_input_tensor),
                std::move(vad_state_tensor),
                std::move(noise_state_tensor),
                std::move(denoise_state_tensor)
            };

            auto output_tensors = g_ort_session->Run(
                Ort::RunOptions{nullptr},
                input_node_names.data(),
                input_tensors.data(),
                input_tensors.size(),
                output_node_names.data(),
                output_node_names.size()
            );

            std::copy(
                output_tensors[0].GetTensorMutableData<float>(),
                output_tensors[0].GetTensorMutableData<float>() + denoise_gru_state_.size(),
                denoise_gru_state_.begin()
            );
            std::copy(
                output_tensors[2].GetTensorMutableData<float>(),
                output_tensors[2].GetTensorMutableData<float>() + noise_gru_state_.size(),
                noise_gru_state_.begin()
            );
            std::copy(
                output_tensors[3].GetTensorMutableData<float>(),
                output_tensors[3].GetTensorMutableData<float>() + vad_gru_state_.size(),
                vad_gru_state_.begin()
            );

            const float* denoise_features = output_tensors[1].GetTensorMutableData<float>();
            const float* vad_score_ptr = output_tensors[4].GetTensorMutableData<float>();
            last_vad_score_ = std::isfinite(vad_score_ptr[0]) ? vad_score_ptr[0] : 0.0f;

            float feature_mean = 0.0f;
            for (int i = 0; i < DENOISE_STATE_SIZE; ++i) {
                feature_mean += denoise_features[i];
            }
            feature_mean /= static_cast<float>(DENOISE_STATE_SIZE);

            last_gain_ = std::clamp(0.25f + 0.75f * last_vad_score_, 0.10f, 1.0f);
            last_gain_ = std::clamp(last_gain_ * (0.75f + 0.25f * feature_mean), 0.05f, 1.0f);

            for (int i = 0; i < FRAME_SIZE; ++i) {
                output_audio[i] = input_audio[i] * last_gain_;
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
            return false;
        }
    }

    float last_vad_score() const { return last_vad_score_; }
    float last_gain() const { return last_gain_; }

private:
    std::vector<float> vad_gru_state_;
    std::vector<float> noise_gru_state_;
    std::vector<float> denoise_gru_state_;
    float last_vad_score_;
    float last_gain_;
};

// --- PortAudio 回调函数 ---
int paCallback(const void *inputBuffer, void *outputBuffer,
               unsigned long framesPerBuffer,
               const PaStreamCallbackTimeInfo* timeInfo,
               PaStreamCallbackFlags statusFlags,
               void *userData) {
    
    Denoiser* denoiser = (Denoiser*)userData;
    const float* in = (const float*)inputBuffer;
    float* out = (float*)outputBuffer;

    // PortAudio 可能会传入 framesPerBuffer > 480，我们需要分块处理
    // 但为了简单演示，我们假设回调正好是 480 或者我们在内部循环处理
    // 这里做一个简单的循环处理，确保每 480 样本处理一次
    
    unsigned long framesLeft = framesPerBuffer;
    while(framesLeft >= FRAME_SIZE) {
        denoiser->Process(in, out);
        in += FRAME_SIZE;
        out += FRAME_SIZE;
        framesLeft -= FRAME_SIZE;
    }

    // 如果 framesLeft > 0 (不足一帧)，通常直接拷贝或填充0，这里简单填充0
    for(unsigned long i=0; i<framesLeft; i++) {
        out[i] = 0.0f;
    }

    return paContinue;
}

// --- 主函数 ---
int main() {
    EnableUtf8Console();

    std::cout << "=== Real-Time Audio Denoise (RNNoise ONNX) ===" << std::endl;

    // 1. 初始化 ONNX Runtime
    try {
        g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "RNNoiseDenoise");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Windows 下处理宽字符路径，防止中文路径报错
#ifdef _WIN32
        std::wstring w_model_path = std::wstring(MODEL_PATH, MODEL_PATH + strlen(MODEL_PATH));
        g_ort_session = new Ort::Session(*g_ort_env, w_model_path.c_str(), session_options);
#else
        g_ort_session = new Ort::Session(*g_ort_env, MODEL_PATH, session_options);
#endif
        g_memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < g_ort_session->GetInputCount(); ++i) {
            auto input_name = g_ort_session->GetInputNameAllocated(i, allocator);
            input_names_ptr.push_back(std::move(input_name));
            input_node_names.push_back(input_names_ptr.back().get());
        }

        for (size_t i = 0; i < g_ort_session->GetOutputCount(); ++i) {
            auto output_name = g_ort_session->GetOutputNameAllocated(i, allocator);
            output_names_ptr.push_back(std::move(output_name));
            output_node_names.push_back(output_names_ptr.back().get());
        }

        std::cout << "Model loaded successfully: " << MODEL_PATH << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
        return -1;
    }

    // 2. 初始化 PortAudio
    PaError err = Pa_Initialize();
    CHECK_ERROR(err);

    PrintDefaultAudioDevices();
    PrintSupportedAudioPairs(SAMPLE_RATE);

    AudioDeviceSelection audio_devices = SelectAudioDevices();

    const PaDeviceInfo* input_device_info = Pa_GetDeviceInfo(audio_devices.input_device);
    const PaDeviceInfo* output_device_info = Pa_GetDeviceInfo(audio_devices.output_device);
    std::cout << "Using input device: "
              << (input_device_info && input_device_info->name ? input_device_info->name : "<unknown>")
              << " (index " << audio_devices.input_device << ")" << std::endl;
    std::cout << "Using output device: "
              << (output_device_info && output_device_info->name ? output_device_info->name : "<unknown>")
              << " (index " << audio_devices.output_device << ")" << std::endl;

    if (!ValidateAudioFormat(audio_devices, SAMPLE_RATE)) {
        Pa_Terminate();
        delete g_ort_session;
        delete g_ort_env;
        return -1;
    }

    Denoiser denoiser;

    PaStream* stream;
    PaStreamParameters input_parameters{};
    input_parameters.device = audio_devices.input_device;
    input_parameters.channelCount = CHANNELS;
    input_parameters.sampleFormat = paFloat32;
    input_parameters.suggestedLatency = input_device_info ? input_device_info->defaultLowInputLatency : 0.0;
    input_parameters.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters output_parameters{};
    output_parameters.device = audio_devices.output_device;
    output_parameters.channelCount = CHANNELS;
    output_parameters.sampleFormat = paFloat32;
    output_parameters.suggestedLatency = output_device_info ? output_device_info->defaultLowOutputLatency : 0.0;
    output_parameters.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(&stream,
                        &input_parameters,
                        &output_parameters,
                        SAMPLE_RATE,
                        FRAME_SIZE,
                        paNoFlag,
                        paCallback,
                        &denoiser);
    CHECK_ERROR(err);

    std::cout << "Starting audio stream... Press Enter to stop." << std::endl;
    err = Pa_StartStream(stream);
    CHECK_ERROR(err);

    std::cin.get(); // 等待回车键退出

    std::cout << "Stopping stream..." << std::endl;
    err = Pa_StopStream(stream);
    CHECK_ERROR(err);
    err = Pa_CloseStream(stream);
    CHECK_ERROR(err);
    Pa_Terminate();

    // 3. 清理 ONNX Runtime
    delete g_ort_session;
    delete g_ort_env;

    std::cout << "Program exited successfully." << std::endl;
    return 0;
}