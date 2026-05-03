#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <portaudio.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
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

namespace fs = std::filesystem;

enum class RunMode {
    Realtime = 1,
    Offline = 2,
};

struct WavAudioData {
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<float> mono_samples;
};

class Denoiser;
static bool ReadWavFile(const fs::path& path, WavAudioData& wav_data);
static bool WriteWavFile(const fs::path& path, const std::vector<float>& mono_samples, uint32_t sample_rate);
static std::vector<float> ResampleLinear(const std::vector<float>& input_samples, uint32_t input_rate, uint32_t output_rate);
static std::vector<float> DenoiseSamples(Denoiser& denoiser, const std::vector<float>& input_samples);

static RunMode PromptForRunMode() {
    while (true) {
        std::cout << "\nSelect run mode:\n"
                  << "  [1] Real-time denoise\n"
                    << "  [2] Denoise an audio file (.wav, .mp3) from data/\n"
                  << "Choose mode (Enter = 1): ";

        std::string line;
        std::getline(std::cin, line);

        if (line.empty()) {
            return RunMode::Realtime;
        }

        try {
            int selected = std::stoi(line);
            if (selected == 1) {
                return RunMode::Realtime;
            }
            if (selected == 2) {
                return RunMode::Offline;
            }
        } catch (...) {
        }

        std::cout << "Please enter 1 or 2." << std::endl;
    }
}

static std::vector<fs::path> ListAudioFiles(const fs::path& directory) {
    std::vector<fs::path> wav_files;
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return wav_files;
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        auto extension = entry.path().extension().wstring();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](wchar_t ch) {
            return static_cast<wchar_t>(std::towlower(ch));
        });

        if (extension == L".wav" || extension == L".mp3") {
            wav_files.push_back(entry.path());
        }
    }

    std::sort(wav_files.begin(), wav_files.end());
    return wav_files;
}

static fs::path BuildDenoisedOutputPath(const fs::path& input_path) {
    return input_path.parent_path() / (input_path.stem().wstring() + L"_denoised.wav");
}

// Convert an MP3 file to a temporary WAV using system ffmpeg. Returns empty path on failure.
static fs::path ConvertMp3ToWavTemp(const fs::path& mp3_path, uint32_t target_sample_rate) {
    try {
        fs::path temp_dir = fs::temp_directory_path();
        std::wstring stem = mp3_path.stem().wstring();
        std::wstring out_name = stem + L"_tmp_converted.wav";
        fs::path out_path = temp_dir / out_name;

        // Build command: ffmpeg -hide_banner -loglevel error -y -i "in" -ar <rate> -ac 1 "out"
        std::wstringstream cmd;
        cmd << L"ffmpeg -hide_banner -loglevel error -y -i \"" << mp3_path.wstring() << L"\" -ar "
            << target_sample_rate << L" -ac 1 \"" << out_path.wstring() << L"\"";

        // Use _wsystem on Windows wide strings
#ifdef _WIN32
        int rc = _wsystem(cmd.str().c_str());
#else
        std::string cmd_utf8;
        {
            std::wstring ws = cmd.str();
            cmd_utf8.assign(ws.begin(), ws.end());
        }
        int rc = std::system(cmd_utf8.c_str());
#endif

        if (rc != 0) {
            std::cerr << "ffmpeg failed to convert MP3: exit code " << rc << std::endl;
            return {};
        }

        return out_path;
    } catch (...) {
        return {};
    }
}

static fs::path PromptForOfflineInputFile() {
    const fs::path data_directory = fs::path(DATA_DIR);
    std::vector<fs::path> wav_files = ListAudioFiles(data_directory);

    if (wav_files.empty()) {
        std::cout << "No audio files (.wav, .mp3) found in data/." << std::endl;
        return {};
    }

    std::cout << "\nAvailable audio files in data/ (.wav, .mp3):" << std::endl;
    for (size_t index = 0; index < wav_files.size(); ++index) {
        std::cout << "  [" << index << "] " << wav_files[index].filename().string() << std::endl;
    }

    while (true) {
        std::cout << "Select input file index (Enter = 0): ";
        std::string line;
        std::getline(std::cin, line);

        if (line.empty()) {
            return wav_files.front();
        }

        try {
            size_t selected = static_cast<size_t>(std::stoul(line));
            if (selected < wav_files.size()) {
                return wav_files[selected];
            }
        } catch (...) {
        }

        std::cout << "Invalid file index." << std::endl;
    }
}

static uint16_t ReadUint16(std::istream& stream) {
    uint8_t bytes[2]{};
    stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    return static_cast<uint16_t>(bytes[0] | (static_cast<uint16_t>(bytes[1]) << 8));
}

static uint32_t ReadUint32(std::istream& stream) {
    uint8_t bytes[4]{};
    stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    return static_cast<uint32_t>(bytes[0]
        | (static_cast<uint32_t>(bytes[1]) << 8)
        | (static_cast<uint32_t>(bytes[2]) << 16)
        | (static_cast<uint32_t>(bytes[3]) << 24));
}

static void WriteUint16(std::ostream& stream, uint16_t value) {
    char bytes[2] = {
        static_cast<char>(value & 0xFF),
        static_cast<char>((value >> 8) & 0xFF)
    };
    stream.write(bytes, sizeof(bytes));
}

static void WriteUint32(std::ostream& stream, uint32_t value) {
    char bytes[4] = {
        static_cast<char>(value & 0xFF),
        static_cast<char>((value >> 8) & 0xFF),
        static_cast<char>((value >> 16) & 0xFF),
        static_cast<char>((value >> 24) & 0xFF)
    };
    stream.write(bytes, sizeof(bytes));
}

static bool ReadWavFile(const fs::path& path, WavAudioData& wav_data) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::cerr << "Failed to open WAV file: " << path.string() << std::endl;
        return false;
    }

    char riff_tag[4]{};
    stream.read(riff_tag, 4);
    if (stream.gcount() != 4 || std::strncmp(riff_tag, "RIFF", 4) != 0) {
        std::cerr << "Not a RIFF file: " << path.string() << std::endl;
        return false;
    }

    (void)ReadUint32(stream);

    char wave_tag[4]{};
    stream.read(wave_tag, 4);
    if (stream.gcount() != 4 || std::strncmp(wave_tag, "WAVE", 4) != 0) {
        std::cerr << "Not a WAVE file: " << path.string() << std::endl;
        return false;
    }

    bool found_fmt = false;
    bool found_data = false;
    std::vector<uint8_t> raw_samples;

    while (stream && (!found_fmt || !found_data)) {
        char chunk_id[4]{};
        stream.read(chunk_id, 4);
        if (stream.gcount() != 4) {
            break;
        }

        uint32_t chunk_size = ReadUint32(stream);

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            wav_data.audio_format = ReadUint16(stream);
            wav_data.channels = ReadUint16(stream);
            wav_data.sample_rate = ReadUint32(stream);
            (void)ReadUint32(stream);
            (void)ReadUint16(stream);
            wav_data.bits_per_sample = ReadUint16(stream);

            if (chunk_size > 16) {
                stream.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
            }

            found_fmt = true;
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            raw_samples.resize(chunk_size);
            stream.read(reinterpret_cast<char*>(raw_samples.data()), chunk_size);
            found_data = true;
        } else {
            stream.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }

        if (chunk_size % 2 != 0) {
            stream.seekg(1, std::ios::cur);
        }
    }

    if (!found_fmt || !found_data) {
        std::cerr << "Missing fmt or data chunk in: " << path.string() << std::endl;
        return false;
    }

    if (wav_data.channels == 0 || wav_data.sample_rate == 0 || wav_data.bits_per_sample == 0) {
        std::cerr << "Invalid WAV metadata in: " << path.string() << std::endl;
        return false;
    }

    size_t bytes_per_sample = wav_data.bits_per_sample / 8;
    if (bytes_per_sample == 0 || raw_samples.size() % (bytes_per_sample * wav_data.channels) != 0) {
        std::cerr << "Unexpected WAV data size in: " << path.string() << std::endl;
        return false;
    }

    size_t frame_count = raw_samples.size() / (bytes_per_sample * wav_data.channels);
    wav_data.mono_samples.resize(frame_count);

    for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
        double sample_sum = 0.0;
        for (uint16_t channel = 0; channel < wav_data.channels; ++channel) {
            size_t sample_offset = (frame_index * wav_data.channels + channel) * bytes_per_sample;
            float sample_value = 0.0f;

            if (wav_data.audio_format == 1 && wav_data.bits_per_sample == 16) {
                int16_t raw_value = static_cast<int16_t>(raw_samples[sample_offset]
                    | (static_cast<int16_t>(raw_samples[sample_offset + 1]) << 8));
                sample_value = static_cast<float>(raw_value) / 32768.0f;
            } else if (wav_data.audio_format == 3 && wav_data.bits_per_sample == 32) {
                float raw_value = 0.0f;
                std::memcpy(&raw_value, &raw_samples[sample_offset], sizeof(float));
                sample_value = raw_value;
            } else if (wav_data.audio_format == 1 && wav_data.bits_per_sample == 8) {
                uint8_t raw_value = raw_samples[sample_offset];
                sample_value = (static_cast<float>(raw_value) - 128.0f) / 128.0f;
            } else {
                std::cerr << "Unsupported WAV format: format=" << wav_data.audio_format
                          << " bits=" << wav_data.bits_per_sample << std::endl;
                return false;
            }

            sample_sum += sample_value;
        }

        wav_data.mono_samples[frame_index] = static_cast<float>(sample_sum / static_cast<double>(wav_data.channels));
    }

    return true;
}

static bool WriteWavFile(const fs::path& path, const std::vector<float>& mono_samples, uint32_t sample_rate) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        std::cerr << "Failed to create WAV file: " << path.string() << std::endl;
        return false;
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * (bits_per_sample / 8);
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_chunk_size = static_cast<uint32_t>(mono_samples.size() * sizeof(int16_t));
    const uint32_t riff_chunk_size = 36 + data_chunk_size;

    stream.write("RIFF", 4);
    WriteUint32(stream, riff_chunk_size);
    stream.write("WAVE", 4);
    stream.write("fmt ", 4);
    WriteUint32(stream, 16);
    WriteUint16(stream, 1);
    WriteUint16(stream, channels);
    WriteUint32(stream, sample_rate);
    WriteUint32(stream, byte_rate);
    WriteUint16(stream, block_align);
    WriteUint16(stream, bits_per_sample);
    stream.write("data", 4);
    WriteUint32(stream, data_chunk_size);

    for (float sample : mono_samples) {
        float clamped = std::clamp(sample, -1.0f, 1.0f);
        int16_t pcm_value = static_cast<int16_t>(std::lround(clamped * 32767.0f));
        WriteUint16(stream, static_cast<uint16_t>(pcm_value));
    }

    return static_cast<bool>(stream);
}

static std::vector<float> ResampleLinear(const std::vector<float>& input_samples, uint32_t input_rate, uint32_t output_rate) {
    if (input_samples.empty() || input_rate == 0 || output_rate == 0 || input_rate == output_rate) {
        return input_samples;
    }

    const double ratio = static_cast<double>(output_rate) / static_cast<double>(input_rate);
    const size_t output_count = static_cast<size_t>(std::max<double>(1.0, std::ceil(input_samples.size() * ratio)));
    std::vector<float> output_samples(output_count, 0.0f);

    for (size_t index = 0; index < output_count; ++index) {
        const double source_position = static_cast<double>(index) / ratio;
        const size_t left_index = static_cast<size_t>(std::floor(source_position));
        const size_t right_index = std::min(left_index + 1, input_samples.size() - 1);
        const double fraction = source_position - static_cast<double>(left_index);

        const double left_sample = static_cast<double>(input_samples[left_index]);
        const double right_sample = static_cast<double>(input_samples[right_index]);
        output_samples[index] = static_cast<float>(left_sample + (right_sample - left_sample) * fraction);
    }

    return output_samples;
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

static std::vector<float> DenoiseSamples(Denoiser& denoiser, const std::vector<float>& input_samples) {
    std::vector<float> output_samples(input_samples.size(), 0.0f);

    std::array<float, FRAME_SIZE> input_frame{};
    std::array<float, FRAME_SIZE> output_frame{};

    size_t offset = 0;
    while (offset < input_samples.size()) {
        size_t frame_count = std::min(static_cast<size_t>(FRAME_SIZE), input_samples.size() - offset);
        std::fill(input_frame.begin(), input_frame.end(), 0.0f);
        std::copy_n(input_samples.data() + offset, frame_count, input_frame.begin());

        if (!denoiser.Process(input_frame.data(), output_frame.data())) {
            return {};
        }

        std::copy_n(output_frame.begin(), frame_count, output_samples.data() + offset);
        offset += frame_count;
    }

    return output_samples;
}

static bool ProcessOfflineFile(const fs::path& input_path) {
    fs::path actual_input = input_path;
    fs::path temp_converted;
    auto ext = input_path.extension().wstring();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t ch){ return static_cast<wchar_t>(std::towlower(ch)); });
    if (ext == L".mp3") {
        // Convert mp3 to temporary wav via ffmpeg
        temp_converted = ConvertMp3ToWavTemp(input_path, SAMPLE_RATE);
        if (temp_converted.empty()) {
            std::cerr << "Failed to convert MP3 to WAV using ffmpeg." << std::endl;
            return false;
        }
        actual_input = temp_converted;
    }

    WavAudioData wav_data;
    if (!ReadWavFile(actual_input, wav_data)) {
        if (!temp_converted.empty() && fs::exists(temp_converted)) {
            fs::remove(temp_converted);
        }
        return false;
    }

    std::cout << "Input file: " << input_path.string() << std::endl;
    std::cout << "Channels: " << wav_data.channels
              << ", sample rate: " << wav_data.sample_rate
              << ", bits: " << wav_data.bits_per_sample
              << ", samples: " << wav_data.mono_samples.size() << std::endl;

    std::vector<float> model_input_samples = wav_data.mono_samples;
    if (wav_data.sample_rate != SAMPLE_RATE) {
        std::cout << "Resampling from " << wav_data.sample_rate << " Hz to " << SAMPLE_RATE << " Hz for model input." << std::endl;
        model_input_samples = ResampleLinear(wav_data.mono_samples, wav_data.sample_rate, SAMPLE_RATE);
    }

    Denoiser denoiser;
    std::vector<float> denoised_samples = DenoiseSamples(denoiser, model_input_samples);
    if (denoised_samples.empty() && !model_input_samples.empty()) {
        std::cerr << "Failed to denoise input WAV." << std::endl;
        return false;
    }

    if (wav_data.sample_rate != SAMPLE_RATE) {
        std::cout << "Resampling output back to " << wav_data.sample_rate << " Hz." << std::endl;
        denoised_samples = ResampleLinear(denoised_samples, SAMPLE_RATE, wav_data.sample_rate);
    }

    fs::path output_path = BuildDenoisedOutputPath(input_path);
    if (fs::exists(output_path)) {
        std::cout << "Skipping existing output: " << output_path.string() << std::endl;
        return true;
    }

    if (!WriteWavFile(output_path, denoised_samples, wav_data.sample_rate)) {
        if (!temp_converted.empty() && fs::exists(temp_converted)) {
            fs::remove(temp_converted);
        }
        return false;
    }

    std::cout << "Output file: " << output_path.string() << std::endl;
    if (!temp_converted.empty() && fs::exists(temp_converted)) {
        fs::remove(temp_converted);
    }
    return true;
}

static int RunOfflineMode() {
    fs::path input_path = PromptForOfflineInputFile();
    if (input_path.empty()) {
        return -1;
    }

    if (!ProcessOfflineFile(input_path)) {
        return -1;
    }

    std::cout << "Offline denoise completed successfully." << std::endl;
    return 0;
}

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

// --- 实时模式入口 ---
static int RunRealtimeMode() {
    EnableUtf8Console();

    std::cout << "=== Real-Time Audio Denoise (RNNoise ONNX) ===" << std::endl;

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

    std::cout << "Program exited successfully." << std::endl;
    return 0;
}

// --- 主函数 ---
int main() {
    EnableUtf8Console();

    std::cout << "=== Audio Denoise (RNNoise ONNX) ===" << std::endl;

    try {
        g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "RNNoiseDenoise");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

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

    RunMode mode = PromptForRunMode();
    int exit_code = 0;
    if (mode == RunMode::Realtime) {
        exit_code = RunRealtimeMode();
    } else {
        exit_code = RunOfflineMode();
    }

    delete g_ort_session;
    delete g_ort_env;
    g_ort_session = nullptr;
    g_ort_env = nullptr;
    g_memory_info.reset();
    input_node_names.clear();
    output_node_names.clear();
    input_names_ptr.clear();
    output_names_ptr.clear();

    return exit_code;
}