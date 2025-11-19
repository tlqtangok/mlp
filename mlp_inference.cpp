#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

class JSONParser
{
private:
    std::string _content;

    std::string _trim(const std::string& str)
    {
        size_t first = str.find_first_not_of(" \t\n\r\"");
        if (first == std::string::npos)
        {
            return "";
        }
        size_t last = str.find_last_not_of(" \t\n\r\",}]");
        return str.substr(first, (last - first + 1));
    }

    std::vector<float> _parse_array(const std::string& content, const std::string& key)
    {
        std::vector<float> result;
        size_t key_pos = content.find("\"" + key + "\"");
        if (key_pos == std::string::npos)
        {
            return result;
        }

        size_t start = content.find("[", key_pos);
        size_t end = content.find("]", start);
        if (start == std::string::npos || end == std::string::npos)
        {
            return result;
        }

        std::string array_content = content.substr(start + 1, end - start - 1);
        std::stringstream ss(array_content);
        std::string token;

        while (std::getline(ss, token, ','))
        {
            std::string trimmed = _trim(token);
            if (!trimmed.empty())
            {
                result.push_back(std::stof(trimmed));
            }
        }
        return result;
    }

public:
    JSONParser(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        _content = buffer.str();
        file.close();
    }

    std::vector<float> get_array(const std::string& key)
    {
        return _parse_array(_content, key);
    }
};

class MLPInference
{
private:
    Ort::Env _env;
    Ort::Session _session;
    Ort::MemoryInfo _memory_info;
    std::vector<float> _scaler_mean;
    std::vector<float> _scaler_scale;
    std::vector<const char*> _input_names;
    std::vector<const char*> _output_names;
    size_t _input_size;
    size_t _output_size;

    void _load_scaler_params(const std::string& scaler_file)
    {
        JSONParser parser(scaler_file);
        _scaler_mean = parser.get_array("mean");
        _scaler_scale = parser.get_array("scale");

        if (_scaler_mean.empty() || _scaler_scale.empty())
        {
            throw std::runtime_error("Failed to parse scaler parameters");
        }

        std::cout << "Loaded scaler parameters: " << std::endl;
        std::cout << "  Mean size: " << _scaler_mean.size() << std::endl;
        std::cout << "  Scale size: " << _scaler_scale.size() << std::endl;
    }

    std::vector<float> _standardize_input(const std::vector<float>& input)
    {
        if (input.size() != _scaler_mean.size())
        {
            throw std::runtime_error("Input size mismatch");
        }

        std::vector<float> standardized(input.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            standardized[i] = (input[i] - _scaler_mean[i]) / _scaler_scale[i];
        }
        return standardized;
    }

    std::vector<float> _apply_softmax(const std::vector<float>& logits)
    {
        float max_logit = *std::max_element(logits.begin(), logits.end());
        
        std::vector<float> exp_values(logits.size());
        float sum_exp = 0.0f;
        
        for (size_t i = 0; i < logits.size(); ++i)
        {
            exp_values[i] = std::exp(logits[i] - max_logit);
            sum_exp += exp_values[i];
        }
        
        std::vector<float> probabilities(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
        {
            probabilities[i] = exp_values[i] / sum_exp;
        }
        
        return probabilities;
    }

    std::vector<float> _run_inference(const std::vector<float>& input)
    {
        std::vector<float> standardized_input = _standardize_input(input);

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(_input_size)};

        auto input_tensor = Ort::Value::CreateTensor<float>(
            _memory_info,
            const_cast<float*>(standardized_input.data()),
            standardized_input.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto output_tensors = _session.Run(
            Ort::RunOptions{nullptr},
            _input_names.data(),
            &input_tensor,
            1,
            _output_names.data(),
            1
        );

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> logits(output_data, output_data + _output_size);

        return logits;
    }

public:
    MLPInference(const std::string& model_path, const std::string& scaler_path)
        : _env(ORT_LOGGING_LEVEL_WARNING, "MLPInference"),
          _session(nullptr),
          _memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        _session = Ort::Session(_env, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_input_nodes = _session.GetInputCount();
        size_t num_output_nodes = _session.GetOutputCount();

        for (size_t i = 0; i < num_input_nodes; i++)
        {
            auto input_name = _session.GetInputNameAllocated(i, allocator);
            _input_names.push_back(input_name.get());
            input_name.release();
        }

        for (size_t i = 0; i < num_output_nodes; i++)
        {
            auto output_name = _session.GetOutputNameAllocated(i, allocator);
            _output_names.push_back(output_name.get());
            output_name.release();
        }

        auto input_shape = _session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        _input_size = input_shape[1];

        auto output_shape = _session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        _output_size = output_shape[1];

        std::cout << "Model loaded successfully" << std::endl;
        std::cout << "  Input size: " << _input_size << std::endl;
        std::cout << "  Output size: " << _output_size << std::endl;

        _load_scaler_params(scaler_path);
    }

    int predict(const std::vector<float>& input)
    {
        std::vector<float> logits = _run_inference(input);
        std::vector<float> probabilities = _apply_softmax(logits);

        int predicted_class = 0;
        float max_prob = probabilities[0];
        for (size_t i = 1; i < probabilities.size(); ++i)
        {
            if (probabilities[i] > max_prob)
            {
                max_prob = probabilities[i];
                predicted_class = static_cast<int>(i);
            }
        }

        return predicted_class;
    }

    std::vector<float> predict_proba(const std::vector<float>& input)
    {
        std::vector<float> logits = _run_inference(input);
        return _apply_softmax(logits);
    }

    std::vector<float> predict_logits(const std::vector<float>& input)
    {
        return _run_inference(input);
    }

    ~MLPInference()
    {
        for (auto name : _input_names)
        {
            delete[] name;
        }
        for (auto name : _output_names)
        {
            delete[] name;
        }
    }
};

int main()
{
    try
    {
        MLPInference mlp_inference("mlp_classifier.onnx", "scaler_params.json");

        std::vector<float> test_input = {0.5f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.1f, 0.7f, 0.5f};

        int predicted_class = mlp_inference.predict(test_input);
        std::cout << "\nPrediction result:" << std::endl;
        std::cout << "  Predicted class: " << predicted_class << std::endl;

        std::vector<float> probabilities = mlp_inference.predict_proba(test_input);
        std::cout << "  Probabilities: ";
        for (size_t i = 0; i < probabilities.size(); ++i)
        {
            std::cout << "Class " << i << ": " << probabilities[i];
            if (i < probabilities.size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        std::vector<float> logits = mlp_inference.predict_logits(test_input);
        std::cout << "  Raw logits: ";
        for (size_t i = 0; i < logits.size(); ++i)
        {
            std::cout << "Class " << i << ": " << logits[i];
            if (i < logits.size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        float prob_sum = 0.0f;
        for (float p : probabilities)
        {
            prob_sum += p;
        }
        std::cout << "  Probability sum: " << prob_sum << " (should be ~1.0)" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


