#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "api/generation_config.h"
#include "api/llm_engine.h"
#include "backend/device_type.h"

#include <memory>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

class PyPcEngine {
public:
    PyPcEngine() : engine_(std::make_unique<LLMEngine>()) {}

    void load(const std::string& model_dir, const std::string& device) {
        std::fprintf(stderr, "[bindings/pc_engine] load enter device=%s model_dir=%s\n",
                     device.c_str(), model_dir.c_str());
        const ComputeDevice parsed = parse_compute_device(device);
        if (!engine_->load(model_dir, parsed)) {
            throw std::runtime_error("failed to load worker-pc model");
        }
        std::fprintf(stderr, "[bindings/pc_engine] load leave\n");
    }

    void reset() {
        engine_->reset();
    }

    void destroy() {
        engine_->destroy();
    }

    GenerationResult generate(const std::vector<int>& input_ids,
                              int max_new_tokens,
                              int repetition_window,
                              const std::vector<int>& stop_tokens) {
        std::fprintf(stderr,
                     "[bindings/pc_engine] generate enter input_tokens=%d max_new_tokens=%d\n",
                     static_cast<int>(input_ids.size()), max_new_tokens);
        GenerationConfig cfg;
        cfg.max_new_tokens = max_new_tokens;
        cfg.repetition_window = repetition_window;
        if (!stop_tokens.empty()) {
            cfg.stop_tokens = stop_tokens;
        }
        GenerationResult result = engine_->generate(input_ids, cfg, nullptr);
        std::fprintf(stderr,
                     "[bindings/pc_engine] generate leave output_tokens=%d\n",
                     static_cast<int>(result.output_ids.size()));
        return result;
    }

private:
    std::unique_ptr<LLMEngine> engine_;
};

}  // namespace

PYBIND11_MODULE(pc_engine, m) {
    m.doc() = "pybind11 binding for worker-pc LLMEngine";

    py::class_<GenerationResult>(m, "GenerationResult")
        .def_readonly("output_ids", &GenerationResult::output_ids)
        .def_readonly("prefill_tokens", &GenerationResult::prefill_tokens)
        .def_readonly("decode_tokens", &GenerationResult::decode_tokens)
        .def_readonly("prefill_ms", &GenerationResult::prefill_ms)
        .def_readonly("decode_ms", &GenerationResult::decode_ms)
        .def_readonly("hit_stop", &GenerationResult::hit_stop)
        .def_readonly("hit_repetition", &GenerationResult::hit_repetition);

    py::class_<PyPcEngine>(m, "Engine")
        .def(py::init<>())
        .def("load", &PyPcEngine::load, py::arg("model_dir"), py::arg("device") = "cpu")
        .def("reset", &PyPcEngine::reset)
        .def("destroy", &PyPcEngine::destroy)
        .def("generate",
             &PyPcEngine::generate,
             py::arg("input_ids"),
             py::arg("max_new_tokens") = 10,
             py::arg("repetition_window") = 6,
             py::arg("stop_tokens") = std::vector<int>{});
}
