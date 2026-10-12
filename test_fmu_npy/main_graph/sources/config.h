#ifndef config_h
#define config_h

#include "onnxruntime_c_api.h"
#include <stdint.h>

// define class name and unique id
#define MODEL_IDENTIFIER main_graph
#define INSTANTIATION_TOKEN "a7463883-3e91-4f47-8ebe-2a410a126aed"

#define CO_SIMULATION

#define GET_FLOAT64
#define SET_FLOAT64
#if FMI_VERSION > 2
#define GET_FLOAT32
#define SET_FLOAT32
#define GET_INT64
#define SET_INT64
#endif

#define FIXED_SOLVER_STEP 1
#define DEFAULT_STOP_TIME 1

typedef enum {
    // Always include time
    vr_time,
    vr_load,
    vr_transformer,
    vr_load_phase_transformer,
    vr_transformer_revphase_load,
    vr_load_phase_transformer_label,
    vr_logits,
} ValueReference;

typedef struct {
    // Always include time
    double time;
    float load[79200];
    float transformer[4320];
    int64_t load_phase_transformer[330];
    int64_t transformer_revphase_load[330];
    int64_t load_phase_transformer_label[330];
    float logits[165];
} ModelData;

typedef struct {
    OrtMemoryInfo* memory_info;
    float* load_data;
    OrtValue* load_tensor;
    float* transformer_data;
    OrtValue* transformer_tensor;
    int64_t* load_phase_transformer_data;
    OrtValue* load_phase_transformer_tensor;
    int64_t* transformer_revphase_load_data;
    OrtValue* transformer_revphase_load_tensor;
    int64_t* load_phase_transformer_label_data;
    OrtValue* load_phase_transformer_label_tensor;
} ModelBuffers;

#endif /* config_h */