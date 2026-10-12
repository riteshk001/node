#include"config.h"
#include "model.h"
#include "onnxruntime_c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _WIN32
#include <windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#endif

#define MAX_PATH_LENGTH 4096

#define ORT_ABORT_ON_ERROR(expr, comp) \
    do { \
    OrtStatus* onnx_status = (expr); \
    if (onnx_status != NULL) { \
            const char* msg = comp->g_ort->GetErrorMessage(onnx_status); \
            logError(comp, "%s\n", msg); \
            comp->g_ort->ReleaseStatus(onnx_status); \
        } \
    } while (0);

void setStartValues(ModelInstance *comp) {
    // Input variables
    for (size_t i = 0; i < 79200; ++i) {
        M(load)[i] = (float) 1.0;
    }
    for (size_t i = 0; i < 4320; ++i) {
        M(transformer)[i] = (float) 1.0;
    }
    for (size_t i = 0; i < 330; ++i) {
        M(load_phase_transformer)[i] = (int64_t) 0;
    }
    for (size_t i = 0; i < 330; ++i) {
        M(transformer_revphase_load)[i] = (int64_t) 0;
    }
    for (size_t i = 0; i < 330; ++i) {
        M(load_phase_transformer_label)[i] = (int64_t) 0;
    }
    // Local variables
}

Status initModel(ModelInstance *comp) {

    ORT_ABORT_ON_ERROR(
        comp->g_ort->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &comp->modelBuffers.memory_info
        ),
        comp);

    // Allocate persistent buffer for load
    const size_t load_shape[] = { 55, 1440 };
    const size_t load_dim = sizeof(load_shape) / sizeof(load_shape[0]);
    size_t load_size = 1;
    for (size_t i = 0; i < load_dim; ++i) {
        load_size *= load_shape[i];
    }
    comp->modelBuffers.load_data = (float*)calloc(load_size, sizeof(float));
    if (comp->modelBuffers.load_data == NULL) {
        logError(comp, "Failed to allocate memory for load_data");
        return Error;
    }
    ORT_ABORT_ON_ERROR(
        comp->g_ort->CreateTensorWithDataAsOrtValue(
            comp->modelBuffers.memory_info,
            comp->modelBuffers.load_data,
            load_size * sizeof(float),
            (const int64_t*)load_shape,
            load_dim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &comp->modelBuffers.load_tensor
        ),
        comp);

    int load_is_tensor;
    ORT_ABORT_ON_ERROR(
        comp->g_ort->IsTensor(
            comp->modelBuffers.load_tensor,
            &load_is_tensor),
            comp
        );
    assert(load_is_tensor);

    // Allocate persistent buffer for transformer
    const size_t transformer_shape[] = { 3, 1440 };
    const size_t transformer_dim = sizeof(transformer_shape) / sizeof(transformer_shape[0]);
    size_t transformer_size = 1;
    for (size_t i = 0; i < transformer_dim; ++i) {
        transformer_size *= transformer_shape[i];
    }
    comp->modelBuffers.transformer_data = (float*)calloc(transformer_size, sizeof(float));
    if (comp->modelBuffers.transformer_data == NULL) {
        logError(comp, "Failed to allocate memory for transformer_data");
        return Error;
    }
    ORT_ABORT_ON_ERROR(
        comp->g_ort->CreateTensorWithDataAsOrtValue(
            comp->modelBuffers.memory_info,
            comp->modelBuffers.transformer_data,
            transformer_size * sizeof(float),
            (const int64_t*)transformer_shape,
            transformer_dim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &comp->modelBuffers.transformer_tensor
        ),
        comp);

    int transformer_is_tensor;
    ORT_ABORT_ON_ERROR(
        comp->g_ort->IsTensor(
            comp->modelBuffers.transformer_tensor,
            &transformer_is_tensor),
            comp
        );
    assert(transformer_is_tensor);

    // Allocate persistent buffer for load_phase_transformer
    const size_t load_phase_transformer_shape[] = { 2, 165 };
    const size_t load_phase_transformer_dim = sizeof(load_phase_transformer_shape) / sizeof(load_phase_transformer_shape[0]);
    size_t load_phase_transformer_size = 1;
    for (size_t i = 0; i < load_phase_transformer_dim; ++i) {
        load_phase_transformer_size *= load_phase_transformer_shape[i];
    }
    comp->modelBuffers.load_phase_transformer_data = (int64_t*)calloc(load_phase_transformer_size, sizeof(int64_t));
    if (comp->modelBuffers.load_phase_transformer_data == NULL) {
        logError(comp, "Failed to allocate memory for load_phase_transformer_data");
        return Error;
    }
    ORT_ABORT_ON_ERROR(
        comp->g_ort->CreateTensorWithDataAsOrtValue(
            comp->modelBuffers.memory_info,
            comp->modelBuffers.load_phase_transformer_data,
            load_phase_transformer_size * sizeof(int64_t),
            (const int64_t*)load_phase_transformer_shape,
            load_phase_transformer_dim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &comp->modelBuffers.load_phase_transformer_tensor
        ),
        comp);

    int load_phase_transformer_is_tensor;
    ORT_ABORT_ON_ERROR(
        comp->g_ort->IsTensor(
            comp->modelBuffers.load_phase_transformer_tensor,
            &load_phase_transformer_is_tensor),
            comp
        );
    assert(load_phase_transformer_is_tensor);

    // Allocate persistent buffer for transformer_revphase_load
    const size_t transformer_revphase_load_shape[] = { 2, 165 };
    const size_t transformer_revphase_load_dim = sizeof(transformer_revphase_load_shape) / sizeof(transformer_revphase_load_shape[0]);
    size_t transformer_revphase_load_size = 1;
    for (size_t i = 0; i < transformer_revphase_load_dim; ++i) {
        transformer_revphase_load_size *= transformer_revphase_load_shape[i];
    }
    comp->modelBuffers.transformer_revphase_load_data = (int64_t*)calloc(transformer_revphase_load_size, sizeof(int64_t));
    if (comp->modelBuffers.transformer_revphase_load_data == NULL) {
        logError(comp, "Failed to allocate memory for transformer_revphase_load_data");
        return Error;
    }
    ORT_ABORT_ON_ERROR(
        comp->g_ort->CreateTensorWithDataAsOrtValue(
            comp->modelBuffers.memory_info,
            comp->modelBuffers.transformer_revphase_load_data,
            transformer_revphase_load_size * sizeof(int64_t),
            (const int64_t*)transformer_revphase_load_shape,
            transformer_revphase_load_dim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &comp->modelBuffers.transformer_revphase_load_tensor
        ),
        comp);

    int transformer_revphase_load_is_tensor;
    ORT_ABORT_ON_ERROR(
        comp->g_ort->IsTensor(
            comp->modelBuffers.transformer_revphase_load_tensor,
            &transformer_revphase_load_is_tensor),
            comp
        );
    assert(transformer_revphase_load_is_tensor);

    // Allocate persistent buffer for load_phase_transformer_label
    const size_t load_phase_transformer_label_shape[] = { 2, 165 };
    const size_t load_phase_transformer_label_dim = sizeof(load_phase_transformer_label_shape) / sizeof(load_phase_transformer_label_shape[0]);
    size_t load_phase_transformer_label_size = 1;
    for (size_t i = 0; i < load_phase_transformer_label_dim; ++i) {
        load_phase_transformer_label_size *= load_phase_transformer_label_shape[i];
    }
    comp->modelBuffers.load_phase_transformer_label_data = (int64_t*)calloc(load_phase_transformer_label_size, sizeof(int64_t));
    if (comp->modelBuffers.load_phase_transformer_label_data == NULL) {
        logError(comp, "Failed to allocate memory for load_phase_transformer_label_data");
        return Error;
    }
    ORT_ABORT_ON_ERROR(
        comp->g_ort->CreateTensorWithDataAsOrtValue(
            comp->modelBuffers.memory_info,
            comp->modelBuffers.load_phase_transformer_label_data,
            load_phase_transformer_label_size * sizeof(int64_t),
            (const int64_t*)load_phase_transformer_label_shape,
            load_phase_transformer_label_dim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &comp->modelBuffers.load_phase_transformer_label_tensor
        ),
        comp);

    int load_phase_transformer_label_is_tensor;
    ORT_ABORT_ON_ERROR(
        comp->g_ort->IsTensor(
            comp->modelBuffers.load_phase_transformer_label_tensor,
            &load_phase_transformer_label_is_tensor),
            comp
        );
    assert(load_phase_transformer_label_is_tensor);

    return OK;
}

void freeModel(ModelInstance *comp) {
    if (comp->modelBuffers.load_tensor)
        comp->g_ort->ReleaseValue(comp->modelBuffers.load_tensor);
    if (comp->modelBuffers.load_data)
        free(comp->modelBuffers.load_data);
    if (comp->modelBuffers.transformer_tensor)
        comp->g_ort->ReleaseValue(comp->modelBuffers.transformer_tensor);
    if (comp->modelBuffers.transformer_data)
        free(comp->modelBuffers.transformer_data);
    if (comp->modelBuffers.load_phase_transformer_tensor)
        comp->g_ort->ReleaseValue(comp->modelBuffers.load_phase_transformer_tensor);
    if (comp->modelBuffers.load_phase_transformer_data)
        free(comp->modelBuffers.load_phase_transformer_data);
    if (comp->modelBuffers.transformer_revphase_load_tensor)
        comp->g_ort->ReleaseValue(comp->modelBuffers.transformer_revphase_load_tensor);
    if (comp->modelBuffers.transformer_revphase_load_data)
        free(comp->modelBuffers.transformer_revphase_load_data);
    if (comp->modelBuffers.load_phase_transformer_label_tensor)
        comp->g_ort->ReleaseValue(comp->modelBuffers.load_phase_transformer_label_tensor);
    if (comp->modelBuffers.load_phase_transformer_label_data)
        free(comp->modelBuffers.load_phase_transformer_label_data);
    if (comp->modelBuffers.memory_info)
        comp->g_ort->ReleaseMemoryInfo(comp->modelBuffers.memory_info);
}

Status calculateValues(ModelInstance *comp) {

    // Update input buffers with current model values
    for (size_t i = 0; i < 79200; ++i) {
        comp->modelBuffers.load_data[i] = (float)M(load)[i];
    }
    for (size_t i = 0; i < 4320; ++i) {
        comp->modelBuffers.transformer_data[i] = (float)M(transformer)[i];
    }
    for (size_t i = 0; i < 330; ++i) {
        comp->modelBuffers.load_phase_transformer_data[i] = (int64_t)M(load_phase_transformer)[i];
    }
    for (size_t i = 0; i < 330; ++i) {
        comp->modelBuffers.transformer_revphase_load_data[i] = (int64_t)M(transformer_revphase_load)[i];
    }
    for (size_t i = 0; i < 330; ++i) {
        comp->modelBuffers.load_phase_transformer_label_data[i] = (int64_t)M(load_phase_transformer_label)[i];
    }

    // Update local variable buffers with current model values

    // Output tensors are allocated by Run() on each call
    OrtValue* logits_tensor = NULL;

    // Declare input node names
    const char* input_names[] = {
        "load",
        "transformer",
        "load_phase_transformer",
        "transformer_revphase_load",
        "load_phase_transformer_label"
    };

    // Declare output node names
    const char* output_names[] = {
        "logits"
    };

    // Gather persistent input tensors
    const OrtValue* input_tensors[] = {
        comp->modelBuffers.load_tensor,
        comp->modelBuffers.transformer_tensor,
        comp->modelBuffers.load_phase_transformer_tensor,
        comp->modelBuffers.transformer_revphase_load_tensor,
        comp->modelBuffers.load_phase_transformer_label_tensor
    };

    // Gather output tensors
    OrtValue* output_tensors[] = {
        logits_tensor
    };

    // Run inference
    ORT_ABORT_ON_ERROR(
        comp->g_ort->Run(
            comp->session,
            NULL,
            input_names,
            input_tensors,
            5,
            output_names,
            1,
            output_tensors
        ),
        comp
    );

    // Check output tensors to be tensors
    int logits_is_tensor;
    ORT_ABORT_ON_ERROR(
        comp->g_ort->IsTensor(
            output_tensors[0],
            &logits_is_tensor),
            comp
        );
    assert (logits_is_tensor);

    // Check local output tensors to be tensors

    // Retrieve pointer to the logits tensor
    float* logits_tensor_data = NULL;
    ORT_ABORT_ON_ERROR(
        comp->g_ort->GetTensorMutableData(
            output_tensors[0],
            (void**)&logits_tensor_data
        ),
        comp
    );

    // Set logits tensor data to model
    for (size_t i = 0; i < 165; ++i) {
        M(logits)[i] = (float)logits_tensor_data[i];
    }

    // Release output tensors (allocated by Run on each call)
    comp->g_ort->ReleaseValue(logits_tensor);

    return OK;
}

Status getFloat64(ModelInstance *comp, ValueReference vr, double values[],
                  size_t nValues, size_t *index) {

    switch (vr)
    {
    case vr_time:
        ASSERT_NVALUES(1);
        values[(*index)++] = M(time);
        return OK;
        // Inputs
        // Outputs
        // Locals
    default:
        logError(comp, "getFloat64: ValueReference %d not available.", vr);
        return Error;
    }
}

Status setFloat64(ModelInstance *comp, ValueReference vr, const double values[],
                  size_t nValues, size_t *index) {
    switch (vr)
    {
        case vr_time:
            ASSERT_NVALUES(1);
            M(time) = values[(*index)++];
            return OK;
        // Inputs
        // Locals
    default:
        logError(comp, "setFloat64: ValueReference %d not available.", vr);
        return Error;
    }
}

#if FMI_VERSION > 2
Status getFloat32(ModelInstance *comp, ValueReference vr, float values[],
                  size_t nValues, size_t *index) {

    switch (vr)
    {
        // Inputs
        case vr_load:
            ASSERT_NVALUES(79200);
            for (size_t i = 0; i < 79200; ++i) {
                values[(*index)++] = (float)M(load)[i];
            }
            return OK;
        case vr_transformer:
            ASSERT_NVALUES(4320);
            for (size_t i = 0; i < 4320; ++i) {
                values[(*index)++] = (float)M(transformer)[i];
            }
            return OK;
        // Outputs
        case vr_logits:
            ASSERT_NVALUES(165);
            for (size_t i = 0; i < 165; ++i) {
                values[(*index)++] = (float)M(logits)[i];
            }
            return OK;
        // Locals
    default:
        logError(comp, "getFloat32: ValueReference %d not available.", vr);
        return Error;
    }
}
Status setFloat32(ModelInstance *comp, ValueReference vr, const float values[],
                  size_t nValues, size_t *index) {
    switch (vr)
    {
        // Inputs
        case vr_load:
            ASSERT_NVALUES(79200);
            for (size_t i = 0; i < 79200; ++i) {
                M(load)[i] = (float)values[(*index)++];
            }
            return OK;
        case vr_transformer:
            ASSERT_NVALUES(4320);
            for (size_t i = 0; i < 4320; ++i) {
                M(transformer)[i] = (float)values[(*index)++];
            }
            return OK;
        // Locals
    default:
        logError(comp, "setFloat32: ValueReference %d not available.", vr);
        return Error;
    }
}
Status getInt64(ModelInstance *comp, ValueReference vr, int64_t values[],
                size_t nValues, size_t *index) {

    switch (vr)
    {
        // Inputs
        case vr_load_phase_transformer:
            ASSERT_NVALUES(330);
            for (size_t i = 0; i < 330; ++i) {
                values[(*index)++] = M(load_phase_transformer)[i];
            }
            return OK;
        case vr_transformer_revphase_load:
            ASSERT_NVALUES(330);
            for (size_t i = 0; i < 330; ++i) {
                values[(*index)++] = M(transformer_revphase_load)[i];
            }
            return OK;
        case vr_load_phase_transformer_label:
            ASSERT_NVALUES(330);
            for (size_t i = 0; i < 330; ++i) {
                values[(*index)++] = M(load_phase_transformer_label)[i];
            }
            return OK;
        // Outputs
        // Locals
    default:
        logError(comp, "getInt64: ValueReference %d not available.", vr);
        return Error;
    }
}
Status setInt64(ModelInstance *comp, ValueReference vr, const int64_t values[],
                size_t nValues, size_t *index) {
    switch (vr)
    {
        // Inputs
        case vr_load_phase_transformer:
            ASSERT_NVALUES(330);
            for (size_t i = 0; i < 330; ++i) {
                M(load_phase_transformer)[i] = values[(*index)++];
            }
            return OK;
        case vr_transformer_revphase_load:
            ASSERT_NVALUES(330);
            for (size_t i = 0; i < 330; ++i) {
                M(transformer_revphase_load)[i] = values[(*index)++];
            }
            return OK;
        case vr_load_phase_transformer_label:
            ASSERT_NVALUES(330);
            for (size_t i = 0; i < 330; ++i) {
                M(load_phase_transformer_label)[i] = values[(*index)++];
            }
            return OK;
        // Locals
    default:
        logError(comp, "setInt64: ValueReference %d not available.", vr);
        return Error;
    }
}
#endif

Status eventUpdate(ModelInstance *comp) {

    comp->valuesOfContinuousStatesChanged   = false;
    comp->nominalsOfContinuousStatesChanged = false;
    comp->terminateSimulation               = false;
    comp->nextEventTimeDefined              = false;

    return OK;
}