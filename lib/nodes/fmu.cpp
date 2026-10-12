/* Node type: Functional Mock-up Unit.
 *
 * Author: Ritesh Karki <ritesh.karki@rwth-aachen.de>, Jitpanu Maneeratpongsuk <jitpanu.maneeratpongsuk@rwth-aachen.de>
 * SPDX-FileCopyrightText: 2026 Institute for Automation of Complex Power Systems, RWTH Aachen University
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <FMI3/fmi3_enums.h>
#include <FMI3/fmi3_function_types.h>
#include <FMI3/fmi3_import.h>
#include <FMI3/fmi3_import_capi.h>
#include <FMI3/fmi3_import_variable.h>
#include <FMI3/fmi3_types.h>
#include <JM/jm_callbacks.h>
#include <JM/jm_types.h>
#include <cnpy.h>
#include <fmilib.h>
#include <pthread.h>
#include <sys/stat.h>

#include <villas/exceptions.hpp>
#include <villas/log.hpp>
#include <villas/node.hpp>
#include <villas/nodes/fmu.hpp>
#include <villas/timing.hpp>

using namespace villas;
using namespace villas::node;


double sampleValueAsDouble(const struct Sample *smp, size_t idx) {
  if (!smp->signals)
    return smp->data[idx].f;

  auto sig = smp->signals->getByIndex(idx);
  if (!sig)
    return smp->data[idx].f;

  switch (sig->type) {
  case SignalType::FLOAT:
    return smp->data[idx].f;
  case SignalType::INTEGER:
    return smp->data[idx].i;
  case SignalType::BOOLEAN:
    return smp->data[idx].b;
  default:
    return smp->data[idx].f;
  }
}

void storeDoubleToSample(struct Sample *smp, size_t idx, double val) {
  if (smp->signals) {
    auto sig = smp->signals->getByIndex(idx);
    if (sig) {
      switch (sig->type) {
      case SignalType::FLOAT:
        smp->data[idx].f = val;
        return;
      case SignalType::INTEGER:
        smp->data[idx].i = static_cast<int64_t>(val);
        return;
      case SignalType::BOOLEAN:
        smp->data[idx].b = static_cast<bool>(val);
        return;
      default:
        break;
      }
    }
  }

  smp->data[idx].f = val;
}


void FmuNode::loggerCallback(jm_callbacks *c, jm_string module,
                             jm_log_level_enu_t log_level, jm_string message) {
  auto *self = static_cast<FmuNode *>(c->context);
  if (self) {
    self->logger->info("{} {}", module, message);
  }
}

FmuNode::FmuNode(const uuid_t &id, const std::string &name)
    : Node(id, name), writing_turn(false), path(""), unpackPath(""), fmu(),
      context() {
  int ret;
  ret = pthread_mutex_init(&mutex, nullptr);
  if (ret)
    throw RuntimeError("Failed to initialize mutex");
  ret = pthread_cond_init(&cv, nullptr);
  if (ret)
    throw RuntimeError("Failed to initialize conditional variable");

  // Initialize callbacks
  callbacks.malloc = malloc;
  callbacks.calloc = calloc;
  callbacks.realloc = realloc;
  callbacks.free = free;
  callbacks.logger = FmuNode::loggerCallback;
  callbacks.log_level = jm_log_level_info;
  callbacks.context = static_cast<FmuNode *>(this);

  // Allocate FMI context
  context = fmi_import_allocate_context(&callbacks);
}

void FmuNode::resolveFmuVariable(FmuSignal &s) {
  auto *var = fmi3_import_get_variable_by_name(fmu, s.name.c_str());

  if (!var)
    throw RuntimeError("Invalid variable {}", s.name);

  s.type = fmi3_import_get_variable_base_type(var);
  s.ref = fmi3_import_get_variable_vr(var);
  s.dims.clear();
  s.num_elem = 1;

  if (fmi3_import_variable_is_array(var)) {
    auto *dim_list = fmi3_import_get_variable_dimension_list(var);
    size_t n = fmi3_import_get_dimension_list_size(dim_list);
    for (size_t i = 0; i < n; ++i) {
      auto *dim = fmi3_import_get_dimension(dim_list, i);
      if (!fmi3_import_get_dimension_has_start(dim)) {
        throw RuntimeError("Runtime dimension not supported");
      }
      size_t ds = fmi3_import_get_dimension_start(dim);
      s.dims.push_back(ds);
      s.num_elem *= ds;
    }
  }
}

void FmuNode::setFmuFromNpy(const FmuSignal &s) {
  if (s.npy_path.empty())
    throw RuntimeError("FMU variable '{}' uses npy source but has no npy path",
                       s.name);

  cnpy::NpyArray arr = cnpy::npy_load(s.npy_path);
  if (arr.num_vals != s.num_elem) {
    throw RuntimeError(
        "NPY '{}' has {} values, but FMU variable '{}' expects {}", s.npy_path,
        arr.num_vals, s.name, s.num_elem);
  }

  fmi3_value_reference_t vr = s.ref;
  fmi3_status_t status = fmi3_status_ok;
  switch (s.type) {
    using enum fmi3_base_type_enu_t;
  case fmi3_base_type_float64:
    status =
        fmi3_import_set_float64(fmu, &vr, 1, arr.data<double>(), s.num_elem);
    break;
  case fmi3_base_type_float32:
    status =
        fmi3_import_set_float32(fmu, &vr, 1, arr.data<float>(), s.num_elem);
    break;
  case fmi3_base_type_int64:
    status = fmi3_import_set_int64(fmu, &vr, 1, arr.data<fmi3_int64_t>(),
                                   s.num_elem);
    break;
  case fmi3_base_type_int32:
    status = fmi3_import_set_int32(fmu, &vr, 1, arr.data<fmi3_int32_t>(),
                                   s.num_elem);
    break;
  case fmi3_base_type_bool:
    status = fmi3_import_set_boolean(fmu, &vr, 1, arr.data<fmi3_boolean_t>(),
                                     s.num_elem);
    break;
  default:
    throw RuntimeError("Unsupported FMU type for npy input '{}'", s.name);
  }

  if (status == fmi3_status_error || status == fmi3_status_fatal)
    throw RuntimeError("Failed to set FMU variable '{}' from npy '{}'", s.name,
                       s.npy_path);
}

void FmuNode::setFmuFromSample(const FmuSignal &s, const struct Sample *smp) {
  if (s.sample_offset + s.num_elem > smp->length) {
    throw RuntimeError(
        "Sample too short for FMU input '{}': need {} values from offset {}, "
        "have length {}",
        s.name, s.num_elem, s.sample_offset, smp->length);
  }

  fmi3_value_reference_t vr = s.ref;
  fmi3_status_t status = fmi3_status_ok;
  switch (s.type) {
    using enum fmi3_base_type_enu_t;
  case fmi3_base_type_float64: {
    std::vector<fmi3_float64_t> buf(s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      buf[i] = sampleValueAsDouble(smp, s.sample_offset + i);
    status = fmi3_import_set_float64(fmu, &vr, 1, buf.data(), s.num_elem);
    break;
  }
  case fmi3_base_type_float32: {
    std::vector<fmi3_float32_t> buf(s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      buf[i] = static_cast<fmi3_float32_t>(
          sampleValueAsDouble(smp, s.sample_offset + i));
    status = fmi3_import_set_float32(fmu, &vr, 1, buf.data(), s.num_elem);
    break;
  }
  case fmi3_base_type_int64: {
    std::vector<fmi3_int64_t> buf(s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      buf[i] = static_cast<fmi3_int64_t>(
          floor(sampleValueAsDouble(smp, s.sample_offset + i)));
    status = fmi3_import_set_int64(fmu, &vr, 1, buf.data(), s.num_elem);
    break;
  }
  case fmi3_base_type_int32: {
    std::vector<fmi3_int32_t> buf(s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      buf[i] = static_cast<fmi3_int32_t>(
          floor(sampleValueAsDouble(smp, s.sample_offset + i)));
    status = fmi3_import_set_int32(fmu, &vr, 1, buf.data(), s.num_elem);
    break;
  }
  case fmi3_base_type_bool: {
    // std::vector<fmi3_int8_t> buf(s.num_elem);
    auto buf = std::make_unique<fmi3_boolean_t[]>(s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      buf[i] = static_cast<fmi3_int8_t>(
          sampleValueAsDouble(smp, s.sample_offset + i));
    status = fmi3_import_set_boolean(fmu, &vr, 1, buf.get(), s.num_elem);
    break;
  }
  default:
    throw RuntimeError("Unsupported FMU type for node input '{}'", s.name);
  }

  if (status == fmi3_status_error || status == fmi3_status_fatal)
    throw RuntimeError("Failed to set FMU variable '{}' from sample", s.name);
}

void FmuNode::writeToNpy(const FmuSignal &s) {
  if (s.npy_path.empty())
    throw RuntimeError("FMU variable '{}' uses npy source but has no npy path",
                       s.name);

  auto shape = s.dims.empty() ? std::vector<size_t>{1} : s.dims;
  fmi3_value_reference_t vr = s.ref;
  fmi3_status_t status = fmi3_status_ok;

  switch (s.type) {
    using enum fmi3_base_type_enu_t;
  case fmi3_base_type_float64: {
    std::vector<fmi3_float64_t> buf(s.num_elem);
    status = fmi3_import_get_float64(fmu, &vr, 1, buf.data(), s.num_elem);
    if (status != fmi3_status_error && status != fmi3_status_fatal)
      cnpy::npy_save(s.npy_path, buf.data(), shape, "w");
    break;
  }
  case fmi3_base_type_float32: {
    std::vector<fmi3_float32_t> buf(s.num_elem);
    status = fmi3_import_get_float32(fmu, &vr, 1, buf.data(), s.num_elem);
    if (status != fmi3_status_error && status != fmi3_status_fatal)
      cnpy::npy_save(s.npy_path, buf.data(), shape, "w");
    break;
  }
  case fmi3_base_type_int64: {
    std::vector<fmi3_int64_t> buf(s.num_elem);
    status = fmi3_import_get_int64(fmu, &vr, 1, buf.data(), s.num_elem);
    if (status != fmi3_status_error && status != fmi3_status_fatal)
      cnpy::npy_save(s.npy_path, buf.data(), shape, "w");
    break;
  }
  case fmi3_base_type_int32: {
    std::vector<fmi3_int32_t> buf(s.num_elem);
    status = fmi3_import_get_int32(fmu, &vr, 1, buf.data(), s.num_elem);
    if (status != fmi3_status_error && status != fmi3_status_fatal)
      cnpy::npy_save(s.npy_path, buf.data(), shape, "w");
    break;
  }
  case fmi3_base_type_bool: {
    // std::vector<fmi3_boolean_t> buf(s.num_elem);
    auto buf = std::make_unique<fmi3_boolean_t[]>(s.num_elem);
    status = fmi3_import_get_boolean(fmu, &vr, 1, buf.get(), s.num_elem);
    if (status != fmi3_status_error && status != fmi3_status_fatal)
      cnpy::npy_save(s.npy_path, buf.get(), shape, "w");
    break;
  }
  default:
    throw RuntimeError("Unsupported FMU type for npy output '{}'", s.name);
  }

  if (status == fmi3_status_error || status == fmi3_status_fatal)
    throw RuntimeError("Failed to get FMU variable '{}' for npy '{}'", s.name,
                       s.npy_path);
}

size_t FmuNode::getFmuToSample(const FmuSignal &s, struct Sample *smp,
                               size_t offset) {
  if (offset + s.num_elem > smp->capacity) {
    throw RuntimeError(
        "Sample capacity {} too small for FMU output '{}' ({} values at "
        "offset {})",
        smp->capacity, s.name, s.num_elem, offset);
  }

  fmi3_value_reference_t vr = s.ref;
  fmi3_status_t status = fmi3_status_ok;
  switch (s.type) {
    using enum fmi3_base_type_enu_t;
  case fmi3_base_type_float64: {
    std::vector<fmi3_float64_t> buf(s.num_elem);
    status = fmi3_import_get_float64(fmu, &vr, 1, buf.data(), s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      storeDoubleToSample(smp, offset + i, buf[i]);
    break;
  }
  case fmi3_base_type_float32: {
    std::vector<fmi3_float32_t> buf(s.num_elem);
    status = fmi3_import_get_float32(fmu, &vr, 1, buf.data(), s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      storeDoubleToSample(smp, offset + i, buf[i]);
    break;
  }
  case fmi3_base_type_int64: {
    std::vector<fmi3_int64_t> buf(s.num_elem);
    status = fmi3_import_get_int64(fmu, &vr, 1, buf.data(), s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      storeDoubleToSample(smp, offset + i, static_cast<double>(buf[i]));
    break;
  }
  case fmi3_base_type_int32: {
    std::vector<fmi3_int32_t> buf(s.num_elem);
    status = fmi3_import_get_int32(fmu, &vr, 1, buf.data(), s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      storeDoubleToSample(smp, offset + i, static_cast<double>(buf[i]));
    break;
  }
  case fmi3_base_type_bool: {
    // std::vector<fmi3_boolean_t> buf(s.num_elem);
    auto buf = std::make_unique<fmi3_boolean_t[]>(s.num_elem);
    status = fmi3_import_get_boolean(fmu, &vr, 1, buf.get(), s.num_elem);
    for (size_t i = 0; i < s.num_elem; ++i)
      storeDoubleToSample(smp, offset + i, static_cast<double>(buf[i]));
    break;
  }
  default:
    throw RuntimeError("Unsupported FMU type for node output '{}'", s.name);
  }

  if (status == fmi3_status_error || status == fmi3_status_fatal)
    throw RuntimeError("Failed to get FMU variable '{}' into sample", s.name);

  return s.num_elem;
}

int FmuNode::prepare() {
  struct stat sb;
  // Check if path is valid
  if (stat(path, &sb) != 0)
    logger->error("The FMU path is invalid: {}, {}", strerror(errno), path);

  if (stat(unpackPath, &sb) != 0)
    logger->error("The unpack path is invalid: {}, {}", strerror(errno),
                  unpackPath);

  fmi_version_enu_t version =
      fmi_import_get_fmi_version(context, path, unpackPath);
  if (version != fmi_version_enu_t::fmi_version_3_0_enu) {
    throw RuntimeError("Not FMI 3.0");
  }

  fmu = fmi3_import_parse_xml(context, unpackPath, 0);
  if (!fmu) {
    throw RuntimeError("Failed to parse xml file");
  }

  if (fmi3_import_create_dllfmu(fmu, fmi3_fmu_kind_cs, NULL, NULL) !=
      jm_status_success) {
    throw RuntimeError("Failed to load FMU binary");
  }

  // Hardwiring resource path to test. To be removed later.
  auto resource_path = std::string(unpackPath) + "/resources";
  if (fmi3_import_instantiate_co_simulation(
          fmu, "instance1", resource_path.c_str(), fmi3_false, fmi3_true, fmi3_false,
          fmi3_true, NULL, 0, NULL) != jm_status_success) {
    throw RuntimeError("Failed to instantiate FMU");
  }

  fmi3_import_enter_initialization_mode(fmu, fmi3_false, 0.0, start_time,
                                        stop_time_defined, stop_time);

  // Resolve VR, type and dimensions, then apply npy inputs while still in init.
  size_t in_offset = 0;
  hasNodeInputs = false;
  for (auto &sig : fmuSignalIn) {
    resolveFmuVariable(sig);
    if (sig.source == FmuSignalSource::FmuSignalFromNpy) {
      setFmuFromNpy(sig);
    } else {
      sig.sample_offset = in_offset;
      in_offset += sig.num_elem;
      hasNodeInputs = true;
    }
  }

  size_t out_offset = 0;
  for (auto &sig : fmuSignalOut) {
    resolveFmuVariable(sig);
    if (sig.source == FmuSignalSource::FmuSignalFromNode) {
      sig.sample_offset = out_offset;
      out_offset += sig.num_elem;
    }
  }

  fmi3_import_exit_initialization_mode(fmu);

  writing_turn = hasNodeInputs;

  return Node::prepare();
}

int FmuNode::check() { return Node::check(); }

int FmuNode::_read(struct Sample *smps[], unsigned cnt) {
  assert(cnt == 1);

  // Lock thread
  pthread_mutex_lock(&mutex);
  while (hasNodeInputs && (writing_turn || currentTime >= stop_time)) {
    pthread_cond_wait(&cv, &mutex);
  }
  double targetTime = currentTime + step_size;
  double remainingStep = step_size;
  smps[0]->signals = in.signals;

  // Stop node if current time exceeds the stop time.
  if (currentTime >= stop_time) {
    logger->info("Reached stop time {}", stop_time);
    setState(State::STOPPING);
    pthread_mutex_unlock(&mutex);
    return -1;
  }

  while (currentTime < targetTime && currentTime < stop_time) {
    bool eventHandlingNeeded = false;
    bool terminateSimulation = false;
    bool earlyReturn = false;
    double lastSuccessfulTime = currentTime;

    // Perform step
    fmi3_status_t status = fmi3_import_do_step(
        fmu, currentTime, remainingStep, fmi3_true, &eventHandlingNeeded,
        &terminateSimulation, &earlyReturn, &lastSuccessfulTime);

    if (status == fmi3_status_error || status == fmi3_status_fatal)
      throw RuntimeError("Error during step");

    if (terminateSimulation) {
      logger->info("FMU called for simulation termination");
      currentTime = lastSuccessfulTime;
      break;
    }

    currentTime = lastSuccessfulTime;
    logger->debug("Current time {}, step size: {}", currentTime, step_size);

    if (earlyReturn) {
      logger->debug("FMU returned early at {}, target time {}",
                    lastSuccessfulTime, targetTime);
      remainingStep = targetTime - currentTime;
    }
  }

  smps[0]->ts.origin = time_now();
  smps[0]->flags = (int)SampleFlags::HAS_DATA | (int)SampleFlags::HAS_TS_ORIGIN;

  size_t sample_len = 0;
  for (const auto &sig : fmuSignalOut) {
    if (sig.source == FmuSignalSource::FmuSignalFromNpy) {
      writeToNpy(sig);
    } else {
      sample_len += getFmuToSample(sig, smps[0], sig.sample_offset);
    }
  }
  smps[0]->length = sample_len;

  // Unlock thread
  if (hasNodeInputs) {
    writing_turn = true;
    pthread_cond_signal(&cv);
  }

  pthread_mutex_unlock(&mutex);

  return 1;
}

int FmuNode::_write(struct Sample *smps[], unsigned cnt) {
  assert(cnt == 1);

  if (fmuSignalIn.empty())
    return 0;

  // Npy-only inputs are applied during prepare(); refresh without turn sync.
  if (!hasNodeInputs) {
    for (const auto &sig : fmuSignalIn)
      setFmuFromNpy(sig);
    return 1;
  }

  // Lock thread
  pthread_mutex_lock(&mutex);
  while (!writing_turn) {
    pthread_cond_wait(&cv, &mutex);
  }

  ts = smps[0]->ts.origin;

  for (const auto &sig : fmuSignalIn) {
    if (sig.source == FmuSignalSource::FmuSignalFromNpy) {
      setFmuFromNpy(sig);
    } else {
      setFmuFromSample(sig, smps[0]);
      logger->debug("Set FMU input '{}' from sample ({} elements)", sig.name,
                    sig.num_elem);
    }
  }

  // Unlock thread
  writing_turn = false;
  pthread_cond_signal(&cv);
  pthread_mutex_unlock(&mutex);

  return 1;
}

void FmuNode::getValueReference(const char *varName,
                                fmi3_value_reference_t &ref,
                                fmi3_base_type_enu_t &type) {
  fmi3_import_variable_t *var = fmi3_import_get_variable_by_name(fmu, varName);
  if (!var) {
    throw SystemError("Invalid variable name '{}'", varName);
  }
  type = fmi3_import_get_variable_base_type(var);
  ref = fmi3_import_get_variable_vr(var);
}

void FmuNode::parseSignal(json_t *json, bool in) {
  if (!json)
    return;

  if (!json_is_array(json))
    throw ConfigError(json, "node-config-node-fmu-signals",
                      "FMU signals must be an array");

  size_t idx;
  json_t *json_sig;
  json_array_foreach(json, idx, json_sig) {
    const char *name = nullptr;
    const char *source_str = nullptr;
    const char *npy = nullptr;
    json_error_t err;

    int ret = json_unpack_ex(json_sig, &err, 0, "{s:s, s?:s, s?:s}", "name",
                             &name, "source", &source_str, "npy", &npy);
    if (ret)
      throw ConfigError(json_sig, err, "node-config-node-fmu-signal");

    FmuSignal s;
    s.name = name;

    if (source_str) {
      if (strcmp(source_str, "npy") == 0)
        s.source = FmuSignalSource::FmuSignalFromNpy;
      else if (strcmp(source_str, "node") == 0)
        s.source = FmuSignalSource::FmuSignalFromNode;
      else {
        throw ConfigError(json_sig, "node-config-node-fmu-signal-source",
                          "Invalid source '{}' for signal '{}' (use 'node' or "
                          "'npy')",
                          source_str, name);
      }
    }

    if (npy)
      s.npy_path = npy;

    if (s.source == FmuSignalSource::FmuSignalFromNpy && s.npy_path.empty()) {
      throw ConfigError(json_sig, "node-config-node-fmu-signal-npy",
                        "Signal '{}' has source=npy but no 'npy' path", name);
    }

    if (in)
      fmuSignalIn.push_back(s);
    else
      fmuSignalOut.push_back(s);
  }
}

int FmuNode::parse(json_t *json) {
  int ret = Node::parse(json);
  if (ret)
    return ret;

  json_t *json_signals_in = nullptr;
  json_t *json_signals_out = nullptr;
  json_error_t err;
  start_time = 0;
  step_size = 0.1;
  stop_time = INT_MAX;
  ret = json_unpack_ex(
      json, &err, 0, "{s:s, s:s, s?:f, s?:f, s?:f, s?:{s:o}, s?:{s:o}}",
      "fmu_path", &path, "fmu_unpack_path", &unpackPath, "stop_time",
      &stop_time, "start_time", &start_time, "step_size", &step_size, "in",
      "signals", &json_signals_in, "out", "signals", &json_signals_out);
  if (ret)
    throw ConfigError(json, err, "node-config-node-fmu");
  if (stop_time == INT_MAX) {
    stop_time_defined = false;
  } else {
    stop_time_defined = true;
  }

  fmuSignalIn.clear();
  fmuSignalOut.clear();
  parseSignal(json_signals_in, true);
  parseSignal(json_signals_out, false);

  // Swap in.signals and out.signals so that in.signals = FMU outputs and
  // out.signals = FMU inputs.
  swapSignals();

  // Wait for _write() only when at least one input comes from another node.
  hasNodeInputs = false;
  for (const auto &sig : fmuSignalIn) {
    if (sig.source == FmuSignalSource::FmuSignalFromNode) {
      hasNodeInputs = true;
      break;
    }
  }
  writing_turn = hasNodeInputs;

  return 0;
}

int FmuNode::start() { return Node::start(); }

int FmuNode::stop() {
  fmi3_import_terminate(fmu);
  fmi3_import_free_instance(fmu);
  fmi3_import_destroy_dllfmu(fmu);
  fmi3_import_free(fmu);
  fmi_import_free_context(context);
  return Node::stop();
}

// Register node
static char n[] = "fmu";
static char d[] = "Interface to Functional Mockup Unit using FMI3.0";
static NodePlugin<FmuNode, n, d,
                  (int)NodeFactory::Flags::SUPPORTS_READ |
                      (int)NodeFactory::Flags::SUPPORTS_WRITE>
    p;
