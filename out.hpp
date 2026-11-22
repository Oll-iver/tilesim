#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "gemm.hpp"
#include "machine.hpp"
#include "sim.hpp"

namespace tilesim {

struct stragglerExperiment {
    double sigma = 0.0;
    std::int64_t steps = 0;
    std::int64_t seeds = 0;
    std::int64_t cycles = 0;
};

struct stragglerOutcome {
    double measuredOverhead = 0.0;
    double predictedOverhead = 0.0;
    double expectedMaxNormal = 0.0;
};

struct runRecord {
    std::string model;
    std::string workload;
    bool hasShape = false;
    gemmShape shape;
    bool hasPlan = false;
    gemmPlan plan;
    std::int64_t steps = 0;
    double jitter = 0.0;
    std::uint64_t seed = 0;
    std::int64_t flops = 0;
    phaseCycles cycles;
};

struct Metrics {
    double seconds = 0.0;
    double tflops = 0.0;
    double utilisation = 0.0;
    double computeShare = 0.0;
    double syncShare = 0.0;
    double exchangeShare = 0.0;
};

Metrics metrics(const Accelerator& acc, std::int64_t flops, const phaseCycles& cycles);

std::string describeMachine(const Accelerator& acc);
std::string formatTable(const Accelerator& acc, const std::vector<runRecord>& runs,
                        bool showWorkload);
std::string csvHeader();
std::string csvRow(const Accelerator& acc, const runRecord& run);

std::string formatTable(const Accelerator& acc, const gemmShape& shape,
                        const std::vector<planEval>& plans);

std::string stragglerHeader();
std::string stragglerRow(const Accelerator& acc, const stragglerExperiment& experiment,
                         const stragglerOutcome& outcome);
std::string formatStraggler(const Accelerator& acc, const stragglerExperiment& experiment,
                            const stragglerOutcome& outcome);

class traceWriter {
  public:
    traceWriter(const std::string& path, double clock, std::int64_t traced);
    ~traceWriter();

    bool traces(std::int64_t tile) const { return tile >= 0 && tile < traced; }

    void compute(std::int64_t tile, std::int64_t start, std::int64_t end, std::int64_t superstep);
    void send(std::int64_t tile, std::int64_t start, std::int64_t end, std::int64_t bytes);
    void receive(std::int64_t tile, std::int64_t start, std::int64_t end, std::int64_t bytes);
    void barrier(std::int64_t start, std::int64_t end, std::int64_t superstep);

    void close();

  private:
    void writeEvent(const std::string& name, std::int64_t tid, std::int64_t start, std::int64_t end,
                    const std::string& argKey, std::int64_t argVal);

    std::string path;
    std::ofstream out;
    double cyclesPer = 1000.0;
    std::int64_t traced = 0;
};

} // namespace tilesim