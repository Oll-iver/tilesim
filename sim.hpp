#pragma once

#include <cstdint>

#include "machine.hpp"
#include "program.hpp"

namespace tilesim {

class traceWriter;


struct SimOptions {
    double jitterSigma = 0.0;
    std::uint64_t seed = 1;
    traceWriter* trace = nullptr;
};

Result simulate(const Accelerator& acc, const Program& program, const SimOptions& options = {});

double expectedMaxNormal(std::int64_t n);
double expectedMaxMulti(std::int64_t n, double sigma);
double predictedOverhead(std::int64_t n, double sigma, std::int64_t workCycles,
                         std::int64_t overheadCycles);

} // namespace tilesim