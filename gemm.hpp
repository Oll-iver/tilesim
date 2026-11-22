#pragma once
#include <cstdint>
#include <vector>

#include "machine.hpp"
#include "program.hpp"

namespace tilesim {

struct gemmShape {
    std::int64_t m = 0;
    std::int64_t n = 0;
    std::int64_t k = 0;
    std::int64_t inBytes = 2;
    std::int64_t accBytes = 4;
};

struct gemmPlan {
    std::int64_t pm = 1;
    std::int64_t pn = 1;
    std::int64_t kb = 1;
};

struct planEval {
    gemmPlan plan;
    std::int64_t estCycles = 0;
    std::int64_t tilesUsed = 0;
    std::int64_t peakSRAM = 0;
};

std::int64_t blockSize(std::int64_t total, std::int64_t parts, std::int64_t index);
std::int64_t panelCount(const gemmShape& shape, const gemmPlan& plan);
void validateGemm(const Accelerator& acc, const gemmShape& shape, const gemmPlan& plan);

std::int64_t estCycles(const Accelerator& acc, const gemmShape& shape, const gemmPlan& plan);
std::int64_t peakSRAM(const gemmShape& shape, const gemmPlan& plan);
bool fits(const Accelerator& acc, const gemmShape& shape, const gemmPlan& plan);

Program buildGemm(const Accelerator& acc, const gemmShape& shape, const gemmPlan& plan);
std::vector<planEval> rankPlans(const Accelerator& acc, const gemmShape& shape, std::size_t limit);

planEval planGemm(const Accelerator& acc, const gemmShape& shape);

} // namespace tilesim