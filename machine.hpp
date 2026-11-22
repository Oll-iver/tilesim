#pragma once

#include <cstdint>
#include <string>

namespace tilesim {

struct Accelerator {
    std::int64_t numTiles = 1472;
    std::int64_t tileSRAM = 624;
    std::int64_t reserveSRAM = 64;
    std::int64_t flops = 128;
    std::int64_t exchangeBytes = 50;
    std::int64_t messageOverhead = 0;
    std::int64_t latency = 150;
    std::int64_t computeOverhead = 20;
    double clock = 1.33;
    bool multicast = true;
};

Accelerator loadConfig(const std::string& path);
void validate(const Accelerator& acc);

double peakFlops(const Accelerator& acc);
double cycleToSecond(const Accelerator& acc, std::int64_t cycles);
std::int64_t usableSRAM(const Accelerator& acc);

} // namespace tilesim
