#include "machine.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace tilesim {

Accelerator loadConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("failed to open config" + path);
    }

    json j = json::parse(f);

    Accelerator acc;

    if (j.contains("numTiles"))
        acc.numTiles = j["numTiles"];
    if (j.contains("tileSRAM"))
        acc.tileSRAM = j["tileSRAM"];
    if (j.contains("reserveSRAM"))
        acc.reserveSRAM = j["reserveSRAM"];
    if (j.contains("flops"))
        acc.flops = j["flops"];
    if (j.contains("exchangeBytes"))
        acc.exchangeBytes = j["exchangeBytes"];
    if (j.contains("messageOverhead"))
        acc.messageOverhead = j["messageOverhead"];
    if (j.contains("latency"))
        acc.latency = j["latency"];
    if (j.contains("computeOverhead"))
        acc.computeOverhead = j["computeOverhead"];
    if (j.contains("clock"))
        acc.clock = j["clock"];
    if (j.contains("multicast"))
        acc.multicast = j["multicast"];

    validate(acc);
    return acc;
}

void validate(const Accelerator& acc) {
    if (acc.numTiles <= 0 || acc.flops <= 0 || acc.exchangeBytes <= 0 || acc.clock <= 0) {
        throw std::invalid_argument("the number of tiles, FLOPs per cycle, clock speed, and "
                                    "exchange bytes per cycle must be greater than zero.");
    }
    if (acc.latency < 0 || acc.messageOverhead < 0 || acc.computeOverhead < 0)
        throw std::invalid_argument("Latency and overhead cannot be negtivem");
}

double peakFlops(const Accelerator& acc) {
    return double(acc.numTiles) * double(acc.flops) * acc.clock * 1e9;
}

double cycleToSecond(const Accelerator& acc, std::int64_t cycles) {
    return double(cycles) / (acc.clock * 1e9);
}

std::int64_t usableSRAM(const Accelerator& acc) { return (acc.tileSRAM - acc.reserveSRAM) * 1024; }

} // namespace tilesim
