#pragma once

#include <cstdint>
#include <vector>

#include "machine.hpp"

namespace tilesim {

struct Message {
    std::int64_t src = 0;
    std::vector<std::int64_t> dsts;
    std::int64_t bytes = 0;
};

struct Superstep {
    std::vector<std::int64_t> flops;
    std::vector<Message> messages;
};

struct Program {
    std::int64_t numTiles = 0;
    std::vector<Superstep> steps;
};

struct phaseCycles {
    std::int64_t compute = 0;
    std::int64_t sync = 0;
    std::int64_t exchange = 0;

    std::int64_t total() const { return compute + sync + exchange; }
};

struct Result {
    phaseCycles totals;
    std::vector<phaseCycles> steps;

    std::int64_t totalCycles() const { return totals.total(); }

    void addStep(const phaseCycles& step) {
        steps.push_back(step);
        totals.compute += step.compute;
        totals.sync += step.sync;
        totals.exchange += step.exchange;
    }
};

void validateProgram(const Program& program, const Accelerator& acc);

std::int64_t totalFlops(const Program& program);
std::int64_t totalReceived(const Program& program);
std::int64_t computeCycles(const Accelerator& acc, std::int64_t flops);
std::int64_t messageCycles(const Accelerator& acc, std::int64_t bytes);
std::int64_t portBound(const Accelerator& acc, std::int64_t numTiles,
                       const std::vector<Message>& messages);
std::int64_t exchangeCycles(const Accelerator& acc, std::int64_t numTiles,
                            const std::vector<Message>& messages);

Result analyse(const Accelerator& acc, const Program& program);

} // namespace tilesim