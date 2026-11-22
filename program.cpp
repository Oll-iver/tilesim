#include "program.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace tilesim {

static std::int64_t ceilDiv(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }

void validateProgram(const Program& program, const Accelerator& acc) {
    validate(acc);

    if (program.numTiles <= 0) {
        throw std::invalid_argument("Tiles must be greater than 0");
    }
    if (program.numTiles != acc.numTiles) {
        throw std::invalid_argument("Program has " + std::to_string(program.numTiles) +
                                    "tiles, accelerator has " + std::to_string(acc.numTiles));
    }

    for (std::size_t s = 0; s < program.steps.size(); s++) {
        std::string prefix = "invalid program (superstep " + std::to_string(s) + "): ";
        const Superstep& step = program.steps[s];

        if (std::int64_t(step.flops.size()) != program.numTiles) {
            throw std::invalid_argument(prefix +
                                        "assumes a different number of tiles to the accelerator");
        }
        for (std::int64_t f : step.flops) {
            if (f < 0) {
                throw std::invalid_argument(prefix + "negative flop count");
            }
        }

        for (const Message& msg : step.messages) {
            if (msg.src < 0 || msg.src >= program.numTiles) {
                throw std::invalid_argument(prefix + "tile does not exist");
            }
            if (msg.dsts.empty()) {
                throw std::invalid_argument(prefix + "has no destination for message");
            }

            if (msg.bytes < 0) {
                throw std::invalid_argument(prefix + "has message(s) with negative size");
            }

            std::unordered_set<std::int64_t> seen;
            for (std::int64_t d : msg.dsts) {
                if (d < 0 || d >= program.numTiles) {
                    throw std::invalid_argument(prefix + "destination does not exist");
                }
                if (d == msg.src) {
                    throw std::invalid_argument(prefix + "tile sends to itself");
                }
                if (seen.count(d) > 0) {
                    throw std::invalid_argument(prefix + "sends to the same tile more than once");
                }
                seen.insert(d);
            }
        }
    }
}

std::int64_t totalFlops(const Program& program) {
    std::int64_t total = 0;
    for (const Superstep& step : program.steps) {
        for (std::int64_t f : step.flops) {
            total += f;
        }
    }
    return total;
}

std::int64_t totalReceieved(const Program& program) {
    std::int64_t total = 0;
    for (const Superstep& step : program.steps) {
        for (const Message& msg : step.messages) {
            total += msg.bytes * std::int64_t(msg.dsts.size());
        }
    }
    return total;
}

std::int64_t computeCycles(const Accelerator& acc, std::int64_t flops) {
    if (flops == 0)
        return 0;
    return ceilDiv(flops, acc.flops) + acc.computeOverhead;
}

std::int64_t messageCycles(const Accelerator& acc, std::int64_t bytes) {
    return ceilDiv(bytes, acc.exchangeBytes) + acc.messageOverhead;
}

std::int64_t portBound(const Accelerator& acc, std::int64_t numTiles,
                       const std::vector<Message>& messages) {
    std::vector<std::int64_t> tx(numTiles, 0);
    std::vector<std::int64_t> rx(numTiles, 0);
    for (const Message& msg : messages) {
        std::int64_t cycles = messageCycles(acc, msg.bytes);
        std::int64_t copies = acc.multicast ? 1 : std::int64_t(msg.dsts.size());

        tx[msg.src] += cycles * copies;
        for (std::int64_t d : msg.dsts) {
            rx[d] += cycles;
        }
    }

    std::int64_t bound = 0;
    for (std::int64_t t = 0; t < numTiles; t++) {
        std::int64_t tileCycles = std::max(tx[t], rx[t]);
        if (tileCycles > bound) {
            bound = tileCycles;
        }
    }
    return bound;
}

std::int64_t exchangeCycles(const Accelerator& acc, std::int64_t numTiles,
                            const std::vector<Message>& messages) {
    if (messages.empty())
        return 0;
    return acc.latency + portBound(acc, numTiles, messages);
}

Result analyse(const Accelerator& acc, const Program& program) {
    validateProgram(program, acc);
    Result result;
    for (const Superstep& step : program.steps) {
        phaseCycles phase;
        for (std::int64_t f : step.flops) {
            std::int64_t c = computeCycles(acc, f);
            if (c > phase.compute) {
                phase.compute = c;
            }
        }
        phase.sync = acc.latency;
        phase.exchange = exchangeCycles(acc, program.numTiles, step.messages);
        result.addStep(phase);
    }
    return result;
}

} // namespace tilesim