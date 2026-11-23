#include "sim.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

// TODO use traceWriter here to generate perfetto trace
//   tile's duration known -> record compute (step start -> tile finish; skip tiles with no work)
//   -> last tile arrives -> barrier completes -> record barrier (last arrival -> barrier done)
//   -> message scheduled -> record send on sender, receive on each receiver (start -> end, bytes)
//      multicast: 1 send, n receives; unicast: 1 send + 1 receive per copy
//   -> next superstep

namespace tilesim {

enum class eventType { tileDone, barrierDone, messageDone };

struct Event {
    std::int64_t time = 0;
    std::uint64_t seq = 0;
    eventType type = eventType::tileDone;
};

struct eventLater {
    bool operator()(const Event& a, const Event& b) const {
        if (a.time != b.time)
            return a.time > b.time;
        return a.seq > b.seq;
    }
};

static const double pi = 3.14159265358979323846;
static const double sqrt2 = 1.41421356237309504880;

static double maxNormal(std::int64_t n, double x) {
    double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * pi);
    double cdf = 0.5 * std::erfc(-x / sqrt2);
    return double(n) * pdf * std::exp(double(n - 1) * std::log(cdf));
}

Result simulate(const Accelerator& acc, const Program& program, const SimOptions& options) {
    validateProgram(program, acc);
    if (options.jitterSigma < 0) {
        throw std::invalid_argument("jitter sigma must be nonnegative");
    }

    std::mt19937_64 rng(options.seed);
    std::normal_distribution<double> standardNormal(0.0, 1.0);

    std::priority_queue<Event, std::vector<Event>, eventLater> pq;
    std::uint64_t nextSeq = 0;
    std::int64_t now = 0;

    Result result;

    for (std::size_t s = 0; s < program.steps.size(); s++) {
        const Superstep& step = program.steps[s];

        std::int64_t timeStepStart = now;
        std::int64_t timeAllArrived = now;
        std::int64_t timeBarrierDone = now;
        std::int64_t waiting = program.numTiles;

        for (std::int64_t t = 0; t < program.numTiles; t++) {
            std::int64_t flops = step.flops[t];
            std::int64_t dur = 0;

            if (flops > 0) {
                if (options.jitterSigma == 0.0) {
                    dur = computeCycles(acc, flops);
                } else {
                    double z = standardNormal(rng);
                    double mult = std::exp(options.jitterSigma * z -
                                           0.5 * options.jitterSigma * options.jitterSigma);
                    double base = double(flops) / double(acc.flops);
                    dur = std::int64_t(std::ceil(base * mult)) + acc.computeOverhead;
                }
            }
            pq.push({timeStepStart + dur, nextSeq++, eventType::tileDone});
        }

        while (!pq.empty()) {
            Event ev = pq.top();
            pq.pop();
            now = ev.time;

            if (ev.type == eventType::tileDone) {
                --waiting;
                if (waiting == 0) {
                    timeAllArrived = now;
                    pq.push({timeAllArrived + acc.latency, nextSeq++, eventType::barrierDone});
                }
            } else if (ev.type == eventType::barrierDone) {
                timeBarrierDone = now;

                if (!step.messages.empty()) {
                    std::int64_t exchStart = timeBarrierDone + acc.latency;
                    std::vector<std::int64_t> txBusy(program.numTiles, exchStart);
                    std::vector<std::int64_t> rxBusy(program.numTiles, exchStart);

                    for (const Message& m : step.messages) {
                        std::int64_t dur = messageCycles(acc, m.bytes);

                        if (acc.multicast) {
                            // Sender and all receivers must be free at the same time
                            std::int64_t start = txBusy[m.src];
                            for (std::int64_t dst : m.dsts) {
                                if (rxBusy[dst] > start)
                                    start = rxBusy[dst];
                            }
                            std::int64_t end = start + dur;
                            txBusy[m.src] = end;
                            for (std::int64_t dst : m.dsts)
                                rxBusy[dst] = end;

                            pq.push({end, nextSeq++, eventType::messageDone});
                        } else {
                            // Unicast: send one copy per destination
                            for (std::int64_t dst : m.dsts) {
                                std::int64_t start = std::max(txBusy[m.src], rxBusy[dst]);
                                std::int64_t end = start + dur;
                                txBusy[m.src] = end;
                                rxBusy[dst] = end;

                                pq.push({end, nextSeq++, eventType::messageDone});
                            }
                        }
                    }
                }
            }
        }
        phaseCycles phase;
        phase.compute = timeAllArrived - timeStepStart;
        phase.sync = timeBarrierDone - timeAllArrived;
        phase.exchange = now - timeBarrierDone;
        result.addStep(phase);
    }

    return result;
}

double expectedMaxNormal(std::int64_t n) {
    if (n < 1)
        throw std::invalid_argument("expected max needs n >= 1");

    int points = 20001;
    double lo = -10.0;
    double hi = 10.0;
    double step = (hi - lo) / (points - 1);

    double sum = 0.0;
    for (int i = 0; i < points; ++i) {
        double x = lo + i * step;
        double weight = (i == 0 || i == points - 1) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
        sum += weight * x * maxNormal(n, x);
    }
    return sum * step / 3.0;
}

double expectedMaxMulti(std::int64_t n, double sigma) {
    if (n < 1)
        throw std::invalid_argument("expected maximum needs n >= 1");

    int points = 20001;
    double lo = -10.0;
    double hi = 10.0;
    double step = (hi - lo) / (points - 1);

    double sum = 0.0;
    for (int i = 0; i < points; ++i) {
        double x = lo + i * step;
        double mult = std::exp(sigma * x - 0.5 * sigma * sigma);
        double weight = (i == 0 || i == points - 1) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
        sum += weight * mult * maxNormal(n, x);
    }
    return sum * step / 3.0;
}

double predictedOverhead(std::int64_t n, double sigma, std::int64_t workCycles,
                         std::int64_t overheadCycles) {
    double work = double(workCycles);
    double overhead = double(overheadCycles);
    return (work * expectedMaxMulti(n, sigma) + overhead) / (work + overhead) - 1.0;
}

} // namespace tilesim