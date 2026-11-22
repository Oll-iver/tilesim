#include "gemm.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tilesim {

static std::int64_t ceilDiv(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }

static bool comparePlans(const planEval& a, const planEval& b) {
    if (a.estCycles != b.estCycles)
        return a.estCycles < b.estCycles;
    if (a.tilesUsed != b.tilesUsed)
        return a.tilesUsed < b.tilesUsed;
    if (a.peakSRAM != b.peakSRAM)
        return a.peakSRAM < b.peakSRAM;
    if (a.plan.pm != b.plan.pm)
        return a.plan.pm < b.plan.pm;
    return a.plan.kb < b.plan.kb;
}

static void addPanelExchange(const gemmShape& shape, const gemmPlan& plan, std::int64_t s,
                             Superstep& step) {
    std::int64_t ks = std::min(plan.kb, shape.k - s * plan.kb);
    std::int64_t ownerCol = s % plan.pn;
    std::int64_t ownerRow = s % plan.pm;

    if (plan.pn > 1) {
        for (std::int64_t i = 0; i < plan.pm; ++i) {
            std::int64_t rows = blockSize(shape.m, plan.pm, i);
            Message msg;
            msg.src = i * plan.pn + ownerCol;
            msg.bytes = rows * ks * shape.inBytes;
            for (std::int64_t j = 0; j < plan.pn; ++j) {
                if (j != ownerCol) {
                    msg.dsts.push_back(i * plan.pn + j);
                }
            }
            step.messages.push_back(msg);
        }
    }

    if (plan.pm > 1) {
        for (std::int64_t j = 0; j < plan.pn; ++j) {
            std::int64_t cols = blockSize(shape.n, plan.pn, j);
            Message msg;
            msg.src = ownerRow * plan.pn + j;
            msg.bytes = cols * ks * shape.inBytes;
            for (std::int64_t i = 0; i < plan.pm; ++i) {
                if (i != ownerRow) {
                    msg.dsts.push_back(i * plan.pn + j);
                }
            }
            step.messages.push_back(msg);
        }
    }
}

static void addPanelCompute(const gemmShape& shape, const gemmPlan& plan, std::int64_t s,
                            Superstep& step) {
    std::int64_t ks = std::min(plan.kb, shape.k - s * plan.kb);
    for (std::int64_t i = 0; i < plan.pm; ++i) {
        std::int64_t rows = blockSize(shape.m, plan.pm, i);
        for (std::int64_t j = 0; j < plan.pn; ++j) {
            std::int64_t cols = blockSize(shape.n, plan.pn, j);
            step.flops[i * plan.pn + j] = 2 * rows * cols * ks;
        }
    }
}

std::int64_t blockSize(std::int64_t total, std::int64_t parts, std::int64_t index) {
    std::int64_t base = total / parts;
    if (index < (total % parts)) {
        return base + 1;
    }
    return base;
}

std::int64_t panelCount(const gemmShape& shape, const gemmPlan& plan) {
    return ceilDiv(shape.k, plan.kb);
}

void validateGemm(const Accelerator& acc, const gemmShape& shape, const gemmPlan& plan) {
    if (shape.m <= 0 || shape.n <= 0 || shape.k < 0 || shape.inBytes <= 0 || shape.accBytes <= 0 ||
        plan.pm <= 0 || plan.pn <= 0 || plan.kb <= 0) {
        throw std::invalid_argument("m,n,k,pm,pn,kb and element sizes must be strictly positive");
    }
    if (plan.pm * plan.pn > acc.numTiles) {
        throw std::invalid_argument("required tiles exceeds tile count");
    }
    if (plan.pm > shape.m || plan.pn > shape.n || plan.kb > shape.k) {
        throw std::invalid_argument("plan dimensions exceed matrix dimensions");
    }
}

std::int64_t estCycles(const Accelerator& acc, const gemmShape& shape, const gemmPlan& plan) {
    std::int64_t mb = ceilDiv(shape.m, plan.pm);
    std::int64_t nb = ceilDiv(shape.n, plan.pn);
    std::int64_t s = panelCount(shape, plan);

    std::int64_t aCycles = 0;
    if (plan.pn > 1) {
        aCycles = messageCycles(acc, mb * plan.kb * shape.inBytes);
    }
    std::int64_t bCycles = 0;
    if (plan.pm > 1) {
        bCycles = messageCycles(acc, plan.kb * nb * shape.inBytes);
    }

    std::int64_t port = aCycles + bCycles;
    if (!acc.multicast) {
        std::int64_t txCost = (plan.pn - 1) * aCycles + (plan.pm - 1) * bCycles;
        if (txCost > port) {
            port = txCost;
        }
    }
    std::int64_t exch = 0;
    if (aCycles + bCycles > 0) {
        exch = acc.latency + port;
    }
    std::int64_t comp = computeCycles(acc, 2 * mb * nb * plan.kb);

    return (s + 1) * acc.latency + s * (comp + exch);
}

std::int64_t peakSRAM(const gemmShape& shape, const gemmPlan& plan) {
    std::int64_t mb = ceilDiv(shape.m, plan.pm);
    std::int64_t nb = ceilDiv(shape.n, plan.pn);
    std::int64_t numPanels = panelCount(shape, plan);
    std::int64_t accBytes = mb * nb * shape.accBytes;
    std::int64_t ownedA = mb * ceilDiv(numPanels, plan.pn) * plan.kb;
    std::int64_t ownedB = ceilDiv(numPanels, plan.pm) * plan.kb * nb;
    std::int64_t rbuffer = (mb + nb) * plan.kb;
    return accBytes + (ownedA + ownedB + rbuffer) * shape.inBytes;
}

bool fits(const Accelerator& acc, const gemmShape& shape, const gemmPlan& plan) {
    return peakSRAM(shape, plan) <= usableSRAM(acc);
}

Program buildGemm(const Accelerator& acc, const gemmShape& shape, const gemmPlan& plan) {
    validateGemm(acc, shape, plan);
    std::int64_t numPanels = panelCount(shape, plan);

    Program prog;
    prog.numTiles = acc.numTiles;

    for (std::int64_t s = 0; s <= numPanels; ++s) {
        Superstep step;
        step.flops = std::vector<std::int64_t>(acc.numTiles, 0);

        if (s > 0) {
            addPanelCompute(shape, plan, s - 1, step);
        }
        if (s < numPanels) {
            addPanelExchange(shape, plan, s, step);
        }
        prog.steps.push_back(step);
    }
    return prog;
}

std::vector<planEval> rankPlans(const Accelerator& acc, const gemmShape& shape, std::size_t limit) {
    validateGemm(acc, shape, gemmPlan{1, 1, 1});

    std::int64_t maxSRAM = usableSRAM(acc);

    // quickly generate a list of feasible kbs to consider,
    // but this is not exhaustive, and may miss some feasible plans
    std::vector<std::int64_t> kbs;
    if (shape.k < 4) {
        kbs.push_back(shape.k);
    } else {
        for (std::int64_t kb = 4; kb <= shape.k && kb <= 1024; kb *= 2) {
            kbs.push_back(kb);
        }
    }

    std::vector<planEval> feasible;
    std::int64_t maxPm = std::min(acc.numTiles, shape.m);

    for (std::int64_t pm = 1; pm <= maxPm; ++pm) {
        std::int64_t maxPn = std::min(acc.numTiles / pm, shape.n);
        for (std::int64_t pn = 1; pn <= maxPn; ++pn) {
            for (std::int64_t kb : kbs) {
                gemmPlan plan;
                plan.pm = pm;
                plan.pn = pn;
                plan.kb = kb;

                std::int64_t sram = peakSRAM(shape, plan);
                if (sram > maxSRAM)
                    continue;

                planEval eval;
                eval.plan = plan;
                eval.estCycles = estCycles(acc, shape, plan);
                eval.tilesUsed = pm * pn;
                eval.peakSRAM = sram;
                feasible.push_back(eval);
            }
        }
    }

    std::sort(feasible.begin(), feasible.end(), comparePlans);

    if (feasible.size() > limit) {
        feasible.resize(limit);
    }
    return feasible;
}

planEval planGemm(const Accelerator& acc, const gemmShape& shape) {
    std::vector<planEval> ranked = rankPlans(acc, shape, 1);
    if (ranked.empty()) {
        throw std::runtime_error("no feasible candidates for gemm plan");
    }
    return ranked[0];
}

} // namespace tilesim