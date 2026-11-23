#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm.hpp"
#include "machine.hpp"
#include "out.hpp"
#include "program.hpp"
#include "sim.hpp"



using namespace tilesim;

// TODO implement --trace
// parse both -> reject --trace for plan/layer/straggler  -> create writer (path, clock, N) 
// -> put in sim options -> runBoth -> close writer

struct layerShape {
    std::int64_t tokens = 0;
    std::int64_t dModel = 0;
    std::int64_t dFF = 0;
};

struct namedGemm {
    std::string name;
    gemmShape shape;
};

std::vector<namedGemm> transformerLayer(const layerShape& layer) {
    if (layer.tokens <= 0 || layer.dModel <= 0 || layer.dFF <= 0) {
        throw std::invalid_argument("layer dimensions cannot be nonpositive");
    }
    std::int64_t t = layer.tokens;
    std::int64_t d = layer.dModel;
    std::int64_t f = layer.dFF;

    return {
        {"qkv", {t, 3 * d, d}},
        {"attn_out", {t, d, d}},
        {"mlp_up", {t, f, d}},
        {"mlp_down", {t, d, f}},
    };
}

Program uniformCompute(std::int64_t tiles, std::int64_t steps, std::int64_t FLOPs) {
    Program program;
    program.numTiles = tiles;

    Superstep step;
    step.flops = std::vector<std::int64_t>(tiles, FLOPs);
    program.steps = std::vector<Superstep>(steps, step);
    return program;
}

stragglerOutcome runStraggler(const Accelerator& acc, const stragglerExperiment& exp,
                              std::uint64_t baseSeed) {
    if (exp.steps < 1 || exp.cycles < 1 || exp.seeds < 1) {
        throw std::invalid_argument("straggler: steps, cycles and seeds must be positive");
    }

    std::int64_t FLOPs = exp.cycles * acc.flops;
    Program program = uniformCompute(acc.numTiles, exp.steps, FLOPs);

    double ideal = double(exp.steps) * double(computeCycles(acc, FLOPs));
    double ratioSum = 0.0;

    for (std::int64_t r = 0; r < exp.seeds; ++r) {
        SimOptions sim;
        sim.jitterSigma = exp.sigma;
        sim.seed = baseSeed + std::uint64_t(r);

        Result res = simulate(acc, program, sim);
        ratioSum += double(res.totals.compute) / ideal;
    }

    stragglerOutcome outcome;
    outcome.measuredOverhead = (ratioSum / double(exp.seeds)) - 1.0;
    outcome.predictedOverhead =
        predictedOverhead(acc.numTiles, exp.sigma, exp.cycles, acc.computeOverhead);
    outcome.expectedMaxNormal = expectedMaxNormal(acc.numTiles);
    return outcome;
}

struct Options {
    Accelerator machine;
    double jitter = 0.0;
    bool jitterSet = false;
    std::uint64_t seed = 1;
    std::int64_t pm = 0, pn = 0, kb = 0;
    bool csv = false;
    std::int64_t limit = 10;
    std::int64_t steps = 50;
    std::int64_t cycles = 10000;
    std::int64_t seeds = 5;
    std::int64_t m = 0, n = 0, k = 0;
    std::int64_t tokens = 0, dModel = 0, dFF = 0;
};

void printUsage() {
    std::cout << "Usage:\n"
              << "  tilesim gemm M N K [options]\n"
              << "  tilesim plan M N K [options]\n"
              << "  tilesim layer TOKENS DMODEL DFF [options]\n"
              << "  tilesim straggler [options]\n\n"
              << "Common options:\n"
              << "  --config FILE       Load machine JSON config\n"
              << "  --tiles N           Override number of tiles\n"
              << "  --no-multicast      Disable hardware multicast\n"
              << "  --jitter SIGMA      Lognormal compute jitter sigma\n"
              << "  --seed S            Random seed (default: 1)\n"
              << "  --csv               Output CSV rows instead of table\n"
              << "  --pm P --pn Q --kb K  Force GEMM plan (gemm only)\n"
              << "  --limit N           Max plans to show (plan only, default: 10)\n"
              << "  --steps N           Supersteps (straggler only, default: 50)\n"
              << "  --cycles N          Work cycles/step (straggler only, default: 10000)\n"
              << "  --seeds N           Number of seeds (straggler only, default: 5)\n";
}

std::string nextArg(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument("missing value after " + flag);
    }
    return argv[++i];
}

void parseFlags(int startIdx, int argc, char** argv, Options& opt) {
    for (int i = startIdx; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--config") {
            opt.machine = loadConfig(nextArg(i, argc, argv, arg));
        } else if (arg == "--tiles") {
            opt.machine.numTiles = std::stoll(nextArg(i, argc, argv, arg));
        } else if (arg == "--no-multicast") {
            opt.machine.multicast = false;
        } else if (arg == "--jitter") {
            opt.jitter = std::stod(nextArg(i, argc, argv, arg));
            opt.jitterSet = true;
        } else if (arg == "--seed") {
            opt.seed = std::stoull(nextArg(i, argc, argv, arg));
        } else if (arg == "--csv") {
            opt.csv = true;
        } else if (arg == "--pm") {
            opt.pm = std::stoll(nextArg(i, argc, argv, arg));
        } else if (arg == "--pn") {
            opt.pn = std::stoll(nextArg(i, argc, argv, arg));
        } else if (arg == "--kb") {
            opt.kb = std::stoll(nextArg(i, argc, argv, arg));
        } else if (arg == "--limit") {
            opt.limit = std::stoll(nextArg(i, argc, argv, arg));
        } else if (arg == "--steps") {
            opt.steps = std::stoll(nextArg(i, argc, argv, arg));
        } else if (arg == "--cycles") {
            opt.cycles = std::stoll(nextArg(i, argc, argv, arg));
        } else if (arg == "--seeds") {
            opt.seeds = std::stoll(nextArg(i, argc, argv, arg));
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
}

std::vector<runRecord> runBoth(const Accelerator& acc, const std::string& workload,
                               const gemmShape& shape, const gemmPlan& plan,
                               const SimOptions& sim) {
    Program program = buildGemm(acc, shape, plan);

    runRecord base;
    base.workload = workload;
    base.hasShape = true;
    base.shape = shape;
    base.hasPlan = true;
    base.plan = plan;
    base.steps = std::int64_t(program.steps.size());
    base.jitter = sim.jitterSigma;
    base.seed = sim.seed;
    base.flops = totalFlops(program);

    runRecord analytical = base;
    analytical.model = "analytical";
    analytical.cycles = analyse(acc, program).totals;

    runRecord des = base;
    des.model = "des";
    des.cycles = simulate(acc, program, sim).totals;

    return {analytical, des};
}

int cmdGemm(const Options& opt) {
    const Accelerator& acc = opt.machine;
    gemmShape shape;
    shape.m = opt.m;
    shape.n = opt.n;
    shape.k = opt.k;

    gemmPlan plan;
    bool forced = (opt.pm > 0 || opt.pn > 0 || opt.kb > 0);
    if (forced) {
        if (opt.pm <= 0 || opt.pn <= 0 || opt.kb <= 0) {
            throw std::invalid_argument("--pm, --pn, and --kb must all be provided together");
        }
        plan.pm = opt.pm;
        plan.pn = opt.pn;
        plan.kb = opt.kb;
        validateGemm(acc, shape, plan);
        if (!fits(acc, shape, plan)) {
            throw std::invalid_argument("forced plan doesn't fit in SRAM (" +
                                        std::to_string(peakSRAM(shape, plan) / 1024) + " KiB > " +
                                        std::to_string(usableSRAM(acc) / 1024) + " KiB)");
        }
    } else {
        planEval best = planGemm(acc, shape);
        plan = best.plan;
    }

    SimOptions sim;
    sim.jitterSigma = opt.jitter;
    sim.seed = opt.seed;

    std::vector<runRecord> runs = runBoth(acc, "gemm", shape, plan, sim);

    if (opt.csv) {
        std::cout << csvHeader();
        for (const runRecord& run : runs) {
            std::cout << csvRow(acc, run);
        }
        return 0;
    }

    std::int64_t panels = panelCount(shape, plan);
    std::cout << describeMachine(acc);
    std::cout << "gemm      " << shape.m << " x " << shape.n << " x " << shape.k << " ("
              << shape.inBytes << "-byte inputs, " << shape.accBytes << "-byte accumulators)\n";
    std::cout << "plan      pm x pn = " << plan.pm << " x " << plan.pn << " ("
              << (plan.pm * plan.pn) << " tiles used), kb = " << plan.kb << ", S = " << panels
              << " panel(s), " << (panels + 1) << " supersteps"
              << (forced ? " [forced]" : " [planner]") << "\n";
    std::cout << "sram      peak " << (peakSRAM(shape, plan) / 1024) << " KiB per tile of "
              << (usableSRAM(acc) / 1024) << " KiB usable\n";
    std::cout << "estimate  " << estCycles(acc, shape, plan) << " cycles (closed form)\n\n";
    std::cout << formatTable(acc, runs, false);
    return 0;
}

int cmdPlan(const Options& opt) {
    gemmShape shape;
    shape.m = opt.m;
    shape.n = opt.n;
    shape.k = opt.k;

    if (opt.limit <= 0) {
        throw std::invalid_argument("--limit must be > 0");
    }
    std::vector<planEval> plans = rankPlans(opt.machine, shape, std::size_t(opt.limit));
    if (plans.empty()) {
        throw std::invalid_argument("no mapping of this GEMM fits in on-chip SRAM");
    }

    std::cout << describeMachine(opt.machine);
    std::cout << "usable SRAM " << (usableSRAM(opt.machine) / 1024) << " KiB per tile\n\n";
    std::cout << formatTable(opt.machine, shape, plans);
    return 0;
}

int cmdLayer(const Options& opt) {
    const Accelerator& acc = opt.machine;
    layerShape layer;
    layer.tokens = opt.tokens;
    layer.dModel = opt.dModel;
    layer.dFF = opt.dFF;

    SimOptions sim;
    sim.jitterSigma = opt.jitter;
    sim.seed = opt.seed;

    std::vector<runRecord> runs;
    runRecord totalAnalytical;
    totalAnalytical.model = "analytical";
    totalAnalytical.workload = "layer_total";
    totalAnalytical.jitter = sim.jitterSigma;
    totalAnalytical.seed = sim.seed;

    runRecord totalDes = totalAnalytical;
    totalDes.model = "des";

    std::string plansText;
    for (const namedGemm& gemm : transformerLayer(layer)) {
        planEval best = planGemm(acc, gemm.shape);
        gemmPlan plan = best.plan;

        plansText += gemm.name + ": " + std::to_string(gemm.shape.m) + " x " +
                     std::to_string(gemm.shape.n) + " x " + std::to_string(gemm.shape.k) +
                     ", plan " + std::to_string(plan.pm) + " x " + std::to_string(plan.pn) +
                     ", kb = " + std::to_string(plan.kb) + "\n";

        for (const runRecord& run : runBoth(acc, gemm.name, gemm.shape, plan, sim)) {
            if (run.model == "des") {
                totalDes.steps += run.steps;
                totalDes.flops += run.flops;
                totalDes.cycles.compute += run.cycles.compute;
                totalDes.cycles.sync += run.cycles.sync;
                totalDes.cycles.exchange += run.cycles.exchange;
            } else {
                totalAnalytical.steps += run.steps;
                totalAnalytical.flops += run.flops;
                totalAnalytical.cycles.compute += run.cycles.compute;
                totalAnalytical.cycles.sync += run.cycles.sync;
                totalAnalytical.cycles.exchange += run.cycles.exchange;
            }
            runs.push_back(run);
        }
    }

    runs.push_back(totalAnalytical);
    runs.push_back(totalDes);

    if (opt.csv) {
        std::cout << csvHeader();
        for (const runRecord& run : runs) {
            std::cout << csvRow(acc, run);
        }
        return 0;
    }

    std::cout << describeMachine(acc);
    std::cout << "layer     tokens = " << layer.tokens << ", d_model = " << layer.dModel
              << ", d_ff = " << layer.dFF << " (attention score and value products excluded)\n";
    std::cout << plansText << "\n" << formatTable(acc, runs, true);
    return 0;
}

int cmdStraggler(const Options& opt) {
    stragglerExperiment exp;
    exp.steps = opt.steps;
    exp.cycles = opt.cycles;
    exp.seeds = opt.seeds;
    exp.sigma = opt.jitterSet ? opt.jitter : 0.05;

    stragglerOutcome outcome = runStraggler(opt.machine, exp, opt.seed);

    if (opt.csv) {
        std::cout << stragglerHeader() << stragglerRow(opt.machine, exp, outcome);
    } else {
        std::cout << formatStraggler(opt.machine, exp, outcome);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help") {
        printUsage();
        return 0;
    }

    try {
        Options opt;

        if (cmd == "gemm" || cmd == "plan") {
            if (argc < 5) {
                throw std::invalid_argument(cmd + " requires 3 arguments: M N K");
            }
            opt.m = std::stoll(argv[2]);
            opt.n = std::stoll(argv[3]);
            opt.k = std::stoll(argv[4]);
            parseFlags(5, argc, argv, opt);
            validate(opt.machine);
            return (cmd == "gemm") ? cmdGemm(opt) : cmdPlan(opt);

        } else if (cmd == "layer") {
            if (argc < 5) {
                throw std::invalid_argument("layer requires 3 arguments: TOKENS DMODEL DFF");
            }
            opt.tokens = std::stoll(argv[2]);
            opt.dModel = std::stoll(argv[3]);
            opt.dFF = std::stoll(argv[4]);
            parseFlags(5, argc, argv, opt);
            validate(opt.machine);
            return cmdLayer(opt);

        } else if (cmd == "straggler") {
            parseFlags(2, argc, argv, opt);
            validate(opt.machine);
            return cmdStraggler(opt);

        } else {
            throw std::invalid_argument("unknown subcommand: " + cmd);
        }
    } catch (const std::exception& e) {
        std::cerr << "tilesim: error: " << e.what() << "\n";
        return 1;
    }
}