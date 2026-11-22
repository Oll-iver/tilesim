#include "out.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace tilesim {

static const std::int64_t lanes = 3;

Metrics metrics(const Accelerator& acc, std::int64_t flops, const phaseCycles& cycles) {
    Metrics m;
    std::int64_t total = cycles.total();
    m.seconds = cycleToSecond(acc, total);
    if (m.seconds > 0.0) {
        m.tflops = double(flops) / m.seconds / 1e12;
    }
    m.utilisation = (m.tflops * 1e12) / peakFlops(acc);

    if (total != 0) {
        m.computeShare = double(cycles.compute) / double(total);
        m.syncShare = double(cycles.sync) / double(total);
        m.exchangeShare = double(cycles.exchange) / double(total);
    }
    return m;
}

std::string describeMachine(const Accelerator& acc) {
    std::ostringstream out;
    out << "machine   " << acc.numTiles << " tiles at " << std::fixed << std::setprecision(3)
        << acc.clock << " GHz, model peak " << std::setprecision(1) << (peakFlops(acc) / 1e12)
        << " TFLOP/s, exchange " << acc.exchangeBytes << " B/cycle/tile, multicast "
        << (acc.multicast ? "on" : "off") << "\n";
    return out.str();
}

std::string formatTable(const Accelerator& acc, const std::vector<runRecord>& runs,
                        bool showWorkload) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);

    if (showWorkload) {
        out << std::left << std::setw(12) << "workload" << " ";
    }
    out << std::left << std::setw(10) << "model" << " " << std::right << std::setw(14) << "cycles"
        << " " << std::setw(12) << "time (us)" << " " << std::setw(9) << "TFLOP/s" << " "
        << std::setw(7) << "util %" << " " << std::setw(9) << "compute %" << " " << std::setw(7)
        << "sync %" << " " << std::setw(7) << "exch %" << "\n";

    for (const runRecord& run : runs) {
        Metrics m = metrics(acc, run.flops, run.cycles);
        if (showWorkload) {
            out << std::left << std::setw(12) << run.workload << " ";
        }
        out << std::left << std::setw(10) << run.model << " " << std::right << std::setw(14)
            << run.cycles.total() << " " << std::setw(12) << (m.seconds * 1e6) << " "
            << std::setw(9) << m.tflops << " " << std::setw(7) << (100.0 * m.utilisation) << " "
            << std::setw(9) << (100.0 * m.computeShare) << " " << std::setw(7)
            << (100.0 * m.syncShare) << " " << std::setw(7) << (100.0 * m.exchangeShare) << "\n";
    }
    return out.str();
}

std::string csvHeader() {
    return "model,workload,m,n,k,tiles,pm,pn,kb,steps,jitter,seed,cycles,seconds,tflops,"
           "utilisation,compute_cycles,sync_cycles,exchange_cycles\n";
}

std::string csvRow(const Accelerator& acc, const runRecord& run) {
    Metrics m = metrics(acc, run.flops, run.cycles);
    std::ostringstream out;

    out << run.model << "," << run.workload << ",";
    if (run.hasShape) {
        out << run.shape.m << "," << run.shape.n << "," << run.shape.k;
    } else {
        out << ",,";
    }
    out << "," << acc.numTiles << ",";
    if (run.hasPlan) {
        out << run.plan.pm << "," << run.plan.pn << "," << run.plan.kb;
    } else {
        out << ",,";
    }
    out << "," << run.steps << "," << run.jitter << "," << run.seed << "," << run.cycles.total()
        << "," << std::scientific << std::setprecision(9) << m.seconds << "," << std::fixed
        << std::setprecision(6) << m.tflops << "," << m.utilisation << "," << run.cycles.compute
        << "," << run.cycles.sync << "," << run.cycles.exchange << "\n";
    return out.str();
}

std::string formatTable(const Accelerator& acc, const gemmShape& shape,
                        const std::vector<planEval>& plans) {
    std::ostringstream out;
    out << std::right << std::setw(4) << "rank" << " " << std::setw(6) << "pm" << " "
        << std::setw(6) << "pn" << " " << std::setw(6) << "kb" << " " << std::setw(6) << "tiles"
        << " " << std::setw(7) << "panels" << " " << std::setw(14) << "estimate" << " "
        << std::setw(12) << "time (us)" << " " << std::setw(12) << "SRAM (KiB)" << "\n";

    for (std::size_t r = 0; r < plans.size(); ++r) {
        const planEval& e = plans[r];
        out << std::setw(4) << (r + 1) << " " << std::setw(6) << e.plan.pm << " " << std::setw(6)
            << e.plan.pn << " " << std::setw(6) << e.plan.kb << " " << std::setw(6) << e.tilesUsed
            << " " << std::setw(7) << panelCount(shape, e.plan) << " " << std::setw(14)
            << e.estCycles << " " << std::fixed << std::setprecision(2) << std::setw(12)
            << (cycleToSecond(acc, e.estCycles) * 1e6) << " " << std::setprecision(1)
            << std::setw(12) << (double(e.peakSRAM) / 1024.0) << "\n";
    }
    return out.str();
}

std::string stragglerHeader() {
    return "tiles,sigma,steps,seeds,cycles_per_step,measured_overhead,predicted_overhead,"
           "expected_max_normal\n";
}

std::string stragglerRow(const Accelerator& acc, const stragglerExperiment& experiment,
                         const stragglerOutcome& outcome) {
    std::ostringstream out;
    out << acc.numTiles << "," << experiment.sigma << "," << experiment.steps << ","
        << experiment.seeds << "," << experiment.cycles << "," << std::fixed << std::setprecision(8)
        << outcome.measuredOverhead << "," << outcome.predictedOverhead << ","
        << outcome.expectedMaxNormal << "\n";
    return out.str();
}

std::string formatStraggler(const Accelerator& acc, const stragglerExperiment& experiment,
                            const stragglerOutcome& outcome) {
    std::ostringstream out;
    out << "straggler  " << acc.numTiles << " tiles, sigma " << experiment.sigma << ", "
        << experiment.steps << " steps x " << experiment.seeds
        << " seeds, C = " << experiment.cycles << " cycles/step, c = " << acc.computeOverhead
        << "\n"
        << std::fixed << std::setprecision(4) << "E[max of n standard normals]   "
        << outcome.expectedMaxNormal << "\n"
        << std::setprecision(3) << "measured compute overhead      "
        << (100.0 * outcome.measuredOverhead) << " %\n"
        << "predicted compute overhead     " << (100.0 * outcome.predictedOverhead) << " %\n";
    return out.str();
}

static json makeMetadata(const std::string& kind, std::int64_t tid, const std::string& name) {
    return json{
        {"ph", "M"}, {"pid", 0}, {"tid", tid}, {"name", kind}, {"args", {{"name", name}}},
    };
}

traceWriter::traceWriter(const std::string& path, double clock, std::int64_t tracedTiles)
    : path(path), out(path), cyclesPer(clock * 1000.0), traced(tracedTiles) {
    if (!out.is_open()) {
        throw std::runtime_error("cannot open trace file '" + path + "'");
    }

    out << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[";
    out << makeMetadata("process_name", 0, "tilesim").dump();

    for (std::int64_t t = 0; t < traced; ++t) {
        std::string tile = "tile " + std::to_string(t);
        out << "," << makeMetadata("thread_name", lanes * t, tile + " compute").dump();
        out << "," << makeMetadata("thread_name", lanes * t + 1, tile + " send").dump();
        out << "," << makeMetadata("thread_name", lanes * t + 2, tile + " receive").dump();
    }
    out << "," << makeMetadata("thread_name", lanes * traced, "barrier").dump();
}

traceWriter::~traceWriter() {
    try {
        close();
    } catch (...) {
    }
}

void traceWriter::writeEvent(const std::string& name, std::int64_t tid, std::int64_t start,
                             std::int64_t end, const std::string& argKey, std::int64_t argVal) {
    if (!out.is_open())
        return;

    json ev = {
        {"ph", "X"},
        {"pid", 0},
        {"tid", tid},
        {"name", name},
        {"ts", double(start) / cyclesPer},
        {"dur", double(end - start) / cyclesPer},
        {"args", {{argKey, argVal}}},
    };
    out << "," << ev.dump();
}

void traceWriter::compute(std::int64_t tile, std::int64_t start, std::int64_t end,
                          std::int64_t superstep) {
    if (!traces(tile))
        return;
    writeEvent("compute", lanes * tile, start, end, "superstep", superstep);
}

void traceWriter::send(std::int64_t tile, std::int64_t start, std::int64_t end,
                       std::int64_t bytes) {
    if (!traces(tile))
        return;
    writeEvent("send", lanes * tile + 1, start, end, "bytes", bytes);
}

void traceWriter::receive(std::int64_t tile, std::int64_t start, std::int64_t end,
                          std::int64_t bytes) {
    if (!traces(tile))
        return;
    writeEvent("receive", lanes * tile + 2, start, end, "bytes", bytes);
}

void traceWriter::barrier(std::int64_t start, std::int64_t end, std::int64_t superstep) {
    writeEvent("sync", lanes * traced, start, end, "superstep", superstep);
}

void traceWriter::close() {
    if (!out.is_open())
        return;
    out << "]}\n";
    out.close();
    if (!out) {
        throw std::runtime_error("failed while writing trace file '" + path + "'");
    }
}

} // namespace tilesim