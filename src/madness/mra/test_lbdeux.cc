/// Unit tests for the LBDeuxPmap process map.
///
/// Each map is written by hand, so that every case reaches one lookup branch or one constructor
/// check. The ranks are only labels here, because the map does not know the size of the World.

#include <madness/mra/mra.h>
#include <madness/mra/lbdeux.h>
#include <madness/world/test_utilities.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace madness;

template <std::size_t NDIM>
using Assignment = std::vector<std::pair<Key<NDIM>, ProcessID>>;

static Key<1> key1(Level n, Translation l) {
    return {n, vec(l)};
}

static Key<3> key3(Level n, Translation x, Translation y, Translation z) {
    return {n, vec(x, y, z)};
}

/// True only for MadnessException. An old-style `throw "..."` or a clean return is a failure.
template <std::size_t NDIM>
static bool constructor_throws(const Assignment<NDIM>& assignment) {
    try {
        LBDeuxPmap<NDIM> pmap(assignment);
    } catch (const MadnessException&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

/// Mapped tree, with a distinct rank for each entry:
///
///     (0,0) -> 0
///       (1,1) -> 1
///         (2,2) -> 2
static int test_nearest_ancestor(World& world) {
    test_output t("owner() returns the nearest mapped ancestor", world.rank() == 0);
    const LBDeuxPmap<1> pmap(Assignment<1>{{Key<1>(0), 0}, {key1(1, 1), 1}, {key1(2, 2), 2}});

    t.checkpoint(pmap.owner(Key<1>(0)), 0, "exact key at the root");
    t.checkpoint(pmap.owner(key1(1, 1)), 1, "exact key at an interior level");
    t.checkpoint(pmap.owner(key1(2, 2)), 2, "exact key at the deepest level");
    t.checkpoint(pmap.owner(key1(3, 4)), 2, "child of the deepest entry");
    t.checkpoint(pmap.owner(key1(4, 9)), 2, "grandchild of the deepest entry");
    t.checkpoint(pmap.owner(key1(2, 3)), 1, "mapped sibling is not an ancestor");
    t.checkpoint(pmap.owner(key1(3, 1)), 0, "grandchild of an unmapped key goes to the root");
    return t.end();
}

/// Each level-20 key below has the translation of its level-3 ancestor in the high 3 bits. The
/// two level-3 entries are siblings, so a lookup that stops one level too high finds neither.
static int test_key_below_deepest_level(World& world) {
    test_output t("owner() resolves a key below the deepest entry", world.rank() == 0);
    const LBDeuxPmap<3> pmap(Assignment<3>{{Key<3>(0), 0}, {key3(3, 5, 2, 7), 1}, {key3(3, 5, 2, 6), 2}});

    t.checkpoint(pmap.owner(key3(20, 667705, 262145, 1048575)), 1, "level 20 inside (3,[5,2,7])");
    t.checkpoint(pmap.owner(key3(20, 667705, 262145, 917503)), 2, "level 20 inside (3,[5,2,6])");
    t.checkpoint(pmap.owner(key3(20, 0, 0, 0)), 0, "level 20 outside both subtrees");
    return t.end();
}

/// A level-0 key with a nonzero translation is not in the box, and only such a key can reach this
/// branch, because the constructor requires the root.
static int test_unmatched_level_zero_key_throws(World& world) {
    test_output t("owner() throws if no ancestor matches", world.rank() == 0);
    const LBDeuxPmap<1> pmap(Assignment<1>{{Key<1>(0), 0}});
    bool threw = false;
    try {
        pmap.owner(key1(0, 1));
    } catch (const MadnessException&) {
        threw = true;
    } catch (...) {
    }
    t.checkpoint(threw, "level-0 key with a nonzero translation");
    return t.end();
}

static int test_missing_root_throws(World& world) {
    test_output t("constructor rejects a map without the root", world.rank() == 0);
    t.checkpoint(constructor_throws<1>({{key1(1, 0), 0}, {key1(1, 1), 1}}), "only level-1 keys");
    t.checkpoint(constructor_throws<1>({}), "empty assignment");
    return t.end();
}

static int test_duplicate_key_throws(World& world) {
    test_output t("constructor rejects a duplicate key", world.rank() == 0);
    t.checkpoint(constructor_throws<1>({{Key<1>(0), 0}, {key1(1, 1), 1}, {key1(1, 1), 1}}), "same rank");
    t.checkpoint(constructor_throws<1>({{Key<1>(0), 0}, {key1(1, 1), 1}, {key1(1, 1), 2}}), "different rank");
    return t.end();
}

static int test_negative_rank_throws(World& world) {
    test_output t("constructor rejects a negative rank", world.rank() == 0);
    t.checkpoint(constructor_throws<1>({{Key<1>(0), 0}, {key1(1, 1), -1}}), "negative rank");
    return t.end();
}

static double gaussian(const coord_3d& r) {
    return exp(-(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
}

/// A negative cost at the root stands in for roundoff in its residual. The real residual can only
/// go below zero after all children split off, which needs a large fac in load_balance().
struct NegativeRootCost {
    double operator()(const Key<3>& key, const FunctionNode<double, 3>&) const {
        return (key.level() == 0) ? -1.e-3 : 1.0;
    }
};

struct ZeroCost {
    double operator()(const Key<3>&, const FunctionNode<double, 3>&) const {
        return 0.0;
    }
};

/// Runs the real driver, then asks the returned map for the root and for every local function key.
template <typename costT>
static int test_driver_keeps_root(World& world, const std::string& name, const costT& cost, const double fac) {
    test_output t(name, world.rank() == 0);
    const real_function_3d f = real_factory_3d(world).f(gaussian);

    LoadBalanceDeux<3> lb(world);
    lb.add_tree(f, cost, true);

    std::shared_ptr<WorldDCPmapInterface<Key<3>>> pmap;
    try {
        pmap = lb.load_balance(fac);
    } catch (const MadnessException&) {
        // The constructor sees the same broadcast map on every rank, so every rank comes here.
        t.checkpoint(false, "load_balance() returns a map");
        return t.end();
    }
    t.checkpoint(true, "load_balance() returns a map");

    const auto valid_rank = [&world](const ProcessID p) { return p >= 0 && p < world.size(); };
    t.checkpoint(valid_rank(pmap->owner(Key<3>(0))), "the map owns the root");

    bool all_keys_owned = true;
    const auto& coeffs = f.get_impl()->get_coeffs();
    for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
        all_keys_owned = all_keys_owned && valid_rank(pmap->owner(it->first));
    }
    t.checkpoint(all_keys_owned, "the map owns every local function key");
    return t.end();
}

// ---------------------------------------------------------------------------------------------
// Evaluation only: the LPT and SFC candidates, the temporary selector, and the LBEXP output.
// ---------------------------------------------------------------------------------------------

template <std::size_t NDIM>
using Subtrees = std::vector<std::pair<Key<NDIM>, double>>;

template <std::size_t NDIM>
using Assigner = Assignment<NDIM> (*)(Subtrees<NDIM>, ProcessID);

/// @return The rank of key, or -1 if the assignment does not have exactly one entry for key
template <std::size_t NDIM>
static ProcessID assigned_rank(const Assignment<NDIM>& assignment, const Key<NDIM>& key) {
    ProcessID rank = -1;
    int matches = 0;
    for (const auto& entry : assignment) {
        if (entry.first == key) {
            rank = entry.second;
            ++matches;
        }
    }
    return (matches == 1) ? rank : -1;
}

template <std::size_t NDIM>
static bool assigned_as(const Assignment<NDIM>& got, const Assignment<NDIM>& want) {
    if (got.size() != want.size()) return false;
    for (const auto& entry : want) {
        if (assigned_rank(got, entry.first) != entry.second) return false;
    }
    return true;
}

/// Every input key appears exactly once, and every rank is in [0, nproc)
template <std::size_t NDIM>
static bool covers_each_key_once(const Subtrees<NDIM>& subtrees, const Assignment<NDIM>& assignment,
                                 const ProcessID nproc) {
    if (assignment.size() != subtrees.size()) return false;
    for (const auto& subtree : subtrees) {
        const ProcessID rank = assigned_rank(assignment, subtree.first);
        if (rank < 0 || rank >= nproc) return false;
    }
    return true;
}

template <std::size_t NDIM>
static bool assign_throws(const Assigner<NDIM> assign, const ProcessID nproc) {
    try {
        assign(Subtrees<NDIM>{{Key<NDIM>(0), 1.0}}, nproc);
    } catch (const MadnessException&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static Subtrees<1> mixed_subtrees() {
    return {{Key<1>(0), 0.5}, {key1(1, 0), 2.0}, {key1(2, 2), 3.0}, {key1(2, 3), 1.0},
            {key1(3, 2), 4.0}, {key1(3, 3), 1.5}, {key1(1, 1), 9.0}};
}

static int test_assignment_invariants(World& world, const std::string& name, const Assigner<1> assign) {
    test_output t(name + " keeps the assignment invariants", world.rank() == 0);
    const Subtrees<1> mixed = mixed_subtrees();
    Subtrees<1> zero = mixed;
    for (auto& subtree : zero) subtree.second = 0.0;

    t.checkpoint(covers_each_key_once(mixed, assign(mixed, 3), 3), "3 ranks");
    t.checkpoint(covers_each_key_once(mixed, assign(mixed, 1), 1), "1 rank");
    t.checkpoint(covers_each_key_once(mixed, assign(mixed, 20), 20), "more ranks than subtrees");
    t.checkpoint(covers_each_key_once(zero, assign(zero, 3), 3), "all costs zero");
    t.checkpoint(assign(mixed, 3) == assign(mixed, 3), "repeated calls are identical");
    t.checkpoint(assign(zero, 3) == assign(zero, 3), "repeated calls with zero costs are identical");
    t.checkpoint(assign_throws<1>(assign, 0), "rejects 0 ranks");
    t.checkpoint(assign_throws<1>(assign, -1), "rejects a negative number of ranks");
    return t.end();
}

/// Extraction must not change LPT: the costliest subtree goes first, and each following subtree
/// goes to the rank with the lowest cost.
static int test_lpt_order(World& world) {
    test_output t("assign_lpt() packs in decreasing cost order", world.rank() == 0);
    // Costs after each step: 5,_ -> 5,4 -> 5,7 -> 7,7
    const auto got = detail::assign_lpt<1>({{key1(2, 0), 2.0}, {key1(2, 1), 5.0}, {key1(2, 2), 3.0}, {key1(2, 3), 4.0}}, 2);
    t.checkpoint(assigned_as<1>(got, {{key1(2, 1), 0}, {key1(2, 3), 1}, {key1(2, 2), 1}, {key1(2, 0), 0}}), "two ranks");
    return t.end();
}

static int test_sfc_contiguous_runs(World& world) {
    test_output t("assign_sfc() cuts contiguous Morton runs", world.rank() == 0);
    // Morton order is root, 0, 1, 2, 3. The prefix sums are 0, 0, 1, 2, 3, 4 and the target is 2.
    const auto got = detail::assign_sfc<1>({{key1(2, 3), 1.0}, {key1(2, 0), 1.0}, {Key<1>(0), 0.0}, {key1(2, 2), 1.0}, {key1(2, 1), 1.0}}, 2);
    t.checkpoint(assigned_as<1>(got, {{Key<1>(0), 0}, {key1(2, 0), 0}, {key1(2, 1), 0}, {key1(2, 2), 1}, {key1(2, 3), 1}}), "shuffled input");
    return t.end();
}

/// With unit costs and one rank per key, the rank of each key is its position in Morton order.
static int test_sfc_morton_order(World& world) {
    test_output t("assign_sfc() uses adaptive Morton order", world.rank() == 0);
    {
        // Level order or translation order puts (1,1) before a descendant of (1,0).
        const auto got = detail::assign_sfc<1>({{key1(1, 1), 1.0}, {key1(2, 1), 1.0}, {key1(1, 0), 1.0}, {Key<1>(0), 1.0}, {key1(2, 0), 1.0}}, 5);
        t.checkpoint(assigned_as<1>(got, {{Key<1>(0), 0}, {key1(1, 0), 1}, {key1(2, 0), 2}, {key1(2, 1), 3}, {key1(1, 1), 4}}), "an ancestor precedes its descendants");
    }
    {
        // Dimension i is bit i of the child index, so z is the most significant dimension.
        const auto got = detail::assign_sfc<3>({{key3(1, 0, 0, 1), 1.0}, {key3(1, 1, 0, 0), 1.0}, {key3(1, 0, 1, 0), 1.0}, {Key<3>(0), 1.0}}, 4);
        t.checkpoint(assigned_as<3>(got, {{Key<3>(0), 0}, {key3(1, 1, 0, 0), 1}, {key3(1, 0, 1, 0), 2}, {key3(1, 0, 0, 1), 3}}), "dimension i is bit i");
    }
    {
        // Child sequences: [1,0,0]->[0,1] and [0,2,0]->[2,0]. Comparing translations as tuples gives the other order.
        const auto got = detail::assign_sfc<3>({{key3(2, 0, 2, 0), 1.0}, {key3(2, 1, 0, 0), 1.0}}, 2);
        t.checkpoint(assigned_as<3>(got, {{key3(2, 1, 0, 0), 0}, {key3(2, 0, 2, 0), 1}}), "the coarsest bit decides first");
    }
    return t.end();
}

static int test_sfc_cuts(World& world) {
    test_output t("assign_sfc() chooses the nearest cut", world.rank() == 0);
    {
        // The prefix sums are 0, 1, 3, 4 and the target is 2, so indices 1 and 2 tie.
        const auto got = detail::assign_sfc<1>({{key1(2, 0), 1.0}, {key1(2, 1), 2.0}, {key1(2, 2), 1.0}}, 2);
        t.checkpoint(assigned_as<1>(got, {{key1(2, 0), 0}, {key1(2, 1), 1}, {key1(2, 2), 1}}), "the lower index wins a tie");
    }
    {
        // The clamped prefix sums are 0, 0, 1, 2 and the target is 1.
        const auto got = detail::assign_sfc<1>({{Key<1>(0), -5.0}, {key1(1, 0), 1.0}, {key1(1, 1), 1.0}}, 2);
        t.checkpoint(assigned_as<1>(got, {{Key<1>(0), 0}, {key1(1, 0), 0}, {key1(1, 1), 1}}), "a negative cost counts as zero");
    }
    {
        const auto got = detail::assign_sfc<1>({{key1(2, 0), 0.0}, {key1(2, 1), 0.0}, {key1(2, 2), 0.0}, {key1(2, 3), 0.0}}, 2);
        t.checkpoint(assigned_as<1>(got, {{key1(2, 0), 0}, {key1(2, 1), 0}, {key1(2, 2), 1}, {key1(2, 3), 1}}), "zero total cost cuts by subtree count");
    }
    return t.end();
}

/// The bound uses the clamped costs and a relative tolerance, so that roundoff does not fail it.
static bool within_load_bound(const Subtrees<1>& subtrees, const ProcessID nproc) {
    const Assignment<1> assignment = detail::assign_sfc<1>(subtrees, nproc);
    if (!covers_each_key_once(subtrees, assignment, nproc)) return false;
    std::vector<double> rank_cost(nproc, 0.0);
    double total = 0.0;
    double max_subtree = 0.0;
    for (const auto& subtree : subtrees) {
        const double cost = std::max(0.0, subtree.second);
        rank_cost[assigned_rank(assignment, subtree.first)] += cost;
        total += cost;
        max_subtree = std::max(max_subtree, cost);
    }
    const double max_rank = *std::max_element(rank_cost.begin(), rank_cost.end());
    return max_rank <= (total / nproc + max_subtree) * (1.0 + 1.e-12);
}

static int test_sfc_load_bound(World& world) {
    test_output t("assign_sfc() keeps the load bound", world.rank() == 0);
    const Subtrees<1> mixed = mixed_subtrees();
    const Subtrees<1> heavy = {{Key<1>(0), 0.0}, {key1(2, 0), 100.0}, {key1(2, 1), 1.0}, {key1(2, 2), 1.0}, {key1(2, 3), 1.0}};
    const Subtrees<1> negative = {{Key<1>(0), -1.0}, {key1(1, 0), 3.0}, {key1(1, 1), 1.0}};
    for (const ProcessID nproc : {1, 2, 3, 5, 20}) {
        const std::string suffix = " on " + std::to_string(nproc) + " ranks";
        t.checkpoint(within_load_bound(mixed, nproc), "mixed costs" + suffix);
        t.checkpoint(within_load_bound(heavy, nproc), "one heavy subtree" + suffix);
        t.checkpoint(within_load_bound(negative, nproc), "negative root cost" + suffix);
    }
    return t.end();
}

static int test_assignment_metrics(World& world) {
    test_output t("the assignment metrics report the load and the bound", world.rank() == 0);
    {
        const Subtrees<1> subtrees = {{Key<1>(0), -1.0}, {key1(2, 0), 2.0}, {key1(2, 1), 3.0}, {key1(2, 2), 5.0}};
        const auto m = detail::lb_assignment_metrics<1>(subtrees, {{Key<1>(0), 0}, {key1(2, 0), 0}, {key1(2, 1), 1}, {key1(2, 2), 2}}, 3);
        t.checkpoint(m.total_cost, 10.0, 1.e-14, "total cost ignores the negative cost");
        t.checkpoint(m.max_subtree_cost, 5.0, 1.e-14, "maximum subtree cost");
        t.checkpoint(m.max_rank_cost, 5.0, 1.e-14, "maximum rank cost");
        t.checkpoint(m.bound, 10.0 / 3.0 + 5.0, 1.e-14, "load bound");
        t.checkpoint(m.bound_ok, "a balanced assignment is within the bound");
        t.checkpoint(m.rank_costs == std::vector<double>{2.0, 3.0, 5.0}, "cost per rank");
    }
    {
        // The maximum rank cost is 4, and the bound is 4/4 + 1 = 2.
        const Subtrees<1> subtrees = {{key1(2, 0), 1.0}, {key1(2, 1), 1.0}, {key1(2, 2), 1.0}, {key1(2, 3), 1.0}};
        const auto m = detail::lb_assignment_metrics<1>(subtrees, {{key1(2, 0), 0}, {key1(2, 1), 0}, {key1(2, 2), 0}, {key1(2, 3), 0}}, 4);
        t.checkpoint(!m.bound_ok, "one overloaded rank is outside the bound");
    }
    return t.end();
}

static constexpr const char* selector_name = "MAD_LB_EXPERIMENT_ALGORITHM";

static void set_selector(const char* value) {
    if (value) setenv(selector_name, value, 1);
    else unsetenv(selector_name);
}

/// Sets a different value on rank 0 and on all other ranks, so that a rank that reads its own
/// environment gets a different result. The two-rank form is what makes this case effective.
static int test_selector_follows_rank_zero(World& world) {
    test_output t("the selector returns the value of rank 0 on every rank", world.rank() == 0);
    const auto check = [&](const char* rank0, const char* others, const std::optional<LBExperimentAlgorithm> want, const std::string& label) {
        set_selector(world.rank() == 0 ? rank0 : others);
        t.checkpoint(detail::select_lb_experiment_algorithm(world) == want, label);
    };
    check("lpt", "sfc", LBExperimentAlgorithm::LPT, "lpt on rank 0");
    check("sfc", "lpt", LBExperimentAlgorithm::SFC, "sfc on rank 0");
    check(nullptr, "sfc", std::nullopt, "unset on rank 0");
    check(nullptr, "invalid", std::nullopt, "an invalid value on another rank is not read");
    set_selector(nullptr);
    return t.end();
}

static int test_invalid_selector_throws_on_every_rank(World& world) {
    test_output t("every rank throws for an invalid selector on rank 0", world.rank() == 0);
    set_selector(world.rank() == 0 ? "invalid" : nullptr);
    bool threw = false;
    try {
        detail::select_lb_experiment_algorithm(world);
    } catch (const MadnessException& e) {
        threw = std::strcmp(e.msg, "invalid MAD_LB_EXPERIMENT_ALGORITHM") == 0;
    } catch (...) {
    }
    t.checkpoint(threw, "MadnessException with the fixed message");
    set_selector(nullptr);

    // All ranks left the broadcast, so a later collective must complete.
    world.gop.fence();
    t.checkpoint(detail::select_lb_experiment_algorithm(world) == std::nullopt, "a later collective completes");
    return t.end();
}

struct UnitCost {
    double operator()(const Key<3>&, const FunctionNode<double, 3>&) const {
        return 1.0;
    }
};

/// Runs the real driver and returns what it wrote to std::cout.
static std::string load_balance_output(World& world, const real_function_3d& f) {
    LoadBalanceDeux<3> lb(world);
    lb.add_tree(f, UnitCost(), true);
    std::ostringstream captured;
    std::streambuf* const old_buffer = std::cout.rdbuf(captured.rdbuf());
    lb.load_balance();
    std::cout.rdbuf(old_buffer);
    return captured.str();
}

static long call_index(const std::string& line) {
    const auto pos = line.find("call=");
    return (pos == std::string::npos) ? -1 : std::stol(line.substr(pos + 5));
}

static int test_load_balance_reports_the_selection(World& world) {
    test_output t("load_balance() writes LBEXP only if the selector is set", world.rank() == 0);
    const real_function_3d f = real_factory_3d(world).f(gaussian);
    const bool is_rank_zero = world.rank() == 0;
    const auto has = [](const std::string& text, const std::string& what) { return text.find(what) != std::string::npos; };

    set_selector(nullptr);
    t.checkpoint(!has(load_balance_output(world, f), "LBEXP"), "no output if the selector is unset");

    set_selector(is_rank_zero ? "sfc" : nullptr);
    const std::string sfc = load_balance_output(world, f);
    t.checkpoint(is_rank_zero ? has(sfc, "LBEXP_ASSIGN") && has(sfc, "algorithm=sfc") && has(sfc, "bound_ok=true") : sfc.empty(), "sfc");

    set_selector(is_rank_zero ? "lpt" : nullptr);
    const std::string lpt = load_balance_output(world, f);
    t.checkpoint(is_rank_zero ? has(lpt, "LBEXP_ASSIGN") && has(lpt, "algorithm=lpt") : lpt.empty(), "lpt");
    if (is_rank_zero) t.checkpoint(call_index(lpt) == call_index(sfc) + 1, "the call index increases");

    set_selector(nullptr);
    return t.end();
}

int main(int argc, char** argv) {
    World& world = initialize(argc, argv);
    startup(world, argc, argv, true);

    FunctionDefaults<3>::set_k(4);
    FunctionDefaults<3>::set_thresh(1.e-3);
    FunctionDefaults<3>::set_cubic_cell(-4.0, 4.0);

    // The launcher can pass the selector to the test. Each case below sets the value that it needs.
    set_selector(nullptr);

    int failures = 0;
    failures += test_nearest_ancestor(world);
    failures += test_key_below_deepest_level(world);
    failures += test_missing_root_throws(world);
    failures += test_duplicate_key_throws(world);
    failures += test_negative_rank_throws(world);
    failures += test_unmatched_level_zero_key_throws(world);
    failures += test_driver_keeps_root(world, "load_balance() keeps a root with negative residual", NegativeRootCost(), 20.0);
    failures += test_driver_keeps_root(world, "load_balance() keeps a root with zero cost", ZeroCost(), 1.0);

    failures += test_assignment_invariants(world, "assign_lpt()", detail::assign_lpt<1>);
    failures += test_assignment_invariants(world, "assign_sfc()", detail::assign_sfc<1>);
    failures += test_lpt_order(world);
    failures += test_sfc_contiguous_runs(world);
    failures += test_sfc_morton_order(world);
    failures += test_sfc_cuts(world);
    failures += test_sfc_load_bound(world);
    failures += test_assignment_metrics(world);
    failures += test_selector_follows_rank_zero(world);
    failures += test_invalid_selector_throws_on_every_rank(world);
    failures += test_load_balance_reports_the_selection(world);

    set_selector(world.rank() == 0 ? "sfc" : nullptr);
    failures += test_driver_keeps_root(world, "sfc: load_balance() keeps a root with negative residual", NegativeRootCost(), 20.0);
    failures += test_driver_keeps_root(world, "sfc: load_balance() keeps a root with zero cost", ZeroCost(), 1.0);
    set_selector(nullptr);

    world.gop.fence();
    finalize();
    return failures;
}
