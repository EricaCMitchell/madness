/*
  This file is part of MADNESS.

  Copyright (C) 2007,2010 Oak Ridge National Laboratory

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  For more information please contact:

  Robert J. Harrison
  Oak Ridge National Laboratory
  One Bethel Valley Road
  P.O. Box 2008, MS-6367

  email: harrisonrj@ornl.gov
  tel:   865-241-3937
  fax:   865-572-0680

  $Id$
*/
#ifndef MADNESS_MRA_IBDEUX_H__INCLUDED
#define MADNESS_MRA_IBDEUX_H__INCLUDED

#include <madness/madness_config.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <madness/world/atomicint.h>
#include <madness/world/worlddc.h>

#include <madness/mra/key.h>
#include <madness/mra/funcdefaults.h>

/// \file mra/lbdeux.h
/// \brief Implements (2nd generation) static load/data balancing for functions
/// \ingroup function

namespace madness {

	template<typename T, std::size_t NDIM>
	class FunctionNode;

	template<typename T, std::size_t NDIM>
	class Function;

    /// Evaluation only: the assignment that MAD_LB_EXPERIMENT_ALGORITHM selects
    enum class LBExperimentAlgorithm { LPT, SFC };

    namespace detail {

        struct CostPerProc {
            double cost;
            int proc;
            CostPerProc() : cost(0.0), proc(0) {}
            CostPerProc(double cost, int proc) : cost(cost), proc(proc) {}
            bool operator<(const CostPerProc& other) const {
                return cost > other.cost;  // Want ascending order
            }
        };

        /// Used to sort subtrees into descending order
        template <std::size_t NDIM>
        bool subtree_cost_less(const std::pair<Key<NDIM>,double>& a, const std::pair<Key<NDIM>,double>& b) {
            return a.second < b.second;
        }

        /// Longest-processing-time assignment: the bin packing that load_balance() always used
        template <std::size_t NDIM>
        std::vector<std::pair<Key<NDIM>,ProcessID>> assign_lpt(std::vector<std::pair<Key<NDIM>,double>> subtrees,
                                                               ProcessID nproc) {
            MADNESS_CHECK_THROW(nproc > 0, "assign_lpt: nproc must be positive");
            std::sort(subtrees.begin(), subtrees.end(), subtree_cost_less<NDIM>);

            std::vector<std::pair<Key<NDIM>,ProcessID>> map;
            map.reserve(subtrees.size());

            // Shove the first nproc entries directly into the queue
            std::priority_queue<CostPerProc> costs;
            for (ProcessID p=0; p<nproc && !subtrees.empty(); ++p) {
                const std::pair<Key<NDIM>,double>& f = subtrees.back();
                costs.push(CostPerProc(f.second,p));
                map.push_back(std::make_pair(f.first,p));
                subtrees.pop_back();
            }

            // Process the remainder using the sorting maintained by the priority queue
            while (!subtrees.empty()) {
                const std::pair<Key<NDIM>,double>& f = subtrees.back();
                CostPerProc top = costs.top();
                costs.pop();
                top.cost += f.second;
                costs.push(top);
                map.push_back(std::make_pair(f.first,top.proc));
                subtrees.pop_back();
            }
            return map;
        }

        /// @return The child index of key at the given depth, for 1 <= depth <= key.level().
        /// Dimension i is bit i of the child index.
        template <std::size_t NDIM>
        unsigned int morton_child_index(const Key<NDIM>& key, const Level depth) {
            const Level bit = key.level() - depth;
            unsigned int index = 0;
            for (std::size_t i=0; i<NDIM; ++i) {
                index |= static_cast<unsigned int>((key.translation()[i] >> bit) & 0x1) << i;
            }
            return index;
        }

        /// Adaptive Morton order: lexicographic order of the root-to-key child indices. If one
        /// sequence is a prefix of the other, the shorter one is the ancestor and sorts first.
        template <std::size_t NDIM>
        bool morton_less(const Key<NDIM>& a, const Key<NDIM>& b) {
            const Level common = std::min(a.level(), b.level());
            for (Level depth=1; depth<=common; ++depth) {
                const unsigned int ia = morton_child_index(a, depth);
                const unsigned int ib = morton_child_index(b, depth);
                if (ia != ib) return ia < ib;
            }
            return a.level() < b.level();
        }

        /// @param[in] prefix Nondecreasing cumulative costs
        /// @param[in] begin The cut of the previous run. A nearer cut cannot be below this index,
        /// because the targets increase and prefix does not decrease.
        /// @return The lowest index in [begin, prefix.size()) that is nearest to target
        inline std::size_t nearest_cut(const std::vector<double>& prefix, const std::size_t begin, const double target) {
            const auto first = prefix.begin() + begin;
            auto above = std::lower_bound(first, prefix.end(), target);
            if (above == prefix.end()) --above;
            if (above == first) return begin;
            const auto below = std::lower_bound(first, above, *(above - 1));
            const bool above_is_nearer = std::abs(*above - target) < std::abs(*below - target);
            return static_cast<std::size_t>((above_is_nearer ? above : below) - prefix.begin());
        }

        /// Sorts the subtrees in adaptive Morton order and assigns run k to rank k. A negative cost
        /// counts as zero. A run can be empty, in particular if nproc exceeds the subtree count.
        template <std::size_t NDIM>
        std::vector<std::pair<Key<NDIM>,ProcessID>> assign_sfc(std::vector<std::pair<Key<NDIM>,double>> subtrees,
                                                               ProcessID nproc) {
            MADNESS_CHECK_THROW(nproc > 0, "assign_sfc: nproc must be positive");
            std::sort(subtrees.begin(), subtrees.end(),
                      [](const std::pair<Key<NDIM>,double>& a, const std::pair<Key<NDIM>,double>& b) {
                          return morton_less(a.first, b.first);
                      });

            const std::size_t nsubtree = subtrees.size();
            std::vector<double> prefix(nsubtree + 1, 0.0);
            for (std::size_t i=0; i<nsubtree; ++i) {
                prefix[i+1] = prefix[i] + std::max(0.0, subtrees[i].second);
            }
            const double total = prefix[nsubtree];

            std::vector<std::pair<Key<NDIM>,ProcessID>> map;
            map.reserve(nsubtree);
            std::size_t begin = 0;
            for (ProcessID k=0; k<nproc; ++k) {
                const std::size_t next = static_cast<std::size_t>(k) + 1;
                std::size_t end = nsubtree;
                if (next < static_cast<std::size_t>(nproc)) {
                    end = (total > 0.0) ? nearest_cut(prefix, begin, next * total / nproc)
                                        : next * nsubtree / nproc;
                }
                for (std::size_t i=begin; i<end; ++i) map.push_back(std::make_pair(subtrees[i].first, k));
                begin = end;
            }
            return map;
        }

        /// The load of an assignment. A negative cost counts as zero.
        struct LBAssignmentMetrics {
            double total_cost = 0.0;
            double max_subtree_cost = 0.0;
            double max_rank_cost = 0.0;
            double bound = 0.0;       ///< total_cost / nproc + max_subtree_cost
            bool bound_ok = false;    ///< max_rank_cost is within the bound, with tolerance 1e-12
            std::vector<double> rank_costs;
        };

        template <std::size_t NDIM>
        LBAssignmentMetrics lb_assignment_metrics(const std::vector<std::pair<Key<NDIM>,double>>& subtrees,
                                                  const std::vector<std::pair<Key<NDIM>,ProcessID>>& assignment,
                                                  const ProcessID nproc) {
            const std::unordered_map<Key<NDIM>,ProcessID,Hash<Key<NDIM>>> owner(assignment.begin(), assignment.end());
            LBAssignmentMetrics metrics;
            metrics.rank_costs.assign(nproc, 0.0);
            for (const auto& subtree : subtrees) {
                const double cost = std::max(0.0, subtree.second);
                metrics.rank_costs[owner.at(subtree.first)] += cost;
                metrics.total_cost += cost;
                metrics.max_subtree_cost = std::max(metrics.max_subtree_cost, cost);
            }
            metrics.max_rank_cost = *std::max_element(metrics.rank_costs.begin(), metrics.rank_costs.end());
            metrics.bound = metrics.total_cost / nproc + metrics.max_subtree_cost;
            metrics.bound_ok = metrics.max_rank_cost <= metrics.bound * (1.0 + 1.e-12);
            return metrics;
        }

        /// Collective. Rank 0 reads MAD_LB_EXPERIMENT_ALGORITHM and broadcasts it, so all ranks
        /// agree, and no rank is left in a later collective if the value is not valid.
        /// @return The selected algorithm, or nullopt if the variable is not set
        /// @throw MadnessException on every rank if the value is not valid
        inline std::optional<LBExperimentAlgorithm> select_lb_experiment_algorithm(World& world) {
            std::string value;
            if (world.rank() == 0) {
                if (const char* env = std::getenv("MAD_LB_EXPERIMENT_ALGORITHM")) value = env;
            }
            world.gop.broadcast_serializable(value, 0);
            if (value.empty()) return std::nullopt;
            if (value == "lpt") return LBExperimentAlgorithm::LPT;
            if (value == "sfc") return LBExperimentAlgorithm::SFC;
            if (world.rank() == 0) madness::print("MAD_LB_EXPERIMENT_ALGORITHM has the rejected value", value);
            MADNESS_EXCEPTION("invalid MAD_LB_EXPERIMENT_ALGORITHM", 0);
        }

        inline const char* lb_experiment_name(const LBExperimentAlgorithm algorithm) {
            return (algorithm == LBExperimentAlgorithm::SFC) ? "sfc" : "lpt";
        }

        /// Rank 0 counts the load_balance() calls that the selector reports
        inline long lb_experiment_call_index = 0;

        inline void report_lb_assignment(const LBExperimentAlgorithm algorithm, const std::size_t nsubtree,
                                         const double seconds, const LBAssignmentMetrics& metrics) {
            std::ostringstream line;
            line << std::setprecision(17) << "LBEXP_ASSIGN call=" << lb_experiment_call_index++
                 << " algorithm=" << lb_experiment_name(algorithm)
                 << " subtrees=" << nsubtree
                 << " seconds=" << seconds
                 << " total_cost=" << metrics.total_cost
                 << " max_subtree_cost=" << metrics.max_subtree_cost
                 << " max_rank_cost=" << metrics.max_rank_cost
                 << " bound=" << metrics.bound
                 << " bound_ok=" << (metrics.bound_ok ? "true" : "false")
                 << " rank_costs=";
            for (std::size_t p=0; p<metrics.rank_costs.size(); ++p) {
                line << (p == 0 ? "" : ",") << metrics.rank_costs[p];
            }
            std::cout << line.str() << std::endl;
        }

    } // namespace detail

    template <std::size_t NDIM>
    class LBDeuxPmap : public WorldDCPmapInterface< Key<NDIM> > {
        typedef Key<NDIM> keyT;
        typedef std::pair<keyT,ProcessID> pairT;
        // Only ever queried with find(), never iterated, so ordering buys nothing.
        // Hash<keyT> returns the hash cached in Key; no rehash on lookup.
        typedef std::unordered_map<keyT,ProcessID,Hash<keyT>> mapT;
        mapT map;
        Level max_level = 0; ///< Deepest level of any entry in map
        typedef typename mapT::const_iterator iteratorT;

    public:
        /// @param[in] v The owner of each subtree root. The root key must be present, keys must
        /// be unique, and ranks must be nonnegative. The caller checks the upper rank bound,
        /// because this class does not know the size of the World.
        /// @throw MadnessException if v is not valid
        LBDeuxPmap(const std::vector<pairT>& v) {
            MADNESS_CHECK_THROW(!v.empty(), "LBDeuxPmap: empty assignment");
            map.reserve(v.size());
            for (const pairT& entry : v) {
                MADNESS_CHECK_THROW(entry.second >= 0, "LBDeuxPmap: negative rank");
                const bool inserted = map.insert(entry).second;
                MADNESS_CHECK_THROW(inserted, "LBDeuxPmap: duplicate key");
                max_level = std::max(max_level, entry.first.level());
            }
            MADNESS_CHECK_THROW(map.count(keyT(0)) == 1, "LBDeuxPmap: missing root");
        }

        /// @return The owner of the nearest ancestor of key that is in the map, or of key itself
        /// @throw MadnessException if no ancestor is in the map, which a valid key cannot cause
        ProcessID owner(const keyT& key) const {
            keyT k = (key.level() > max_level) ? key.parent(key.level() - max_level) : key;
            while (true) {
                const iteratorT it = map.find(k);
                if (it != map.end()) return it->second;
                if (k.level() <= 0) {
                    madness::print("LBDeuxPmap: no owner for key", key);
                    MADNESS_EXCEPTION("LBDeuxPmap: lookup failed", 0);
                }
                k = k.parent();
            }
        }

        void print() const {
            madness::print("LBDeuxPmap");
        }
    };



    template <std::size_t NDIM>
    class LBNodeDeux {
        static const int nchild = (1<<NDIM);
        typedef Key<NDIM> keyT;
        typedef LBNodeDeux<NDIM> nodeT;
        typedef WorldContainer<keyT,nodeT> treeT;
        
        double child_cost[nchild]; // Removed volatile since all parallel updates will be under mutex from active message and no unprotected reads
        double my_cost;
        double total_cost;
        bool gotkids;
        bool deleted; ///< Set by deleter(). A retained node can have any cost, even a negative one.
        
        AtomicInt nsummed;

        /// Computes index of child key in this node using last bit of translations
        int index(const keyT& key) {
            int ind = 0;
            for (std::size_t d=0; d<NDIM; ++d) ind += ((key.translation()[d])&0x1) << d;
            return ind;
        }

    public:
        LBNodeDeux()
                : my_cost(0.0), total_cost(0.0), gotkids(false), deleted(false) {
            nsummed = 0;
            for (int i=0; i<nchild; ++i)
                child_cost[i] = 0.0;
        }

        LBNodeDeux(const LBNodeDeux<NDIM>& other) :
            my_cost(other.my_cost), total_cost(other.total_cost), gotkids(other.gotkids), deleted(other.deleted)
        {
            nsummed = other.nsummed;
            for (int i=0; i<nchild; ++i)
                child_cost[i] = other.child_cost[i];
        }

        LBNodeDeux<NDIM>& operator=(const LBNodeDeux<NDIM>& other) {
            for (int i=0; i<nchild; ++i)
                child_cost[i] = other.child_cost[i];
            my_cost = other.my_cost;
            total_cost = other.total_cost;
            gotkids = other.gotkids;
            deleted = other.deleted;
            nsummed = other.nsummed;

            return *this;
        }

        bool has_children() const {
            return gotkids;
        }

        double get_total_cost() const {
            return total_cost;
        }

        /// True if partitioning logically deleted this node, so it is not a subtree root
        bool is_deleted() const {
            return deleted;
        }

        /// Accumulates cost into this node
        void add(double cost, bool got_kids) {
            total_cost = (my_cost += cost);
            gotkids = gotkids || got_kids;
        }

        /// Accumulates cost up the tree from children
        void sum(const treeT& tree, const keyT& child, double value) {
            child_cost[index(child)] = value;
            ++nsummed;
            if (nsummed == nchild) {
                for (int i=0; i<nchild; ++i) total_cost += child_cost[i];
                if (child.level() > 1) {
                    keyT key = child.parent();
                    keyT parent = key.parent();
                    const_cast<treeT&>(tree).task(parent, &nodeT::sum, tree, key, double(total_cost));
                }
            }
        }


        /// Logically deletes this node and its descendants

        /// Cannot actually erase this node from the container since the send() handler
        /// is holding an accessor to it.
        void deleter(const treeT& tree, const keyT& key) {
            deleted = true;
            if (has_children()) {
                for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                    const keyT child = kit.key();
                    const_cast<treeT&>(tree).task(child, &nodeT::deleter, tree, child);
                }
            }
        }

        /// Descends tree deleting all except internal nodes and sub-tree parents
        void partition(const treeT& tree, const keyT& key, double avg) {
            if (has_children()) {
                // Sort children in descending cost order
                keyT keys[nchild];
                double vals[nchild];
                for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                    const keyT child = kit.key();
                    int ind = index(child);
                    keys[ind] = child;
                    vals[ind] = child_cost[ind];
                }
                for (int i=0; i<nchild; ++i) {
                    for (int j=i+1; j<nchild; ++j) {
                        if (vals[i] < vals[j]) {
                            std::swap(vals[i],vals[j]);
                            std::swap(keys[i],keys[j]);
                        }
                    }
                }

                // Split off subtrees in decreasing cost order
                for (int i=0; i<nchild; ++i) {
                    if (total_cost <= avg) {
                        const_cast<treeT&>(tree).task(keys[i], &nodeT::deleter, tree, keys[i]);
                    }
                    else {
                        total_cost -= vals[i];
                        const_cast<treeT&>(tree).task(keys[i], &nodeT::partition, tree, keys[i], avg);
                    }
                }
            }
        }

        template <typename Archive>
        void serialize(Archive& ar) {
            ar & archive::wrap_opaque(this,1);
        }
    };


    template <std::size_t NDIM>
    class LoadBalanceDeux {
        typedef Key<NDIM> keyT;
        typedef LBNodeDeux<NDIM> nodeT;
        typedef WorldContainer<keyT,nodeT> treeT;
        typedef typename treeT::iterator iteratorT;
        typedef typename treeT::const_iterator const_iteratorT;
        World& world;
        treeT tree;


        template <typename T, typename costT>
        struct add_op {
            LoadBalanceDeux* lb;
            const costT& costfn;
            add_op(LoadBalanceDeux* lb, const costT& costfn) : lb(lb), costfn(costfn) {}
            void operator()(const keyT& key, const FunctionNode<T,NDIM>& node) const {
 	        if (lb->tree.is_local(key))
		    lb->tree.send(key, &nodeT::add, costfn(key,node), node.has_children());
		else
		    lb->tree.task(key, &nodeT::add, costfn(key,node), node.has_children());
            }
        };

        /// Sums costs up the tree returning to everyone the total cost
        double sum() {
            world.gop.fence();
            const_iteratorT end = tree.end();
            for (const_iteratorT it=tree.begin(); it!=end; ++it) {
                const keyT& key = it->first;
                const nodeT& node = it->second;
                if (!node.has_children() && key.level() > 0) {
                    tree.task(key.parent(), &nodeT::sum, tree, key, node.get_total_cost());
                }
            }
            world.gop.fence();
            double total;
            keyT key0(0);
            if (world.rank() == tree.owner(key0)) {
                total = tree.find(key0).get()->second.get_total_cost();
            }
            world.gop.broadcast(total, tree.owner(key0));
            world.gop.fence();

            return total;
        }

    public:
        LoadBalanceDeux(World& world)
                : world(world)
                , tree(world, FunctionDefaults<NDIM>::get_pmap()) {
            world.gop.fence();
        };

        /// Accumulates cost from a function
        template <typename T, typename costT>
        void add_tree(const Function<T,NDIM>& f, const costT& costfn, bool fence=false) {
            const_cast<Function<T,NDIM>&>(f).unaryop_node(add_op<T,costT>(this,costfn), fence);
        }

        /// Printing for the curious
        void print_tree(const keyT& key = keyT(0)) {
            Future<iteratorT> futit = tree.find(key);
            iteratorT it = futit.get();
            if (it != tree.end()) {
                for (int i=0; i<key.level(); ++i) std::cout << "  ";
                print(key, it->second.get_total_cost());

                if (it->second.has_children()) {
                    for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                        print_tree(kit.key());
                    }
                }
            }
        }

        /// Per-rank cost bar chart under the given pmap.
        ///
        /// Walks the aggregator tree locally; each entry that is not deleted
        /// contributes its total_cost to bucket pmap->owner(key). Buckets are
        /// summed across ranks and rank 0 prints a horizontal bar chart with
        /// total/avg/max/avg/max/min ratios.
        ///
        /// Call BEFORE load_balance() with the old pmap to see the per-key
        /// distribution that the partitioner is about to act on; call AFTER
        /// with the returned pmap to see how the bin-packed subtrees fell out.
        /// Both invocations sum to the same total work, so the bars are
        /// directly comparable.
        void print_cost_per_rank(const std::string& label,
                                 const std::shared_ptr< WorldDCPmapInterface<keyT> >& pmap,
                                 std::ostream& os = std::cout) {
            const int nproc = world.size();
            std::vector<double> cost(nproc, 0.0);

            const_iteratorT end = tree.end();
            for (const_iteratorT it = tree.begin(); it != end; ++it) {
                if (it->second.is_deleted()) continue;
                cost[pmap->owner(it->first)] += it->second.get_total_cost();
            }
            world.gop.sum(cost.data(), nproc);
            world.gop.fence();
            if (world.rank() != 0) return;

            double total = 0.0;
            double mx = 0.0;
            double mn = std::numeric_limits<double>::max();
            for (double c : cost) {
                total += c;
                if (c > mx) mx = c;
                if (c < mn) mn = c;
            }
            const double avg = (nproc > 0) ? total / nproc : 0.0;
            const int barwidth = 40;

            std::ios::fmtflags saved_flags = os.flags();
            std::streamsize saved_prec = os.precision();

            os << "=== Load balance summary: " << label << " ===\n";
            os << std::fixed << std::setprecision(2)
               << "  total=" << total
               << "  avg=" << avg
               << "  max/avg=" << (avg > 0.0 ? mx / avg : 0.0);
            if (mn > 0.0) os << "  max/min=" << mx / mn;
            os << "\n";
            for (int p = 0; p < nproc; ++p) {
                int n = (mx > 0.0) ? static_cast<int>(std::lround(barwidth * cost[p] / mx)) : 0;
                if (n < 0) n = 0;
                if (n > barwidth) n = barwidth;
                os << "  rank " << std::setw(3) << p << " |"
                   << std::string(n, '#')
                   << std::string(barwidth - n, ' ')
                   << "| " << std::setw(10) << cost[p]
                   << "  (" << (avg > 0.0 ? cost[p] / avg : 0.0) << "x avg)\n";
            }
            os << "================================" << std::endl;

            os.flags(saved_flags);
            os.precision(saved_prec);
        }

        /// Actually does the partitioning of the tree
        std::shared_ptr< WorldDCPmapInterface<keyT> > load_balance(double fac = 1.0, bool printstuff=false) {
            world.gop.fence();
            // Compute full tree of costs
            double avg = sum()/(world.size()*fac);
            //if (world.rank() == 0) print_tree();
            world.gop.fence();

            // Create partitioning
            keyT key0(0);
            if (world.rank() == tree.owner(key0)) {
                tree.send(key0, &nodeT::partition, tree, key0, avg*1.1);
            }
            world.gop.fence();

            // Collect entire vector onto node0
            std::vector< std::pair<keyT,double> > results;
            const_iteratorT end = tree.end();
            for (const_iteratorT it=tree.begin(); it!=end; ++it) {
                if (!it->second.is_deleted()) {
                    results.push_back(std::make_pair(it->first,it->second.get_total_cost()));
                }
            }
            results = world.gop.concat0(results, 128*1024*1024);
            world.gop.fence();

            // Every rank observes the value of rank 0 before any rank-specific assignment path
            const std::optional<LBExperimentAlgorithm> experiment = detail::select_lb_experiment_algorithm(world);

            std::vector< std::pair<keyT,ProcessID> > map;

            if (world.rank() == 0) {
                const ProcessID nproc = world.size();
                if (printstuff) {
                    std::vector< std::pair<keyT,double> > sorted = results;
                    std::sort(sorted.begin(), sorted.end(), detail::subtree_cost_less<NDIM>);
                    print("THESE ARE THE INITIAL SUBTREES");
                    for (unsigned int i=0; i<sorted.size(); ++i) print(i,sorted[i]);
                }

                const double start = wall_time();
                map = (experiment == LBExperimentAlgorithm::SFC) ? detail::assign_sfc<NDIM>(results, nproc)
                                                                 : detail::assign_lpt<NDIM>(results, nproc);
                const double seconds = wall_time() - start;

                if (printstuff || experiment) {
                    const detail::LBAssignmentMetrics metrics = detail::lb_assignment_metrics<NDIM>(results, map, nproc);
                    if (printstuff) {
                        print("THIS IS THE MAP");
                        print(map);
                        print("THESE ARE THE COSTS PER PROCESSOR");
                        for (ProcessID p=0; p<nproc; ++p) print(p,metrics.rank_costs[p]);
                    }
                    if (experiment) detail::report_lb_assignment(*experiment, results.size(), seconds, metrics);
                }
            }

            world.gop.fence();
            world.gop.broadcast_serializable(map, 0);
            world.gop.fence();

            // Return the Procmap

            return std::shared_ptr< WorldDCPmapInterface<keyT> >(new LBDeuxPmap<NDIM>(map));
        }
    };
}


#endif // MADNESS_MRA_IBDEUX_H__INCLUDED

