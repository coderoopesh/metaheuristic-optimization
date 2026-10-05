// =====================================================================
// Project: Metaheuristic TSP Solver
// Author: Roopesh Singh
// License: MIT License (https://opensource.org/licenses/MIT)
// =====================================================================


#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

// --- OPTIMAL BENCHMARK DATABASE (TSPLIB) ---
const std::unordered_map<std::string, int> OPTIMALS = {
    {"berlin52.tsp", 7542},    {"eil51.tsp", 426},        {"st70.tsp", 675},
    {"eil76.tsp", 538},        {"rat99.tsp", 1211},       {"kroA100.tsp", 21282},
    {"kroB100.tsp", 22141},    {"kroC100.tsp", 20749},    {"kroD100.tsp", 21294},
    {"kroE100.tsp", 22068},    {"rd100.tsp", 7910},       {"eil101.tsp", 629},
    {"lin105.tsp", 14379},     {"pr107.tsp", 44303},      {"bier127.tsp", 118282},
    {"ch130.tsp", 6110},       {"pr136.tsp", 96772},      {"pr144.tsp", 58537},
    {"ch150.tsp", 6528},       {"u159.tsp", 42080},       {"rat195.tsp", 2323},
    {"d198.tsp", 15780},       {"lin318.tsp", 42029},     {"pcb442.tsp", 50778},
    {"rat783.tsp", 8806},      {"vm1748.tsp", 336556},    {"pcb3038.tsp", 137694},
    {"rl5915.tsp", 565530}
};

// --- HELPER FUNCTIONS ---
static inline int calc_cost(const std::vector<int>& tour, const std::vector<std::vector<double>>& dist, int n) {
    double c = 0.0;
    for (int i = 0; i < n; ++i) {
        c += dist[tour[i]][tour[(i + 1) % n]];
    }
    return static_cast<int>(std::round(c));
}

// --- GREEDY INSERTION HELPER FOR RUIN & RECREATE ---
std::vector<int> greedy_insert(std::vector<int> partial_tour, const std::vector<int>& removed_nodes, const std::vector<std::vector<double>>& dist) {
    for (int node : removed_nodes) {
        double best_increase = 1e18;
        int best_pos = 0;
        int m = partial_tour.size();
        for (int i = 0; i < m; ++i) {
            int u = partial_tour[i];
            int v = partial_tour[(i + 1) % m];
            double increase = dist[u][node] + dist[node][v] - dist[u][v];
            if (increase < best_increase) {
                best_increase = increase;
                best_pos = i + 1;
            }
        }
        partial_tour.insert(partial_tour.begin() + best_pos, node);
    }
    return partial_tour;
}

// --- LOCAL SEARCH KERNELS ---
std::vector<int> run_2opt(std::vector<int> best_t, const std::vector<std::vector<double>>& dist, const std::vector<std::vector<int>>& candidates, int n) {
    std::vector<int> pos(n);
    for (int idx = 0; idx < n; ++idx) pos[best_t[idx]] = idx;
    std::vector<bool> dlb(n, false);

    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < n; ++i) {
            int u = best_t[i];
            if (dlb[u]) continue;

            int u_next = best_t[(i + 1) % n];
            bool u_improved = false;

            for (int v : candidates[u]) {
                int j = pos[v];
                int v_next = best_t[(j + 1) % n];

                if (v == u || v == u_next || v_next == u) continue;

                if (dist[u][v] + dist[u_next][v_next] < dist[u][u_next] + dist[v][v_next] - 1e-5) {
                    int left = (i < j) ? i + 1 : j + 1;
                    int right = (i < j) ? j : i;

                    while (left < right) {
                        std::swap(best_t[left], best_t[right]);
                        pos[best_t[left]] = left;
                        pos[best_t[right]] = right;
                        left++; right--;
                    }

                    int prev_u = best_t[(i - 1 + n) % n];
                    int prev_v = best_t[(j - 1 + n) % n];
                    dlb[u] = dlb[u_next] = dlb[v] = dlb[v_next] = dlb[prev_u] = dlb[prev_v] = false;

                    improved = true;
                    u_improved = true;
                    break;
                }
            }
            if (!u_improved) dlb[u] = true;
        }
    }
    return best_t;
}

std::vector<int> run_or_opt(std::vector<int> best_t, const std::vector<std::vector<double>>& dist, const std::vector<std::vector<int>>& candidates, int n) {
    std::vector<int> pos(n);
    for (int idx = 0; idx < n; ++idx) pos[best_t[idx]] = idx;
    std::vector<int> scratch(n);
    std::vector<bool> dlb(n, false);

    bool improved = true;
    while (improved) {
        improved = false;
        for (int L : {3, 2, 1}) {
            for (int i = 0; i < n; ++i) {
                int u_first = best_t[i];
                if (dlb[u_first]) continue;

                int u_last = best_t[(i + L - 1) % n];
                int u_prev = best_t[(i - 1 + n) % n];
                int u_next = best_t[(i + L) % n];

                double removal_saved = dist[u_prev][u_first] + dist[u_last][u_next];
                double removal_added = dist[u_prev][u_next];
                bool u_improved = false;

                for (int v : candidates[u_first]) {
                    int j = pos[v];
                    int rel_j = (j - i + n) % n;
                    if (rel_j < L || rel_j == n - 1) continue;

                    int v_next = best_t[(j + 1) % n];
                    double insertion_added = dist[v][u_first] + dist[u_last][v_next];
                    double insertion_saved = dist[v][v_next];

                    double delta = (removal_added + insertion_added) - (removal_saved + insertion_saved);

                    if (delta < -1e-5) {
                        int idx_out = 0;
                        int p = (i + L) % n;
                        for (int k = 0; k < rel_j + 1 - L; ++k) { scratch[idx_out++] = best_t[p]; p = (p + 1) % n; }
                        p = i;
                        for (int k = 0; k < L; ++k) { scratch[idx_out++] = best_t[p]; p = (p + 1) % n; }
                        p = (j + 1) % n;
                        for (int k = 0; k < n - 1 - rel_j; ++k) { scratch[idx_out++] = best_t[p]; p = (p + 1) % n; }

                        for (int k = 0; k < n; ++k) {
                            best_t[k] = scratch[k];
                            pos[best_t[k]] = k;
                        }

                        dlb[u_first] = dlb[u_last] = dlb[u_prev] = dlb[u_next] = dlb[v] = dlb[v_next] = false;
                        improved = true;
                        u_improved = true;
                        break;
                    }
                }
                if (!u_improved) dlb[u_first] = true;
            }
        }
    }
    return best_t;
}

// --- TOUR FILE EXPORT HELPER ---
void save_tour_file(const std::string& problem_name, const std::vector<int>& tour, int cost) {
    std::string base_name = problem_name;
    size_t last_slash = base_name.find_last_of("/\\");
    if (last_slash != std::string::npos) base_name = base_name.substr(last_slash + 1);
    size_t ext = base_name.find(".tsp");
    if (ext != std::string::npos) base_name = base_name.substr(0, ext);

    std::string tour_filename = base_name + ".tour";
    std::ofstream out(tour_filename);
    if (!out.is_open()) {
        std::cerr << "[-] Error: Could not create tour file: " << tour_filename << "\n";
        return;
    }

    out << "NAME : " << base_name << "\n"
    << "TYPE : TOUR\n"
    << "DIMENSION : " << tour.size() << "\n"
    << "COMMENT : Cost = " << cost << "\n"
    << "TOUR_SECTION\n";

    for (int node : tour) {
        out << (node + 1) << "\n";
    }
    out << "-1\nEOF\n";
    out.close();
    std::cout << "[+] Tour saved successfully to: " << tour_filename << "\n";
}

// --- MAIN SOLVER CLASS ---
class OptimalSolver {
public:
    std::vector<std::vector<double>> dist;
    std::vector<std::pair<double, double>> coords;
    std::vector<int> nodes;
    int n;
    std::string name;
    std::vector<std::vector<int>> candidates;
    std::mt19937 rng;

    OptimalSolver(const std::vector<std::vector<double>>& dist_matrix,
                  const std::pair<double, double>* coords_ptr,
                  const std::vector<int>& nodes_vec,
                  const std::string& problem_name)
    : dist(dist_matrix), nodes(nodes_vec), n(dist_matrix.size()), name(problem_name), rng(42) {

        if (coords_ptr) {
            coords.assign(coords_ptr, coords_ptr + n);
        }

        int max_neighbors = std::min(n - 1, n > 1000 ? 30 : 25);
        candidates.resize(n);
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<double, int>> row;
            row.reserve(n - 1);
            for (int j = 0; j < n; ++j) {
                if (i != j) row.push_back({dist[i][j], j});
            }
            std::partial_sort(row.begin(), row.begin() + max_neighbors, row.end());
            for (int k = 0; k < max_neighbors; ++k) {
                candidates[i].push_back(row[k].second);
            }
        }
    }

    std::vector<int> refine_tour(std::vector<int> tour) {
        while (true) {
            int prev_cost = calc_cost(tour, dist, n);
            tour = run_2opt(tour, dist, candidates, n);
            tour = run_or_opt(tour, dist, candidates, n);
            int curr_cost = calc_cost(tour, dist, n);
            if (curr_cost >= prev_cost) break;
        }
        return tour;
    }

    std::vector<int> solve(int total_kicks = 20000) {
        std::string lookup_name = name;
        size_t last_slash = lookup_name.find_last_of("/\\");
        if (last_slash != std::string::npos) lookup_name = lookup_name.substr(last_slash + 1);
        if (lookup_name.find(".tsp") == std::string::npos) lookup_name += ".tsp";

        int target = 0;
        if (OPTIMALS.count(lookup_name)) {
            target = OPTIMALS.at(lookup_name);
        }

        std::cout << "\n=========================================================\n"
        << " PROBLEM:  " << lookup_name << " | N = " << n << " | TARGET OPTIMAL: " << target << "\n"
        << "=========================================================\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << std::left << std::setw(8) << "Kick" << " | "
        << std::setw(14) << "Method" << " | "
        << std::setw(10) << "Cost" << " | "
        << std::setw(10) << "Error" << " | Time\n";

        std::vector<int> best_init_tour;
        int best_init_cost = 2e9;

        std::vector<int> start_nodes(n);
        std::iota(start_nodes.begin(), start_nodes.end(), 0);
        std::shuffle(start_nodes.begin(), start_nodes.end(), rng);
        int num_starts = std::min(3, n);

        for (int s = 0; s < num_starts; ++s) {
            int curr = start_nodes[s];
            std::vector<int> tour = {curr};
            std::vector<bool> visited(n, false);
            visited[curr] = true;

            for (int step = 1; step < n; ++step) {
                int next_node = -1;
                double min_d = 1e18;
                for (int j = 0; j < n; ++j) {
                    if (!visited[j] && dist[curr][j] < min_d) {
                        min_d = dist[curr][j];
                        next_node = j;
                    }
                }
                tour.push_back(next_node);
                visited[next_node] = true;
                curr = next_node;
            }

            int c = calc_cost(tour, dist, n);
            if (c < best_init_cost) {
                best_init_cost = c;
                best_init_tour = tour;
            }
        }

        std::vector<int> best_tour = refine_tour(best_init_tour);
        int best_cost = calc_cost(best_tour, dist, n);

        auto get_elapsed = [&]() {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double>(now - start_time).count();
        };

        auto format_err = [&](int cost) -> std::string {
            if (target <= 0) return "N/A";
            double err_val = ((double)(cost - target) / target) * 100.0;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(3) << err_val << "%";
            return ss.str();
        };

        std::cout << std::left << std::setw(8) << 0 << " | "
        << std::setw(14) << "Init (NN+LS)" << " | "
        << std::setw(10) << best_cost << " | "
        << std::setw(10) << format_err(best_cost) << " | "
        << std::fixed << std::setprecision(2) << get_elapsed() << "s\n";

        std::vector<int> current_tour = best_tour;
        int current_cost = best_cost;
        int stagnation = 0;

        // --- DYNAMIC STAGE THRESHOLDS BASED ON PROBLEM SIZE ---
        double scale_factor = 1.0;
        if (n > 10000) {
            scale_factor = 0.20;
        } else if (n > 1000) {
            scale_factor = 0.45;
        }

        int stage_1 = std::max(15,  static_cast<int>((n / 10.0) * scale_factor));
        int stage_2 = std::max(30,  static_cast<int>((n / 4.0)  * scale_factor));
        int stage_3 = std::max(60,  static_cast<int>((n / 2.0)  * scale_factor));
        int stage_4 = std::max(100, static_cast<int>((n * 0.35) * scale_factor));
        int stage_5 = std::max(150, static_cast<int>((n * 0.70) * scale_factor));

        int metro_trigger = std::max(20, (int)(stage_2 * 0.6));
        std::uniform_real_distribution<double> uniform_01(0.0, 1.0);

        for (int k = 1; k <= total_kicks; ++k) {
            std::vector<int> temp = current_tour;
            std::string method = "Double-B";

            if (stagnation < stage_1) {
                // Double-Bridge Kick
                std::vector<int> idxs;
                while (idxs.size() < 4) {
                    int r = rng() % n;
                    if (std::find(idxs.begin(), idxs.end(), r) == idxs.end()) idxs.push_back(r);
                }
                std::sort(idxs.begin(), idxs.end());

                std::vector<int> new_t;
                new_t.insert(new_t.end(), temp.begin(), temp.begin() + idxs[0]);
                new_t.insert(new_t.end(), temp.begin() + idxs[2], temp.begin() + idxs[3]);
                new_t.insert(new_t.end(), temp.begin() + idxs[1], temp.begin() + idxs[2]);
                new_t.insert(new_t.end(), temp.begin() + idxs[0], temp.begin() + idxs[1]);
                new_t.insert(new_t.end(), temp.begin() + idxs[3], temp.end());
                temp = new_t;
                method = "Double-B";
            }
            else if (stagnation < stage_2) {
                // Displace Kick
                int s_len = 4 + (rng() % std::max(1, std::min(8, (n / 10) - 4)));
                int idx = rng() % (n - s_len);
                std::vector<int> seg(temp.begin() + idx, temp.begin() + idx + s_len);
                temp.erase(temp.begin() + idx, temp.begin() + idx + s_len);
                int ins = rng() % (temp.size() + 1);
                temp.insert(temp.begin() + ins, seg.begin(), seg.end());
                method = "DISPLACE";
            }
            else if (stagnation < stage_3) {
                // Triple-Bridge Kick
                std::vector<int> idxs;
                int min_gap = std::max(3, n / 100);
                int attempts = 0;
                while (idxs.size() < 6 && attempts < 100) {
                    idxs.clear();
                    while (idxs.size() < 6) {
                        int r = rng() % n;
                        if (std::find(idxs.begin(), idxs.end(), r) == idxs.end()) idxs.push_back(r);
                    }
                    std::sort(idxs.begin(), idxs.end());

                    bool valid = true;
                    for (size_t i = 0; i < idxs.size(); ++i) {
                        int next_idx = (i + 1 < idxs.size()) ? idxs[i + 1] : (idxs[0] + n);
                        if (next_idx - idxs[i] < min_gap) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) break;
                    attempts++;
                }
                if (idxs.size() < 6) {
                    idxs.clear();
                    int step = n / 7;
                    for (int i = 0; i < 6; ++i) idxs.push_back((i + 1) * step % n);
                    std::sort(idxs.begin(), idxs.end());
                }

                std::vector<int> new_t;
                new_t.insert(new_t.end(), temp.begin(), temp.begin() + idxs[0]);
                new_t.insert(new_t.end(), temp.begin() + idxs[4], temp.begin() + idxs[5]);
                new_t.insert(new_t.end(), temp.begin() + idxs[2], temp.begin() + idxs[3]);
                new_t.insert(new_t.end(), temp.begin() + idxs[0], temp.begin() + idxs[1]);
                new_t.insert(new_t.end(), temp.begin() + idxs[3], temp.begin() + idxs[4]);
                new_t.insert(new_t.end(), temp.begin() + idxs[1], temp.begin() + idxs[2]);
                new_t.insert(new_t.end(), temp.begin() + idxs[5], temp.end());
                temp = new_t;
                method = "Triple-B";
            }
            else if (stagnation < stage_4) {
                // Edge-Kill Heuristic
                std::vector<std::pair<double, int>> edge_costs;
                edge_costs.reserve(n);
                for (int i = 0; i < n; ++i) {
                    edge_costs.push_back({dist[temp[i]][temp[(i + 1) % n]], i});
                }
                std::sort(edge_costs.begin(), edge_costs.end(), [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });

                int num_remove = std::max(4, (int)(n * (0.08 + (double)(rng() % 7) * 0.01)));
                std::vector<bool> remove_indices(n, false);
                for (int i = 0; i < num_remove && i < edge_costs.size(); ++i) {
                    remove_indices[edge_costs[i].second] = true;
                }

                std::vector<int> temp_list;
                std::vector<int> removed_nodes;
                for (int i = 0; i < n; ++i) {
                    if (remove_indices[i]) {
                        removed_nodes.push_back(temp[i]);
                    } else {
                        temp_list.push_back(temp[i]);
                    }
                }
                std::shuffle(removed_nodes.begin(), removed_nodes.end(), rng);
                temp = greedy_insert(temp_list, removed_nodes, dist);
                method = "Edge-Kill";
            }
            else if (stagnation < stage_5) {
                // Spectral Ruin Heuristic
                int r_s = (int)(n * (0.10 + uniform_01(rng) * 0.10));
                r_s = std::max(15, std::min(r_s, n - 1));
                int center_node = temp[rng() % n];

                std::vector<std::pair<double, int>> neighbors;
                neighbors.reserve(n);
                for (int j = 0; j < n; ++j) {
                    if (j == center_node) continue;
                    double overlap = 0.0;
                    for (int k = 0; k < n; ++k) {
                        overlap += std::min(dist[center_node][k], dist[j][k]);
                    }
                    neighbors.push_back({overlap / n, j});
                }
                std::sort(neighbors.begin(), neighbors.end());

                std::vector<bool> is_ruined(n, false);
                for (int i = 0; i < r_s && i < neighbors.size(); ++i) {
                    is_ruined[neighbors[i].second] = true;
                }

                std::vector<int> temp_list;
                std::vector<int> ruined_nodes;
                for (int node : temp) {
                    if (is_ruined[node]) {
                        ruined_nodes.push_back(node);
                    } else {
                        temp_list.push_back(node);
                    }
                }
                std::shuffle(ruined_nodes.begin(), ruined_nodes.end(), rng);
                temp = greedy_insert(temp_list, ruined_nodes, dist);
                method = "Spectral Ruin";
            }
            else {
                stagnation = 0;
                current_tour = best_tour;
                current_cost = best_cost;
                continue;
            }

            std::vector<int> refined = refine_tour(temp);
            int refined_cost = calc_cost(refined, dist, n);
            int cost_diff = refined_cost - current_cost;

            if (refined_cost < best_cost) {
                best_cost = refined_cost;
                best_tour = refined;
                current_cost = refined_cost;
                current_tour = refined;
                stagnation = 0;

                std::cout << std::left << std::setw(8) << k << " | "
                << std::setw(14) << method << " | "
                << std::setw(10) << best_cost << " | "
                << std::setw(10) << format_err(best_cost) << " | "
                << std::fixed << std::setprecision(2) << get_elapsed() << "s\n";
            }
            else if (cost_diff < 0) {
                current_cost = refined_cost;
                current_tour = refined;
                stagnation += 1;
            }
            else if (stagnation > metro_trigger) {
                double temp_sched = std::max(0.5, (best_cost * 0.005) * std::pow(0.985, stagnation - metro_trigger));
                double prob = std::exp(-((double)cost_diff) / temp_sched);
                if (uniform_01(rng) < prob) {
                    current_cost = refined_cost;
                    current_tour = refined;
                    stagnation += 1;
                } else {
                    stagnation += 1;
                }
            }
            else {
                stagnation += 1;
            }

            if (target > 0 && best_cost <= target) {
                std::cout << "\n[!] Exact Optimal Solution Found!\n";
                break;
            }
        }

        save_tour_file(name, best_tour, best_cost);
        return best_tour;
    }
};

// --- AUTOMATED GITHUB FILE DOWNLOADER HELPER ---
bool ensure_file_exists(std::string& filename) {
    std::ifstream check(filename);
    if (check.is_open()) return true;

    if (filename.find(".tsp") == std::string::npos) filename += ".tsp";

    std::cout << "[*] File '" << filename << "' not found locally.\n";
    std::cout << "[*] Attempting to download from GitHub repository (mastqe/tsplib)..." << std::endl;

    std::string url = "https://raw.githubusercontent.com/mastqe/tsplib/master/" + filename;
    std::string cmd = "curl -s -L -o " + filename + " " + url;
    int ret = std::system(cmd.c_str());

    std::ifstream verify(filename);
    if (ret == 0 && verify.is_open()) {
        verify.seekg(0, std::ios::end);
        if (verify.tellg() > 50) {
            std::cout << "[+] Successfully downloaded " << filename << "!\n";
            return true;
        }
    }
    std::cerr << "[-] Error: Could not automatically fetch " << filename << " from GitHub.\n";
    std::remove(filename.c_str());
    return false;
}

// --- INTERACTIVE PROMPT MENU ---
void print_menu() {
    std::cout << "\n=========================================================\n"
              << "                     TSP SOLVER           \n"
              << "=========================================================\n"
              << "  Type Popular Instances TSP file Name\n"
              << "  berlin52 for berlin52.tsp  (N = 52), Or\n"
              << "  rl5915 for rl5915.tsp (N = 5,915), Or\n"
              << "  Enter custom problem name or filename (e.g., pr107.tsp)\n"
              << "---------------------------------------------------------\n"
              << "Type File Name: ";
}

// --- MAIN ENTRY POINT ---
int main(int argc, char* argv[]) {
    std::string filename = "";
    int max_kicks = 20000;

    if (argc >= 2) {
        filename = argv[1];
        if (argc >= 3) max_kicks = std::stoi(argv[2]);
    } else {
        print_menu();
        std::string choice;
        std::cin >> choice;
        filename = choice;

        std::cout << "Enter max kicks (default 20000, press Enter to skip): ";
        std::string kick_input;
        std::cin.ignore();
        std::getline(std::cin, kick_input);
        if (!kick_input.empty()) {
            max_kicks = std::stoi(kick_input);
        }
    }

    if (!filename.empty() && 
        filename.find(".tsp") == std::string::npos && 
        filename.find(".TSP") == std::string::npos) {
        filename += ".tsp";
    }

    if (!ensure_file_exists(filename)) return 1;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return 1;
    }

    std::string line;
    std::vector<std::pair<double, double>> coords;
    bool in_coord_section = false;

    while (std::getline(file, line)) {
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        line = line.substr(first);

        if (line.rfind("NODE_COORD_SECTION", 0) == 0) {
            in_coord_section = true;
            continue;
        }
        if (line.rfind("EOF", 0) == 0) break;

        if (in_coord_section) {
            std::stringstream ss(line);
            int id;
            double x, y;
            if (ss >> id >> x >> y) {
                coords.push_back({x, y});
            }
        }
    }
    file.close();

    int n = coords.size();
    if (n == 0) {
        std::cerr << "Error: No coordinates parsed from " << filename << "\n";
        return 1;
    }

    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
    std::vector<int> nodes(n);
    for (int i = 0; i < n; ++i) {
        nodes[i] = i;
        for (int j = 0; j < n; ++j) {
            double dx = coords[i].first - coords[j].first;
            double dy = coords[i].second - coords[j].second;
            dist[i][j] = std::round(std::sqrt(dx * dx + dy * dy));
        }
    }

    OptimalSolver solver(dist, coords.data(), nodes, filename);
    solver.solve(max_kicks);

    return 0;
}