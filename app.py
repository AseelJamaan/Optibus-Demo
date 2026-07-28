import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces



class BusRoutingAlnsEnv(gym.Env):
    """
    PPO + ALNS environment for clustered school bus routing.

    PPO learns to choose:
    1) destroy operator
    2) repair operator
    3) degree of destruction
    4) temperature level

    Fixes applied (v2):
    - constrained_cost: uses a fixed penalty instead of a cumulative penalty over time
    - make_observation: normalizes temperature to [0, 1] instead of using the raw range [0.1, 2000]
    - observation_space: the upper bound for the temperature feature is now 1.0
    - step: norm_imp uses best_cost as the reference, and reject reward is set to 0.0
    - improvement_list: stores only accepted, better, or best solutions
    - debug printing is available in reset() and reset_for_cluster()
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        all_data: dict,
        cluster_pool: list,
        mode: str = "mixed",
        max_iterations: int = 500,
        stop_time: int = 60,
        max_student_time: int = 90 * 60,
        penalty_factor: float = 1000.0,       # FIX 1: changed from 10000 to avoid cost explosion
        alpha: float = 0.3,
        beta: float = 0.7,
        max_temperature: float = 2000.0,
        seed: int | None = None,
        debug: bool = False,                   # FIX 2: debug mode for useful diagnostic printing
    ):
        super().__init__()

        if mode != "mixed":
            raise ValueError("Only 'mixed' mode is allowed")

        self.all_data = all_data
        self.cluster_pool = cluster_pool

        if not self.all_data:
            raise ValueError("all_data is empty.")
        if not self.cluster_pool:
            raise ValueError("cluster_pool is empty.")

        self.df = None
        self.df_clusters = None
        self.current_file = None

        self.mode = mode
        self.current_mode = None
        self.max_iterations = max_iterations
        self.debug = debug

        self.SCHOOL_NODE = 0
        self.STOP_TIME = stop_time
        self.MAX_STUDENT_TIME = max_student_time
        self.PENALTY_FACTOR = penalty_factor
        self.ALPHA = alpha
        self.BETA = beta

        self.max_temperature = max_temperature
        self.temperature = max_temperature

        self.seed_value = seed if seed is not None else 42
        self._rng = random.Random(self.seed_value)

        self.distance_matrix = None
        self.time_matrix = None
        self.raw_distance_matrix = None

        self.cluster_ids = None
        self.unique_clusters = None

        # [destroy_operator, repair_operator, destroy_level, temperature_level]
        self.action_space = spaces.MultiDiscrete([3, 2, 9, 1000])

        # FIX 3: temperature in the observation is normalized to [0, 1] instead of [0.1, 2000]
        self.observation_space = spaces.Box(
            low=np.array( [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(9,),
            dtype=np.float32,
        )

        self.current_cluster = None
        self.student_nodes = None

        self.initial_solution = None
        self.current_solution = None
        self.best_solution = None

        self.initial_cost = None
        self.current_cost = None
        self.best_cost = None

        self.iteration = 0
        self.done = False
        self.stagcount = 0
        self.improvement = 0
        self.current_updated = 0
        self.current_improved = 0
        self.reward = 0.0
        self.improvement_list = []

    # ------------------------------------------------------------
    # Pure travel time
    # ------------------------------------------------------------
    def route_travel_time(self, route):
        return sum(self.time_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

    # ------------------------------------------------------------
    # Total bus time = travel + stop time
    # ------------------------------------------------------------
    def route_time(self, route):
        travel = self.route_travel_time(route)
        inner_students = len(route) - 2
        stops = inner_students * self.STOP_TIME
        return travel + stops

    # ------------------------------------------------------------
    # Hybrid objective
    # ------------------------------------------------------------
    def hybrid_cost(self, route):
        travel_time = sum(self.time_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        travel_dist = sum(self.distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        inner_students = len(route) - 2
        stops = inner_students * self.STOP_TIME
        return self.ALPHA * (travel_time + stops) + self.BETA * travel_dist

    # ------------------------------------------------------------
    # Total route distance
    # ------------------------------------------------------------
    def route_total_distance(self, route):
        total = 0.0
        for i in range(len(route) - 1):
            total += self.raw_distance_matrix[route[i]][route[i + 1]]
        return total

    # ------------------------------------------------------------
    # Morning student ride time
    # ------------------------------------------------------------
    def student_trip_time_morning(self, route):
        inner = route[1:-1]
        if not inner:
            return 0.0
        travel = sum(self.time_matrix[route[i]][route[i + 1]] for i in range(1, len(route) - 1))
        stops = len(inner) * self.STOP_TIME
        return travel + stops

    # ------------------------------------------------------------
    # Afternoon student ride time
    # ------------------------------------------------------------
    def student_trip_time_afternoon(self, route):
        inner = route[1:-1]
        if not inner:
            return 0.0
        travel = sum(self.time_matrix[route[i]][route[i + 1]] for i in range(0, len(route) - 2))
        stops = len(inner) * self.STOP_TIME
        return travel + stops

    # ------------------------------------------------------------
    # FIX 4: Constrained cost uses a fixed penalty instead of a cumulative penalty
    # Reason: the old penalty formula (10000 * (1 + excess_min)) caused the cost to explode
    # to very large values such as 190,000+, making the cost landscape unstable for PPO learning.
    # The fixed penalty gives a clear and stable punishment for infeasible routes.
    # ------------------------------------------------------------
    def constrained_cost(self, route):
        base = self.hybrid_cost(route)

        if self.current_mode == "morning":
            max_t = self.student_trip_time_morning(route)
        else:
            max_t = self.student_trip_time_afternoon(route)

        if max_t <= self.MAX_STUDENT_TIME:
            return base

        # Fixed penalty: it does not accumulate over time
        return base + self.PENALTY_FACTOR

    # ------------------------------------------------------------
    # Initial route using nearest neighbor
    # ------------------------------------------------------------
    def initial_route(self, student_nodes):
        unvisited = set(int(s) for s in student_nodes)
        if not unvisited:
            return [self.SCHOOL_NODE, self.SCHOOL_NODE]

        route = [self.SCHOOL_NODE]
        curr = self.SCHOOL_NODE

        while unvisited:
            nxt = min(unvisited, key=lambda j: self.distance_matrix[curr][j])
            route.append(nxt)
            unvisited.remove(nxt)
            curr = nxt

        route.append(self.SCHOOL_NODE)
        return route

    # ------------------------------------------------------------
    # Destroy operator 1: random removal
    # ------------------------------------------------------------
    def random_remove(self, route, percent=0.15):
        r = route.copy()
        inner = r[1:-1]
        if len(inner) == 0:
            return r, []

        k = max(1, int(len(inner) * percent))
        removed = self._rng.sample(inner, k)

        for x in removed:
            r.remove(x)

        return r, removed

    # ------------------------------------------------------------
    # Destroy operator 2: worst removal
    # ------------------------------------------------------------
    def worst_remove(self, route, k=7):
        r = route.copy()
        inner = r[1:-1]
        if len(inner) <= 1:
            return r, []

        k = min(k, len(inner))
        scores = []

        for i in range(1, len(r) - 1):
            b = r[i - 1]
            n = r[i]
            a = r[i + 1]

            inc_time = self.time_matrix[b][n] + self.time_matrix[n][a] - self.time_matrix[b][a]
            inc_dist = self.distance_matrix[b][n] + self.distance_matrix[n][a] - self.distance_matrix[b][a]
            inc = self.ALPHA * inc_time + self.BETA * inc_dist
            scores.append((inc, n))

        scores.sort(reverse=True)
        removed = [n for _, n in scores[:k]]

        for n in removed:
            r.remove(n)

        return r, removed

    # ------------------------------------------------------------
    # Destroy operator 3: Shaw removal
    # ------------------------------------------------------------
    def shaw_remove(self, route, k=3, gamma=0.3):
        r = route.copy()
        inner = r[1:-1]
        if not inner:
            return r, []

        k = min(k, len(inner))
        seed = self._rng.choice(inner)
        removed = {seed}

        sc = []
        for n in inner:
            if n == seed:
                continue

            sim_time = self.time_matrix[seed][n]
            sim_dist = self.distance_matrix[seed][n]
            rel = gamma * sim_time + (1 - gamma) * sim_dist
            sc.append((rel, n))

        sc.sort(key=lambda x: x[0])

        for _, n in sc:
            if len(removed) >= k:
                break
            removed.add(n)

        for x in removed:
            r.remove(x)

        return r, list(removed)

    # ------------------------------------------------------------
    # Repair operator 1: regret-2 insertion
    # ------------------------------------------------------------
    def regret_2_insert(self, route, removed):
        r = route.copy()
        removed = removed.copy()

        while removed:
            best_node = None
            best_pos = None
            best_regret = -1

            for node in removed:
                incs = []
                for i in range(1, len(r)):
                    b = r[i - 1]
                    a = r[i]

                    inc_time = self.time_matrix[b][node] + self.time_matrix[node][a] - self.time_matrix[b][a]
                    inc_dist = self.distance_matrix[b][node] + self.distance_matrix[node][a] - self.distance_matrix[b][a]
                    inc = self.ALPHA * inc_time + self.BETA * inc_dist
                    incs.append((inc, i))

                incs.sort(key=lambda x: x[0])
                best_inc = incs[0][0]
                best_idx = incs[0][1]
                second_inc = incs[1][0] if len(incs) > 1 else best_inc
                regret = second_inc - best_inc

                if regret > best_regret:
                    best_regret = regret
                    best_node = node
                    best_pos = best_idx

            r.insert(best_pos, best_node)
            removed.remove(best_node)

        return r

    # ------------------------------------------------------------
    # Repair operator 2: greedy insertion
    # ------------------------------------------------------------
    def greedy_insert(self, route, removed):
        r = route.copy()
        removed = removed.copy()
        self._rng.shuffle(removed)

        for node in removed:
            best_pos = None
            best_cost = float("inf")

            for i in range(1, len(r)):
                trial = r[:i] + [node] + r[i:]
                c = self.constrained_cost(trial)
                if c < best_cost:
                    best_cost = c
                    best_pos = i

            r.insert(best_pos, node)

        return r

    # ------------------------------------------------------------
    # FIX 5: make_observation normalizes temperature to [0, 1]
    # Reason: the neural network can struggle when one feature has a large range such as [0.1, 2000]
    # while the other features are in [0, 1]. Normalization improves learning.
    # ------------------------------------------------------------
    def make_observation(self):
        improvement = float(self.improvement)

        if self.best_cost is None or self.best_cost == 0:
            cost_ratio = 1.0
        else:
            cost_ratio = self.current_cost / self.best_cost
        cost_ratio = min(cost_ratio, 3.0)

        is_current_best = (
            1.0 if self.current_cost == self.best_cost else 0.0
        )

        # FIX 5: temperature is normalized to [0, 1]
        temperature_norm = (self.temperature - 0.1) / (self.max_temperature - 0.1)
        temperature_norm = min(max(temperature_norm, 0.0), 1.0)

        stag_norm = (self.stagcount / self.max_iterations)
        stag_norm = min(stag_norm, 1.0)

        progress = self.iteration / self.max_iterations
        progress = min(progress, 1.0)

        current_updated = float(self.current_updated)
        current_improved = float(self.current_improved)

        mode_flag = 0.0 if self.current_mode == "morning" else 1.0

        state = np.array(
            [
                improvement,
                cost_ratio,
                is_current_best,
                temperature_norm,   # normalized
                stag_norm,
                progress,
                current_updated,
                current_improved,
                mode_flag,
            ],
            dtype=np.float32,
        )

        return state

    # ------------------------------------------------------------
    # Randomly choose episode mode (morning or afternoon)
    # ------------------------------------------------------------
    def choose_episode_mode(self):
        self.current_mode = self._rng.choice(["morning", "afternoon"])

    # ------------------------------------------------------------
    # FIX 6: helper function for diagnostic printing
    # ------------------------------------------------------------
    def _print_debug_info(self, label=""):
        if not self.debug:
            return
        t_morning = self.student_trip_time_morning(self.initial_solution)
        t_afternoon = self.student_trip_time_afternoon(self.initial_solution)
        feasible_m = t_morning <= self.MAX_STUDENT_TIME
        feasible_a = t_afternoon <= self.MAX_STUDENT_TIME
        print(f"\n{'='*50}")
        print(f"[DEBUG] {label}")
        print(f"  File        : {self.current_file}")
        print(f"  Cluster     : {self.current_cluster}")
        print(f"  Mode        : {self.current_mode}")
        print(f"  Students    : {len(self.student_nodes)}")
        print(f"  Initial Cost: {self.initial_cost:.2f}")
        print(f"  Trip Morning: {t_morning/60:.1f} min → {'OK' if feasible_m else 'INFEASIBLE'}")
        print(f"  Trip Aftern : {t_afternoon/60:.1f} min → {'OK' if feasible_a else 'INFEASIBLE'}")
        print(f"{'='*50}")

    # ------------------------------------------------------------
    # Reset randomly to one cluster
    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.choose_episode_mode()

        self.current_file, self.current_cluster = self._rng.choice(self.cluster_pool)

        selected_data = self.all_data[self.current_file]
        self.df = selected_data["df"]
        self.df_clusters = selected_data["df_clusters"]

        dist_cols = [c for c in self.df.columns if "distance" in c.lower()]
        time_cols = [c for c in self.df.columns if "time" in c.lower()]

        if not dist_cols:
            raise ValueError(f"{self.current_file}: no distance columns found")
        if not time_cols:
            raise ValueError(f"{self.current_file}: no time columns found")
        if "cluster_id" not in self.df_clusters.columns:
            raise ValueError(f"{self.current_file}: df_clusters must contain 'cluster_id'")

        self.distance_matrix = self.df[dist_cols].to_numpy(dtype=float)
        self.time_matrix = self.df[time_cols].to_numpy(dtype=float)
        self.raw_distance_matrix = self.distance_matrix.copy()

        self.cluster_ids = self.df_clusters["cluster_id"].to_numpy()
        self.unique_clusters = sorted(c for c in np.unique(self.cluster_ids) if c != -1)

        mask = self.df_clusters["cluster_id"].to_numpy() == self.current_cluster
        local_pos = np.where(mask)[0]

        if len(local_pos) == 0:
            raise ValueError(f"{self.current_file}: cluster {self.current_cluster} has no students")

        self.student_nodes = local_pos.astype(int)

        route = self.initial_route(self.student_nodes)

        self.initial_solution = route.copy()
        self.current_solution = route.copy()
        self.best_solution = route.copy()

        self.initial_cost = self.constrained_cost(route)
        self.current_cost = self.initial_cost
        self.best_cost = self.initial_cost

        self.iteration = 0
        self.done = False
        self.stagcount = 0
        self.improvement = 0
        self.current_updated = 0
        self.current_improved = 0
        self.reward = 0.0
        self.temperature = self.max_temperature
        self.improvement_list = []

        self._print_debug_info("reset()")   # FIX 6

        return self.make_observation(), {}

    # ------------------------------------------------------------
    # Reset for a specific cluster (used in evaluation)
    # ------------------------------------------------------------
    def reset_for_cluster(self, file_name, cluster_id, seed=None, options=None, forced_mode=None):
        super().reset(seed=seed)

        if seed is not None:
            self.seed_value = seed
            self._rng = random.Random(seed)

        if forced_mode is not None:
            if forced_mode not in ("morning", "afternoon"):
                raise ValueError("forced_mode must be 'morning' or 'afternoon'")
            self.current_mode = forced_mode
        else:
            self.choose_episode_mode()

        if (file_name, cluster_id) not in self.cluster_pool:
            raise ValueError(f"({file_name}, {cluster_id}) not found in cluster_pool")

        self.current_file = file_name
        self.current_cluster = cluster_id

        selected_data = self.all_data[self.current_file]
        self.df = selected_data["df"]
        self.df_clusters = selected_data["df_clusters"]

        dist_cols = [c for c in self.df.columns if "distance" in c.lower()]
        time_cols = [c for c in self.df.columns if "time" in c.lower()]

        if not dist_cols:
            raise ValueError(f"{self.current_file}: no distance columns found")
        if not time_cols:
            raise ValueError(f"{self.current_file}: no time columns found")
        if "cluster_id" not in self.df_clusters.columns:
            raise ValueError(f"{self.current_file}: df_clusters must contain 'cluster_id'")

        self.distance_matrix = self.df[dist_cols].to_numpy(dtype=float)
        self.time_matrix = self.df[time_cols].to_numpy(dtype=float)
        self.raw_distance_matrix = self.distance_matrix.copy()

        self.cluster_ids = self.df_clusters["cluster_id"].to_numpy()
        self.unique_clusters = sorted(c for c in np.unique(self.cluster_ids) if c != -1)

        mask = self.df_clusters["cluster_id"].to_numpy() == self.current_cluster
        local_pos = np.where(mask)[0]

        if len(local_pos) == 0:
            raise ValueError(f"{self.current_file}: cluster {self.current_cluster} has no students")

        self.student_nodes = local_pos.astype(int)

        route = self.initial_route(self.student_nodes)

        self.initial_solution = route.copy()
        self.current_solution = route.copy()
        self.best_solution = route.copy()

        self.initial_cost = self.constrained_cost(route)
        self.current_cost = self.initial_cost
        self.best_cost = self.initial_cost

        self.iteration = 0
        self.done = False
        self.stagcount = 0
        self.improvement = 0
        self.current_updated = 0
        self.current_improved = 0
        self.reward = 0.0
        self.temperature = self.max_temperature
        self.improvement_list = []

        self._print_debug_info("reset_for_cluster()")  # FIX 6

        return self.make_observation(), {}

    # ------------------------------------------------------------
    # Acceptance rule (Simulated Annealing)
    # ------------------------------------------------------------
    def consider_candidate(self, current_cost, candidate_cost):
        if candidate_cost < self.best_cost:
            return "best"

        if candidate_cost < current_cost:
            return "better"

        probability = math.exp(-(candidate_cost - current_cost) / max(self.temperature, 1e-8))
        if self._rng.random() < probability:
            return "accept"

        return "reject"

    # ------------------------------------------------------------
    # One PPO step = one ALNS iteration
    # ------------------------------------------------------------
    def step(self, action):
        self.iteration += 1
        self.stagcount += 1
        self.reward = 0.0
        self.improvement = 0
        self.current_updated = 0
        self.current_improved = 0

        current_route = self.current_solution.copy()
        current_cost = self.current_cost

        destroy_ops = [self.random_remove, self.worst_remove, self.shaw_remove]
        repair_ops = [self.regret_2_insert, self.greedy_insert]

        d_idx, r_idx, destroy_level_idx, temp_idx = action

        # PPO chooses temperature
        T_min = 0.1
        N = self.action_space.nvec[3]
        self.temperature = T_min + (int(temp_idx) / (N - 1)) * (self.max_temperature - T_min)

        # PPO chooses destruction degree
        factors = {
            0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5,
            5: 0.6, 6: 0.7, 7: 0.8, 8: 0.9
        }
        frac = factors[int(destroy_level_idx)]

        inner_count = max(1, len(current_route) - 2)

        if int(d_idx) == 0:
            partial, removed = destroy_ops[int(d_idx)](current_route, percent=frac)
        elif int(d_idx) == 1:
            k = max(1, int(frac * inner_count))
            partial, removed = destroy_ops[int(d_idx)](current_route, k=k)
        else:
            k = max(1, int(frac * inner_count))
            partial, removed = destroy_ops[int(d_idx)](current_route, k=k)

        if len(removed) == 0:
            if self.iteration >= self.max_iterations:
                self.done = True
            return self.make_observation(), self.reward, self.done, False, {}

        candidate_route = repair_ops[int(r_idx)](partial, removed)
        candidate_cost = self.constrained_cost(candidate_route)

        outcome = self.consider_candidate(current_cost, candidate_cost)

        old_cost = current_cost

        # ============================================================
        # REWARD SHAPING (v2)
        #
        # Sources:
        #   - Ropke & Pisinger (2006): σ1=33, σ2=9, σ3=13
        #   - PPO-ALNS (J. Combinatorial Optimization, 2025): normalized bonus
        #
        # Changes from v1:
        #   1. norm_imp uses best_cost as the reference, which is more stable than initial_cost
        #   2. reject = 0.0 instead of -0.5, to avoid punishing exploration
        #   3. improvement_list stores only accepted solutions
        # ============================================================

        # FIX 7: norm_imp uses best_cost as the reference instead of initial_cost
        # Reason: best_cost updates during the search, giving a more accurate learning signal,
        # and it is always less than or equal to initial_cost, so it avoids inflated values.
        ref_cost = max(self.best_cost, 1e-8)
        norm_imp = (old_cost - candidate_cost) / ref_cost
        norm_imp = max(-1.0, min(1.0, norm_imp))   # strict clipping

        if outcome == "best":

            # New global best — σ1=33 in Ropke & Pisinger
            immediate_reward = 10.0
            immediate_reward += 2.0 * norm_imp

            self.best_solution = candidate_route.copy()
            self.current_solution = candidate_route.copy()
            self.best_cost = candidate_cost
            self.current_cost = candidate_cost
            self.improvement = 1
            self.current_updated = 1
            self.current_improved = 1
            self.stagcount = 0

            # FIX 8: improvement_list only stores accepted solutions
            self.improvement_list.append(old_cost - candidate_cost)

        elif outcome == "better":
            # Better than current — σ2=9
            immediate_reward = 3.0
            immediate_reward += 1.0 * norm_imp

            self.current_solution = candidate_route.copy()
            self.current_cost = candidate_cost

            self.improvement = 1
            self.current_updated = 1
            self.current_improved = 1

            self.improvement_list.append(old_cost - candidate_cost)

        elif outcome == "accept":
            # Accepted by SA — σ3=13, higher than better to encourage exploration
            immediate_reward = 1.5

            self.current_solution = candidate_route.copy()
            self.current_cost = candidate_cost

            self.current_updated = 1

            # Do not add this to improvement_list because the solution is worse than the current solution

        else:
            # FIX 9: reject = 0.0 instead of -0.5
            # Reason: the rejection penalty was discouraging the agent from exploration
            # and made it greedy too early, which harms the SA behavior
            immediate_reward = 0.0

        self.reward = immediate_reward

        if self.iteration >= self.max_iterations:
            self.done = True

        return self.make_observation(), float(self.reward), self.done, False, {}

    # ------------------------------------------------------------
    # Run several episodes for one cluster and choose best one
    # Priority:
    # 1) feasible under 90 minutes
    # 2) lower cost
    # ------------------------------------------------------------
    def run_best_all_clusters(self, model, episodes_per_cluster=10, deterministic=True):
        all_best_results = []
        self.all_episode_results = []#stability
        unique_pairs = sorted(set(self.cluster_pool), key=lambda x: (str(x[0]), x[1]))

        for file_name, cluster_id in unique_pairs:

            all_results = []

            for ep in range(episodes_per_cluster):

                # =========================
                # MORNING
                # =========================
                obs, _ = self.reset_for_cluster(
                    file_name,
                    cluster_id,
                    seed=100 + ep,
                    forced_mode="morning"
                )

                done = False
                truncated = False
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, truncated, info = self.step(action)

                morning_route = self.best_solution.copy()
                TD_morning = float(self.route_total_distance(morning_route))
                TT_morning = float(self.route_time(morning_route))
                student_trip_time_morning = self.student_trip_time_morning(morning_route)
                feasible_morning = bool(student_trip_time_morning <= self.MAX_STUDENT_TIME)

                # =========================
                # AFTERNOON
                # =========================
                obs, _ = self.reset_for_cluster(
                    file_name,
                    cluster_id,
                    seed=200 + ep,
                    forced_mode="afternoon"
                )

                done = False
                truncated = False
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, truncated, info = self.step(action)

                afternoon_route = self.best_solution.copy()
                TD_afternoon = float(self.route_total_distance(afternoon_route))
                TT_afternoon = float(self.route_time(afternoon_route))
                student_trip_time_afternoon = self.student_trip_time_afternoon(afternoon_route)
                feasible_afternoon = bool(student_trip_time_afternoon <= self.MAX_STUDENT_TIME)

                # =========================
                # COMBINED
                # =========================
                TD_total = TD_morning + TD_afternoon
                TT_total = TT_morning + TT_afternoon
                OBJ_total = 0.7 * TD_total + 0.3 * TT_total

                result = {
                    "episode": ep + 1,
                    "file": file_name,
                    "cluster": int(cluster_id),

                    "morning_route": morning_route,
                    "afternoon_route": afternoon_route,

                    "TD_morning": TD_morning,
                    "TD_afternoon": TD_afternoon,
                    "TD_total": TD_total,

                    "TT_morning": TT_morning,
                    "TT_afternoon": TT_afternoon,
                    "TT_total": TT_total,

                    "student_trip_time_morning": float(student_trip_time_morning),
                    "student_trip_time_afternoon": float(student_trip_time_afternoon),

                    "feasible_morning_90min": feasible_morning,
                    "feasible_afternoon_90min": feasible_afternoon,
                    "feasible_90min": feasible_morning and feasible_afternoon,

                    "OBJ_total": float(OBJ_total),
                }

                all_results.append(result)
                self.all_episode_results.append(result) #stability
            # =========================
            # BEST RUN SELECTION
            # =========================
            feasible_results = [r for r in all_results if r["feasible_90min"]]

            if feasible_results:
                best_result = min(feasible_results, key=lambda x: x["OBJ_total"])
            else:
                best_result = min(all_results, key=lambda x: x["OBJ_total"])

            all_best_results.append(best_result)

        return all_best_results


    # ------------------------------------------------------------
    # Print results like ALNS
    # ------------------------------------------------------------
    def print_results_like_alns(self, results):
        for res in results:
            file_name = res["file"]
            cluster_id = res["cluster"]
            route = res["best_route"]
            bus_time_min = res["bus_time_sec"] / 60
            student_time_min = res["student_trip_time_sec"] / 60
            bus_distance_m = res["bus_distance_m"]
            bus_distance_km = bus_distance_m / 1000
            students_count = len(route) - 2

            print("\n==================================================")
            print(f"FILE: {file_name}  |  BUS / CLUSTER {cluster_id}  |  STUDENTS: {students_count}")
            print("==================================================")

            if res["mode"] == "morning":
                print("\nMORNING ROUTE:")
            else:
                print("\nAFTERNOON ROUTE:")

            print("Best Episode:", res["episode"])
            print("Route:", route)
            print("Bus Total Time (min):", bus_time_min)
            print("Student Trip Time (min):", student_time_min)
            print("Bus Total Distance (meters):", bus_distance_m)
            print("Bus Total Distance (km):", bus_distance_km)

            if res["feasible_90min"]:
                if res["mode"] == "morning":
                    print("Morning student trip time is within 90 minutes.")
                else:
                    print("Afternoon student trip time is within 90 minutes.")
            else:
                if res["mode"] == "morning":
                    print("Morning student trip time EXCEEDS 90 minutes!")
                else:
                    print("Afternoon student trip time EXCEEDS 90 minutes!")


#-------------------------------#
#stability calculations
def compute_stability(env_test):
    stability_runs_df = pd.DataFrame(env_test.all_episode_results)

    stability_df = stability_runs_df.groupby(["file", "cluster"]).agg(
        TT_mean=("TT_total", "mean"),
        TT_std=("TT_total", "std"),
        TD_mean=("TD_total", "mean"),
        TD_std=("TD_total", "std"),
        OBJ_mean=("OBJ_total", "mean"),
        OBJ_std=("OBJ_total", "std"),
        feasible_rate=("feasible_90min", "mean")
    ).reset_index()

    overall_stability = pd.DataFrame([{
        "file": "OVERALL_STABILITY",
        "cluster": "-",
        "TT_std": stability_runs_df["TT_total"].std(),
        "TD_std": stability_runs_df["TD_total"].std(),
        "OBJ_std": stability_runs_df["OBJ_total"].std()
    }])

    print("\nSTABILITY PER CLUSTER:")
    print(stability_df)

    print("\nOVERALL STABILITY:")
    print(overall_stability)

    return stability_df, overall_stability, stability_runs_df
# ------------------------------------------------------------
# Train PPO
# ------------------------------------------------------------


# ============================================================
# GRADIO DEMO - DRL-ALNS + PPO VERSION
# ============================================================

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from math import ceil, floor
import random, math
import requests
import base64

# ============================================================
# 0) GLOBAL CONSTANTS
# ============================================================

distance_matrix = None
time_matrix = None

SCHOOL_NODE = 0
STOP_TIME = 60
MAX_STUDENT_TIME = 90 * 60
PENALTY_FACTOR = 10000.0
ALPHA = 0.3
BETA  = 0.7

# ============================================================
# DRL-ALNS Environment is defined above and replaces the old ALNS routing algorithm.
# ============================================================

# ============================================================
# 9) CLUSTERING
# ============================================================

def run_clustering(merged_file):
    global distance_matrix, time_matrix

    if merged_file is None:
        return None, None, None, " Please upload the merged student CSV file first.", None

    df = pd.read_csv(merged_file)
    df = df.reset_index(drop=True)

    dist_cols = [c for c in df.columns if "distance" in c.lower()]
    time_cols = [c for c in df.columns if "time" in c.lower()]

    distance_matrix = df[dist_cols].to_numpy(dtype=float)
    time_matrix     = df[time_cols].to_numpy(dtype=float)

    df["node_id"] = df.index

    df_students = df.iloc[1:].reset_index(drop=True).copy()
    df_students["node_id"] = df_students.index + 1

    N = len(df_students)
    coords = df_students[["lat", "lon"]].values

    scaler = StandardScaler()
    coords_norm = scaler.fit_transform(coords)

    min_cap = 14
    max_cap = 44

    k_min = ceil(N / max_cap)
    k_max = floor(N / min_cap)
    possible_K = list(range(k_min, k_max + 1))

    if N < 28:
        possible_K = [1]
    elif N == 28:
        possible_K = [2]
    else:
        possible_K = [k for k in possible_K if k >= 2]

    results = {"K": [], "sil": [], "chi": [], "dbi": [], "labels": []}

    for k in possible_K:
        km = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=300,
            max_iter=700,
            random_state=42
        )
        labels = km.fit_predict(coords_norm)
        sizes = np.bincount(labels, minlength=k)

        if k == 2:
            ratio = min(sizes) / max(sizes)
            if ratio < 0.6:
                continue
        else:
            if (sizes < min_cap).any() or (sizes > max_cap).any():
                continue

        sil = silhouette_score(coords_norm, labels)
        dbi = davies_bouldin_score(coords_norm, labels)
        chi = calinski_harabasz_score(coords_norm, labels)

        results["K"].append(k)
        results["sil"].append(sil)
        results["chi"].append(chi)
        results["dbi"].append(dbi)
        results["labels"].append(labels)

    if len(results["K"]) == 0:
        return None, None, None, " No feasible bus configuration found.", None

    rank_df = pd.DataFrame(results)
    rank_df["Silhouette Rank"] = rank_df["sil"].rank(ascending=False)
    rank_df["CH Rank"] = rank_df["chi"].rank(ascending=False)
    rank_df["DBI Rank"] = rank_df["dbi"].rank(ascending=True)
    rank_df["Internal Avg Rank"] = (rank_df["Silhouette Rank"] + rank_df["CH Rank"] + rank_df["DBI Rank"]) / 3
    rank_df["Final Rank"] = rank_df["Internal Avg Rank"].rank(ascending=True)
    rank_df = rank_df.sort_values("Final Rank")

    best_row = rank_df.iloc[0]
    best_labels = best_row["labels"]
    best_k = int(best_row["K"])

    df_students["cluster_ai"] = best_labels.astype(int) + 1

    summary_rows = []
    for cid in sorted(df_students["cluster_ai"].unique()):
        part = df_students[df_students["cluster_ai"] == cid]
        summary_rows.append({
            "رقم الباص": int(cid),
            "Bus / Cluster ID": int(cid),
            "Students Count": len(part)
        })
    summary_df = pd.DataFrame(summary_rows)

    msg = f" تم تجهيز الباصات بنجاح. عدد الباصات: {best_k}"

    return df, df_students, rank_df, msg, summary_df

# ============================================================
# 10) HELPERS
# ============================================================

def get_students_for_bus(bus_id, df_students):
    try:
        bus_id = int(bus_id)
        subset = df_students[df_students["cluster_ai"] == bus_id]

        student_list = [
            f"{row['node_id']} - {row['student_name']}"
            for _, row in subset.iterrows()
        ]

        return gr.update(choices=student_list, value=student_list)
    except:
        return gr.update(choices=[], value=[])

def parse_selected_nodes(selected_list):
    if not selected_list:
        return []
    ids = []
    for item in selected_list:
        try:
            nid = int(str(item).split("-")[0].strip())
            ids.append(nid)
        except:
            continue
    return ids

def convert_route_to_df_indices(route, df_full):
    converted = []
    for nid in route:
        if nid == SCHOOL_NODE:
            converted.append(0)
        else:
            match = df_full.index[df_full["node_id"] == nid]
            converted.append(int(match[0]) if len(match) > 0 else 0)
    return converted

# ============================================================
# 11) OSRM ROAD ROUTING
# ============================================================

import folium

LAT_COL = "lat"
LON_COL = "lon"


def _fetch_osrm(coords):
    """
    Fetch real road geometry from OSRM for a list of (lat, lon) waypoints.
    Returns list of (lat, lon) along the actual road network.
    Falls back to straight-line coords on any error.
    """
    # OSRM expects lon,lat order
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = (
    "https://router.project-osrm.org/"
    f"route/v1/driving/{coord_str}"
)
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("code") == "Ok":
            road_coords = [
                (pt[1], pt[0])
                for pt in data["routes"][0]["geometry"]["coordinates"]
            ]
            return road_coords
    except Exception as e:
        print(f"OSRM error: {e}")
    return coords  # fallback: straight lines


def get_road_route(coords, chunk_size=25):
    """
    Split long routes into chunks (OSRM has a ~100-waypoint limit on the
    public server; 25 keeps requests fast and reliable).
    Joins segments without duplicating the shared boundary waypoint.
    """
    if len(coords) <= chunk_size:
        return _fetch_osrm(coords)

    full_road = []
    for i in range(0, len(coords) - 1, chunk_size - 1):
        chunk = coords[i:i + chunk_size]
        segment = _fetch_osrm(chunk)
        if not full_road:
            full_road.extend(segment)
        else:
            full_road.extend(segment[1:])   # skip duplicated boundary point

    return full_road if full_road else coords


def plot_single_route_map(route, df, title="Bus Route"):
    """
    Draw the route on a Folium map following the real road network via OSRM.
    Stop markers are placed at the exact student coordinates.
    """
    # Waypoint coordinates in (lat, lon) order
    waypoint_coords = [(df.loc[n, LAT_COL], df.loc[n, LON_COL]) for n in route]

    # Fetch real road geometry
    road_coords = get_road_route(waypoint_coords)

    m = folium.Map(
        location=waypoint_coords[0],
        zoom_start=13,
        control_scale=True
    )

    # Draw the road-following polyline
    folium.PolyLine(
        road_coords,          # ← real road path, not straight lines
        weight=6,
        color="#1f77b4",
        opacity=0.9,
        tooltip=title
    ).add_to(m)

    # School marker (first and last node)
    folium.Marker(
        location=waypoint_coords[0],
        tooltip="School",
        popup="School",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    # Student stop markers
    stop_no = 1
    for i in range(1, len(route) - 1):
        node = route[i]
        folium.Marker(
            location=(df.loc[node, LAT_COL], df.loc[node, LON_COL]),
            tooltip=f"Stop {stop_no}",
            popup=f"Stop {stop_no} | Node {node}",
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-size:12px;
                    font-weight:bold;
                    color:white;
                    background:#1f77b4;
                    border-radius:14px;
                    width:26px;
                    height:26px;
                    text-align:center;
                    line-height:26px;
                    border:2px solid white;
                ">
                    {stop_no}
                </div>
            """)
        ).add_to(m)
        stop_no += 1

    m.fit_bounds(waypoint_coords)
    return m._repr_html_()


# ============================================================
# 12) ROUTING using Saved PPO-based DRL-ALNS Model
# ============================================================

# Load the trained PPO model once
# ============================================================
# PPO MODEL LOCATION
# This is the saved trained PPO model used by the DRL-ALNS routing stage.
# ============================================================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "ppo_bus_alns_shared.zip"
)

ppo_model = None


def get_ppo_model():
    global ppo_model

    if ppo_model is None:
        from stable_baselines3 import PPO

        ppo_model = PPO.load(
            MODEL_PATH,
            device="cpu"
        )

    return ppo_model

def run_routing(bus_id, selected_students, df_full, df_students):
    if df_full is None or df_students is None:
        return None, "الرجاء رفع البيانات أولاً.", None, "", None

    if bus_id is None:
        return None, " الرجاء اختيار الباص أولاً.", None, "", None

    cid = int(bus_id)

    # Get students belonging to the selected bus
    sub = df_students[df_students["cluster_ai"] == cid].copy()
    local_node_ids = sub["node_id"].to_numpy().astype(int)

    # Handle selected / absent students
    chosen_nodes = parse_selected_nodes(selected_students)

    if chosen_nodes:
        student_nodes = [n for n in local_node_ids if n in chosen_nodes]
    else:
        student_nodes = list(local_node_ids)

    if len(student_nodes) == 0:
        return None, " لا يوجد طلاب محددين لهذا الباص.", None, "", None

    # ============================================================
    # Prepare data for DRL-ALNS Environment
    # ============================================================

    df_new = df_full.copy().reset_index(drop=True)

    # All students are initially excluded using cluster_id = -1
    df_clusters_new = pd.DataFrame({
        "cluster_id": [-1] * len(df_new)
    })

    # Assign only selected / present students to the current bus cluster
    for node_id in student_nodes:
        node_id = int(node_id)
        if 0 <= node_id < len(df_clusters_new):
            df_clusters_new.loc[node_id, "cluster_id"] = cid

    current_file_name = "CURRENT_DEMO_DATA"

    all_data_current = {
        current_file_name: {
            "df": df_new,
            "df_clusters": df_clusters_new
        }
    }

    current_pool = [(current_file_name, cid)]

    # ============================================================
    # Create DRL-ALNS Environment
    # ============================================================

    env_drl_alns = BusRoutingAlnsEnv(
        all_data=all_data_current,
        cluster_pool=current_pool,
        mode="mixed",
        max_iterations=500,
        penalty_factor=1000.0,
        seed=123,
        debug=False
    )

    # ============================================================
    # Apply Saved PPO-based DRL-ALNS Model
    # ============================================================
    model = get_ppo_model()
    drl_results = env_drl_alns.run_best_all_clusters(
        model=model,
        episodes_per_cluster=5,
        deterministic=True
    )

    r = drl_results[0]

    best_morning_route = r["morning_route"]
    best_afternoon_route = r["afternoon_route"]

    # ============================================================
    # Compute route metrics
    # ============================================================

    morning_total = r["TT_morning"]
    morning_students_time = r["student_trip_time_morning"]
    morning_distance = r["TD_morning"]

    afternoon_total = r["TT_afternoon"]
    afternoon_students_time = r["student_trip_time_afternoon"]
    afternoon_distance = r["TD_afternoon"]

    morning_status = (
        "مناسب ضمن الوقت المحدد"
        if morning_students_time <= MAX_STUDENT_TIME
        else "يحتاج مراجعة لتقليل وقت الرحلة"
    )

    afternoon_status = (
        "مناسب ضمن الوقت المحدد"
        if afternoon_students_time <= MAX_STUDENT_TIME
        else "يحتاج مراجعة لتقليل وقت الرحلة"
    )

    # ============================================================
    # Display morning result
    # ============================================================

    morning_text_out = f"""
<div class='friendly-route-card'>
  <div class='route-badge'> المسار الصباحي</div>
  <h3>باص {cid} جاهز للانطلاق</h3>
  <div class='friendly-metrics'>
    <div><b>{len(best_morning_route) - 2}</b><span>محطة</span></div>
    <div><b>{morning_students_time / 60:.1f}</b><span>دقيقة وقت الطالب</span></div>
    <div><b>{morning_distance / 1000:.2f}</b><span>كم</span></div>
  </div>
  <p class='route-status'>{morning_status}</p>
</div>
"""

    # ============================================================
    # Display afternoon result
    # ============================================================

    afternoon_text_out = f"""
<div class='friendly-route-card'>
  <div class='route-badge'> مسار العودة</div>
  <h3>باص {cid} جاهز للانطلاق</h3>
  <div class='friendly-metrics'>
    <div><b>{len(best_afternoon_route) - 2}</b><span>محطة</span></div>
    <div><b>{afternoon_students_time / 60:.1f}</b><span>دقيقة وقت الطالب</span></div>
    <div><b>{afternoon_distance / 1000:.2f}</b><span>كم</span></div>
  </div>
  <p class='route-status'>{afternoon_status}</p>
</div>
"""

    # ============================================================
    # Plot routes on map
    # ============================================================

    fig_m = plot_single_route_map(
        best_morning_route,
        df_full,
        title=f"Bus {cid} | Morning Route | DRL-ALNS"
    )

    fig_a = plot_single_route_map(
        best_afternoon_route,
        df_full,
        title=f"Bus {cid} | Afternoon Route | DRL-ALNS"
    )

    # ============================================================
    # Build ETA table
    # ============================================================

    routes_dict = {
        cid: {
            "morning": best_morning_route,
            "afternoon": best_afternoon_route
        }
    }

    df_eta = build_eta_dataframe_from_routes(
        routes_dict,
        df_students,
        distance_matrix
    )

    return fig_m, morning_text_out, fig_a, afternoon_text_out, df_eta

# ============================================================
# 13) ETA COMPUTATION
# ============================================================

SPEED_KMH = 38.0
STOP_TIME_ETA = 1  # minute


def dist_to_minutes(dist_meters, speed_kmh=SPEED_KMH):
    dist_km = dist_meters / 1000.0
    time_hours = dist_km / speed_kmh
    return time_hours * 60.0


def compute_student_etas_distance(route, distance_matrix):
    etas = {}
    cumulative = 0.0

    for i in range(1, len(route) - 1):
        a = route[i - 1]
        b = route[i]
        dist_m = distance_matrix[a][b]
        seg_time = dist_to_minutes(dist_m)
        cumulative += seg_time
        eta = cumulative + (i * STOP_TIME_ETA)
        etas[b] = (round(eta, 2), i)

    return etas


def build_eta_dataframe_from_routes(routes_dict, df_students, distance_matrix):
    all_rows = []

    for cid, routes in routes_dict.items():
        for trip_type in ["morning", "afternoon"]:
            route = routes[trip_type]
            etas = compute_student_etas_distance(route, distance_matrix)

            for node, (eta, order) in etas.items():
                row = df_students[df_students["node_id"] == node].iloc[0]
                all_rows.append({
                    "bus_id": int(cid),
                    "trip_type": trip_type,
                    "order_in_route": order,
                    "student_id": row.get("id", node),
                    "student_name": row["student_name"],
                    "node_id": int(node),
                    "eta_minutes": eta,
                    "phone": str(row.get("phone", ""))
                })

    df_eta = pd.DataFrame(all_rows)
    df_eta["trip_type"] = pd.Categorical(df_eta["trip_type"], ["morning", "afternoon"], ordered=True)
    return df_eta.sort_values(["bus_id", "trip_type", "order_in_route"]).reset_index(drop=True)


# ============================================================
# 14) TWILIO WHATSAPP NOTIFICATIONS
# ============================================================


from twilio.rest import Client
from datetime import datetime

account_sid = "YOUR_TWILIO_ACCOUNT_SID"
auth_token  = "YOUR_TWILIO_AUTH_TOKEN"
client = Client(account_sid, auth_token)


def send_whatsapp_notifications(df_eta, trip_type_selected):
    df_filtered = df_eta[df_eta["trip_type"] == trip_type_selected]

    if len(df_filtered) == 0:
        return " No students found for this trip type."

    df_sample = df_filtered.sample(n=min(5, len(df_filtered)))

    logs = []

    for _, row in df_sample.iterrows():
        student = row["student_name"]
        eta     = row["eta_minutes"]
        phone   = str(row["phone"])
        to_whatsapp = "whatsapp:+966" + phone

        if trip_type_selected == "morning":
            body = f"عزيزي ولي الأمر،\nسيصل باص المدرسة لأخذ طفلكم {student} خلال {eta} دقائق."
        else:
            body = f"عزيزي ولي الأمر،\nسيصل باص المدرسة وبرفقته طفلكم {student} خلال {eta} دقائق."

        try:
            msg = client.messages.create(
                body=body,
                from_="whatsapp:+14155238886",
                to=to_whatsapp
            )
            logs.append(f" Sent to {student} ({phone})")
        except Exception as e:
            logs.append(f" Failed for {student}: {e}")

    return "\n".join(logs)


# ============================================================
# 15) PREMIUM MULTI-PAGE UI
# ============================================================

BUS_IMAGE_PATH = "https://fragile-peach-yqgzvonxnp-d1lqhzjw83.edgeone.dev/school_bus.jpg"

# Place the correct logo image in the same folder as the code and name it logo.png
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MINISTRY_LOGO_PATH = os.path.join(
    BASE_DIR,
    "logo.png"
)

def image_to_data_uri(path):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = path.split(".")[-1].lower()
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{data}"
    except Exception:
        return path

def ministry_logo_img():
    src = image_to_data_uri(MINISTRY_LOGO_PATH)
    return f"<img class='ministry-logo-img' src='{src}' alt='Ministry Logo'>"


def show_page(which):
    return [
        gr.update(visible=(which == 1)),
        gr.update(visible=(which == 2)),
        gr.update(visible=(which == 3)),
        gr.update(visible=(which == 4)),
        gr.update(visible=(which == 5)),
    ]


def premium_nav(active_step=1):
    items = [(1, "الرئيسية", "home"), (2, "رفع البيانات", "file"), (3, "تجهيز الطلاب", "group"), (4, "المسارات", "route"), (5, "الإشعارات", "bell")]
    links = ""
    for num, label, icon in items:
        cls = "top-link active" if num == active_step else "top-link"
        links += f"""<div class='{cls}'><span class='top-icon'>{step_icon_svg(icon)}</span><b>{label}</b></div>"""
    return f"""
    <div class='site-navbar'>
        <div class='ministry-brand'>
            {ministry_logo_img()}
            <div><div class='ministry-name'>هيئة العامة للنقل</div><div class='ministry-sub'>المملكة العربية السعودية</div></div>
        </div>
        <div class='top-links'>{links}</div>
        <div class='admin-mini'><div class='bell'><span>3</span></div><div class='avatar'></div><div><b>مرحباً، المدير</b><small>مدير النظام</small></div></div>
    </div>
    """


def page_header(kicker, title, subtitle):
    return f"""
    <div class='page-head'>
        <div class='kicker'>{kicker}</div>
        <div class='page-title'>{title}</div>
        <div class='page-subtitle'>{subtitle}</div>
    </div>
    """


def formal_bus_svg(color="#62B69A", accent="#F3C84B"):
    return f"""
    <svg class='bus-svg formal-bus-svg' viewBox='0 0 240 150' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>
        <defs>
            <linearGradient id='busBody{color[-2:]}' x1='0' x2='1' y1='0' y2='1'>
                <stop offset='0%' stop-color='{color}'/>
                <stop offset='100%' stop-color='{accent}'/>
            </linearGradient>
        </defs>
        <rect x='38' y='46' width='164' height='62' rx='18' fill='url(#busBody{color[-2:]})'/>
        <rect x='58' y='29' width='104' height='30' rx='12' fill='url(#busBody{color[-2:]})' opacity='.95'/>
        <rect x='56' y='57' width='32' height='27' rx='6' fill='#EAF8F6' opacity='.95'/>
        <rect x='96' y='57' width='32' height='27' rx='6' fill='#EAF8F6' opacity='.95'/>
        <rect x='136' y='57' width='32' height='27' rx='6' fill='#EAF8F6' opacity='.95'/>
        <rect x='174' y='60' width='18' height='29' rx='5' fill='#EAF8F6' opacity='.95'/>
        <rect x='50' y='94' width='135' height='5' rx='3' fill='#1A2A2A' opacity='.20'/>
        <circle cx='75' cy='112' r='15' fill='#24323A'/>
        <circle cx='75' cy='112' r='7' fill='#EEF3F2'/>
        <circle cx='166' cy='112' r='15' fill='#24323A'/>
        <circle cx='166' cy='112' r='7' fill='#EEF3F2'/>
        <rect x='31' y='72' width='9' height='20' rx='4' fill='#263238' opacity='.55'/>
        <rect x='202' y='72' width='9' height='20' rx='4' fill='#263238' opacity='.55'/>
        <rect x='49' y='44' width='76' height='5' rx='3' fill='#FFFFFF' opacity='.72'/>
    </svg>"""


def step_icon_svg(kind):
    icons = {
        'upload': '<svg viewBox="0 0 24 24"><path d="M12 16V6m0 0 4 4m-4-4-4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>',
        'group': '<svg viewBox="0 0 24 24"><circle cx="8" cy="8" r="3"/><circle cx="16" cy="8" r="3"/><path d="M3 20c.7-3 2.6-5 5-5s4.3 2 5 5"/><path d="M11 20c.6-2.4 2.2-4 5-4 2.3 0 4 1.4 5 4"/></svg>',
        'bus': '<svg viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="11" rx="3"/><path d="M7 16v2m10-2v2"/><path d="M7 9h10"/><path d="M8 13h.01M16 13h.01"/></svg>',
        'route': '<svg viewBox="0 0 24 24"><circle cx="6" cy="18" r="2"/><circle cx="18" cy="6" r="2"/><path d="M8 18h3a3 3 0 0 0 0-6h2a3 3 0 0 0 3-3"/></svg>',
        'bell': '<svg viewBox="0 0 24 24"><path d="M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"/><path d="M10 21h4"/></svg>',
        'home': '<svg viewBox="0 0 24 24"><path d="m3 11 9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/></svg>',
        'file': '<svg viewBox="0 0 24 24"><path d="M6 2h8l4 4v16H6z"/><path d="M14 2v5h5"/></svg>'
    }
    return icons.get(kind, icons['bus'])

def make_bus_cards_html(summary_df=None):
    if summary_df is None or len(summary_df) == 0:
        return f"""
        <div class='bus-result-empty'><div class='empty-scene'>{formal_bus_svg('#F3C84B', '#62B69A')}</div>
        <b>لم يتم تجهيز الباصات بعد</b><span>ارفعي ملف البيانات ثم اضغطي على زر تجهيز الباصات.</span></div>"""
    colors = [("#78C7B2", "#F3C84B"),("#F3C84B", "#62B69A"),("#82B8E8", "#F3C84B"),("#BCA3E8", "#78C7B2"),("#F4A5AD", "#F3C84B"),("#97D5C2", "#F3C84B")]
    cards = ""; total_students = 0
    for idx, row in summary_df.reset_index(drop=True).iterrows():
        bus_id = int(row.get("Bus / Cluster ID", idx + 1)); students = int(row.get("Students Count", 0)); total_students += students
        color, accent = colors[idx % len(colors)]
        cards += f"""<div class='bus-user-card'><div class='bus-title'>الباص {bus_id}</div>{formal_bus_svg(color, accent)}<div class='bus-count'><b>{students}</b><span>طالب</span></div><div class='bus-status-soft'>جاهز <span>✓</span></div></div>"""
    return f"""<div class='bus-result-shell'><div class='bus-result-head'><div><h3>ملخص الباصات</h3><p>تم تجهيز الباصات. اختاري الباص من الخطوة التالية لعرض الطلاب وإنشاء المسار.</p></div><div class='total-pill'>إجمالي الطلاب: <b>{total_students}</b></div></div><div class='bus-result-grid'>{cards}</div></div>"""


def cute_upload_scene():
    return f"""
    <div class='upload-hero-scene formal-scene'>
        <div class='city-bg'>
            <span class='shape-cloud c1'></span>
            <span class='shape-cloud c2'></span>
            <span class='shape-sun'></span>
            <div class='buildings'></div>
        </div>
        {formal_bus_svg('#62B69A', '#F3C84B')}
    </div>"""


def side_steps_html(active=1):
    steps = [
        (1, "رفع البيانات", "ارفع ملف الطلاب والمواقع", "upload"),
        (2, "تجهيز الطلاب", "تنظيم الطلاب في مجموعات", "group"),
        (3, "الباصات والطلاب", "عرض الباصات وعدد الطلاب", "bus"),
        (4, "المسارات", "إظهار مسارات الباصات", "route"),
        (5, "الإشعارات", "إرسال إشعارات لأولياء الأمور", "bell")
    ]
    html = ""
    for n, title, sub, kind in steps:
        cls = "workflow-step active" if n == active else "workflow-step"
        html += f"""<div class='{cls}'><div class='step-icon'>{step_icon_svg(kind)}</div><div><b>{title}</b><span>{sub}</span></div><em>{n}</em></div>"""
    return f"""<div class='workflow-card'><h3>خطوات النظام</h3>{html}<div class='side-cute-scene formal-side-scene'>{formal_bus_svg('#62B69A', '#F3C84B')}</div></div>"""


PREMIUM_CSS = """
:root {
    --bg0:#F8FFFC;
    --bg1:#EEF9F3;
    --card:#FFFFFF;
    --card2:#F7FCF9;
    --green:#62B69A;
    --green2:#2F8F7B;
    --yellow:#F3C84B;
    --blue:#4CA6A8;
    --mint:#A8D7B8;
    --text:#1A2A2A;
    --muted:#647873;
    --soft:#EEF9F3;
    --line:rgba(47,143,123,.16);
    --shadow:0 22px 70px rgba(42,120,105,.13);
}

.gradio-container {
    background:
        radial-gradient(circle at 12% 8%, rgba(98,182,154,.18) 0, transparent 30%),
        radial-gradient(circle at 86% 14%, rgba(243,200,75,.16) 0, transparent 28%),
        radial-gradient(circle at 45% 90%, rgba(98,182,154,.08) 0, transparent 36%),
        linear-gradient(135deg, #FFFFFF 0%, #F8FFFC 48%, #EEF9F3 100%) !important;
    color: var(--text) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

#premium-shell {
    max-width: 1240px;
    margin: 0 auto;
    padding: 8px 8px 24px;
}

footer {display:none !important;}

.premium-navbar {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:22px;
    margin: 8px 0 20px;
    padding:16px 18px;
    border:1px solid var(--line);
    border-radius:26px;
    background:rgba(255,255,255,.78);
    backdrop-filter: blur(18px);
    box-shadow: var(--shadow);
}

.brand-block {display:flex; align-items:center; gap:12px;}
.brand-dot {
    width:42px; height:42px; border-radius:15px;
    background: linear-gradient(135deg, var(--green), var(--yellow));
    box-shadow:0 12px 30px rgba(98,182,154,.28);
    position:relative;
}
.brand-dot:after {content:''; position:absolute; inset:0; display:grid; place-items:center; font-size:20px;}
.brand-name {font-size:20px; font-weight:950; letter-spacing:-.04em; color:var(--text);}
.brand-caption {font-size:12px; color:var(--muted); margin-top:2px;}
.nav-steps {display:flex; gap:10px; flex-wrap:wrap;}
.nav-step {
    display:flex; align-items:center; gap:8px;
    padding:8px 12px; border-radius:999px;
    border:1px solid rgba(47,143,123,.12);
    background:rgba(47,143,123,.045);
    color:#7A718A;
}
.nav-step span {
    width:23px; height:23px; border-radius:50%; display:grid; place-items:center;
    background:white; color:#2F8F7B; font-size:12px; font-weight:900;
    border:1px solid rgba(47,143,123,.14);
}
.nav-step p {margin:0; font-size:12px; font-weight:850;}
.nav-step.active {
    background:linear-gradient(135deg, rgba(98,182,154,.16), rgba(243,200,75,.16));
    color:var(--text);
    border-color:rgba(98,182,154,.30);
    box-shadow:0 12px 28px rgba(98,182,154,.10);
}
.nav-step.active span {
    color:white;
    border:0;
    background:linear-gradient(135deg, var(--green), var(--yellow));
}

.hero-premium, .glass-card, .side-card, .mini-card {
    border:1px solid var(--line) !important;
    background:rgba(255,255,255,.82) !important;
    backdrop-filter: blur(18px);
    border-radius:32px !important;
    box-shadow: var(--shadow) !important;
}
.hero-premium {padding:26px !important; overflow:hidden;}
.glass-card {padding:22px !important;}

.hero-grid {
    display:grid;
    grid-template-columns: 1.05fr .95fr;
    gap:26px;
    align-items:stretch;
}
.hero-kicker, .kicker {
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:8px 12px;
    border-radius:999px;
    background:rgba(98,182,154,.10);
    border:1px solid rgba(98,182,154,.18);
    color:#2F8F7B;
    font-size:12px;
    font-weight:950;
    letter-spacing:.10em;
    text-transform:uppercase;
}
.hero-title {
    margin-top:24px;
    font-size:62px;
    line-height:.94;
    font-weight:1000;
    letter-spacing:-.075em;
    color:var(--text);
}
.hero-title span {
    background:linear-gradient(135deg, #2F8F7B, #F3C84B);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
}
.hero-text {
    margin-top:22px;
    max-width:620px;
    color:var(--muted);
    line-height:1.75;
    font-size:16px;
}
.hero-actions {margin-top:24px;}
.fake-button-note {
    background:linear-gradient(135deg, #62B69A, #2F8F7B 55%, #F3C84B);
    color:white; border:0; border-radius:18px;
    padding:13px 20px; font-weight:950;
    box-shadow:0 18px 34px rgba(47,143,123,.20);
}
.feature-row {
    margin-top:28px;
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:12px;
}
.feature-pill {
    padding:15px; border-radius:22px; border:1px solid rgba(47,143,123,.12);
    background:linear-gradient(180deg, rgba(98,182,154,.055), rgba(255,255,255,.70));
}
.feature-pill b {display:block; color:var(--text); font-size:14px; margin-bottom:5px;}
.feature-pill small {color:var(--muted); font-size:12px; line-height:1.45;}

.visual-panel {
    position:relative;
    min-height:430px;
    border-radius:30px;
    background:
        radial-gradient(circle at 70% 26%, rgba(98,182,154,.22), transparent 34%),
        linear-gradient(135deg, rgba(139,92,246,.12), rgba(243,200,75,.08)),
        #FFFFFF;
    border:1px solid rgba(47,143,123,.14);
    overflow:hidden;
}
.visual-orb {position:absolute; width:85px !important; height:260px; border-radius:50%; background:rgba(98,182,154,.18); filter:blur(2px); right:22px; top:48px;}
.visual-image {
    position:absolute; inset:38px 24px 24px 24px;
    border-radius:26px; overflow:hidden;
    border:1px solid rgba(47,143,123,.16);
    box-shadow:0 30px 70px rgba(42,120,105,.16);
    background:#fff;
}
.visual-image img {width:100%; height:100%; object-fit:cover; filter:saturate(1.06) contrast(1.02);}
.floating-stat {
    position:absolute; left:22px; bottom:24px; right:22px;
    display:grid; grid-template-columns:repeat(3,1fr); gap:10px;
}
.floating-stat div {
    padding:13px 10px; border-radius:18px;
    background:rgba(255,255,255,.86); border:1px solid rgba(47,143,123,.13);
    box-shadow:0 10px 28px rgba(42,120,105,.10);
}
.floating-stat strong {display:block; color:#2F8F7B; font-size:18px;}
.floating-stat small {color:var(--muted); font-size:11px;}

.page-head {margin:8px 0 18px;}
.page-title {margin-top:12px; font-size:38px; font-weight:1000; letter-spacing:-.06em; color:var(--text);}
.page-subtitle {margin-top:8px; max-width:760px; color:var(--muted); line-height:1.65; font-size:14px;}

.card-title {font-size:22px; font-weight:950; letter-spacing:-.035em; color:var(--text); margin-bottom:5px;}
.card-subtitle {font-size:13px; color:var(--muted); line-height:1.6; margin-bottom:16px;}

.status-strip {
    display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:14px 0 18px;
}
.stat-box {
    padding:16px; border-radius:22px;
    background:linear-gradient(180deg, rgba(139,92,246,.07), rgba(255,255,255,.75));
    border:1px solid rgba(47,143,123,.12);
}
.stat-box .label {color:#7A718A; font-size:11px; font-weight:950; text-transform:uppercase; letter-spacing:.08em;}
.stat-box .value {color:var(--text); font-size:22px; font-weight:1000; margin-top:6px;}

button.primary-btn, .primary-btn button {
    background:linear-gradient(135deg, #62B69A, #2F8F7B 55%, #F3C84B) !important;
    color:white !important;
    border:none !important;
    border-radius:16px !important;
    font-weight:950 !important;
    padding:12px 18px !important;
    box-shadow:0 18px 34px rgba(47,143,123,.18) !important;
}
button.secondary-btn, .secondary-btn button {
    background:#FFFFFF !important;
    color:#2F8F7B !important;
    border:1px solid rgba(47,143,123,.18) !important;
    border-radius:16px !important;
    font-weight:900 !important;
    box-shadow:0 10px 22px rgba(42,120,105,.08) !important;
}
.gr-button {border-radius:16px !important; font-weight:900 !important;}

/* Gradio blocks */
.block, .form, .wrap, input, textarea, select {
    border-radius:18px !important;
}
input, textarea, select {
    background:#FFFFFF !important;
    color:var(--text) !important;
    border-color:rgba(47,143,123,.18) !important;
}
label, .label-wrap, .block-title, .markdown {color:var(--text) !important;}
.dataframe, table {border-radius:18px !important; overflow:hidden !important;}
.map-shell iframe, .map-shell > div {border-radius:28px !important; overflow:hidden !important; border:1px solid rgba(47,143,123,.15) !important; min-height:560px;}
.route-text-box {min-height:260px;}

/* Tabs */
.tab-nav button {
    color:var(--text) !important;
    border-radius:16px !important;
}


.cute-bus-row {
    display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:14px 0 10px;
}
.cute-bus-card {
    flex:1; min-width:130px; padding:12px 10px; border-radius:22px;
    background:linear-gradient(135deg, rgba(98,182,154,.10), rgba(243,200,75,.16));
    border:1px solid rgba(47,143,123,.12);
    text-align:center;
}
.cute-bus {font-size:34px; display:block; margin-bottom:4px; filter: drop-shadow(0 8px 10px rgba(47,143,123,.12));}
.cute-bus-card b {display:block; color:#2F8F7B; font-size:18px;}
.cute-bus-card span {color:var(--muted); font-size:12px;}
.friendly-route-card {direction:rtl; text-align:right; padding:4px 0;}
.friendly-route-card h3 {font-size:23px; margin:10px 0 16px; color:var(--text);}
.route-badge {display:inline-flex; padding:8px 12px; border-radius:999px; background:rgba(98,182,154,.12); color:#2F8F7B; font-weight:900;}
.friendly-metrics {display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:10px 0 14px;}
.friendly-metrics div {background:#fff; border:1px solid rgba(47,143,123,.14); border-radius:18px; padding:12px 8px; text-align:center;}
.friendly-metrics b {display:block; font-size:22px; color:#2F8F7B;}
.friendly-metrics span {font-size:12px; color:var(--muted);}
.route-status {border-radius:16px; background:linear-gradient(135deg, rgba(98,182,154,.10), rgba(243,200,75,.12)); border:1px solid rgba(47,143,123,.13); padding:12px; color:#2F8F7B; font-weight:800;}


.bus-result-empty {
    min-height:240px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    gap:8px;
    border:1px dashed rgba(47,143,123,.22);
    background:linear-gradient(135deg, rgba(98,182,154,.08), rgba(243,200,75,.10));
    border-radius:28px;
    text-align:center;
    color:var(--muted);
}
.bus-result-empty b {font-size:22px; color:var(--text);}
.bus-result-empty span {font-size:14px;}
.empty-bus {font-size:62px; filter:drop-shadow(0 12px 16px rgba(47,143,123,.16));}
.bus-result-shell {direction:rtl; text-align:right;}
.bus-result-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:14px;
    padding:14px 16px;
    margin-bottom:14px;
    border-radius:24px;
    background:linear-gradient(135deg, rgba(98,182,154,.10), rgba(243,200,75,.13));
    border:1px solid rgba(47,143,123,.13);
}
.bus-result-head h3 {margin:0 0 6px; font-size:24px; color:var(--text);}
.bus-result-head p {margin:0; color:var(--muted); font-size:13px;}
.total-pill {
    white-space:nowrap;
    border-radius:999px;
    padding:10px 14px;
    background:#fff;
    border:1px solid rgba(47,143,123,.14);
    color:#2F8F7B;
    font-weight:900;
}
.bus-result-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit, minmax(165px, 1fr));
    gap:14px;
}
.bus-user-card {
    position:relative;
    overflow:hidden;
    min-height:150px;
    padding:16px;
    border-radius:28px;
    background:#fff;
    border:1px solid rgba(47,143,123,.14);
    box-shadow:0 16px 34px rgba(47,143,123,.09);
}
.bus-user-card:before {
    content:'';
    position:absolute;
    width:120px; height:120px;
    border-radius:50%;
    left:-34px; bottom:-42px;
    background:rgba(243,200,75,.22);
}
.bus-user-card.green {background:linear-gradient(135deg, #FFFFFF, rgba(98,182,154,.14));}
.bus-user-card.yellow {background:linear-gradient(135deg, #FFFFFF, rgba(243,200,75,.20));}
.bus-user-card.mint {background:linear-gradient(135deg, #FFFFFF, rgba(191,221,142,.18));}
.bus-user-card.blue {background:linear-gradient(135deg, #FFFFFF, rgba(126,190,204,.14));}
.bus-user-card.peach {background:linear-gradient(135deg, #FFFFFF, rgba(255,199,132,.18));}
.bus-user-card.violet {background:linear-gradient(135deg, #FFFFFF, rgba(170,150,220,.13));}
.bus-top {display:flex; justify-content:space-between; align-items:center; position:relative; z-index:1;}
.bus-emoji {font-size:40px; filter:drop-shadow(0 10px 12px rgba(47,143,123,.15));}
.bus-status-dot {
    padding:6px 10px;
    border-radius:999px;
    background:rgba(98,182,154,.13);
    color:#2F8F7B;
    font-size:12px;
    font-weight:900;
}
.bus-title {position:relative; z-index:1; margin-top:10px; color:#2F8F7B; font-size:22px; font-weight:1000;}
.bus-count {position:relative; z-index:1; margin-top:8px; display:flex; align-items:baseline; gap:6px;}
.bus-count b {font-size:34px; color:var(--text);}
.bus-count span {font-size:15px; color:var(--muted); font-weight:800;}
.bus-caption {position:relative; z-index:1; margin-top:3px; color:var(--muted); font-size:12px;}
.page-cute-illustration {
    margin:14px 0;
    padding:16px;
    border-radius:26px;
    background:linear-gradient(135deg, rgba(98,182,154,.09), rgba(243,200,75,.14));
    border:1px solid rgba(47,143,123,.12);
    display:flex;
    align-items:center;
    justify-content:center;
    gap:12px;
    color:#2F8F7B;
    font-weight:950;
}
.page-cute-illustration .big {font-size:48px; filter:drop-shadow(0 10px 12px rgba(47,143,123,.14));}


/* ===== Clean ministry cute web style overrides ===== */
.gradio-container {direction:rtl !important;}
#premium-shell {max-width:1420px !important;}
.site-navbar {direction:rtl; display:flex; align-items:center; justify-content:space-between; gap:24px; margin:0 0 22px; padding:18px 28px; background:rgba(255,255,255,.92); border:1px solid rgba(47,143,123,.12); border-radius:0 0 26px 26px; box-shadow:0 12px 34px rgba(40,98,87,.08); backdrop-filter: blur(16px);}
.ministry-brand {display:flex; align-items:center; gap:8px; min-width:120px;}

.ministry-logo-img {
    width:85px !important;
    max-width:85px !important;
    height:auto !important;
    object-fit:contain !important;
    display:block !important;
}
.mot-logo {width:72px; height:48px; border-radius:8px 20px 8px 8px; transform:skewX(-18deg); background:linear-gradient(135deg,#62B69A 0%,#9BCB85 58%,#F3C84B 100%); position:relative; box-shadow:0 10px 24px rgba(47,143,123,.14);}
.mot-star {transform:skewX(18deg); color:white; font-size:24px; position:absolute; top:8px; right:12px; font-weight:900;}
.ministry-name {font-size:15px; font-weight:950; color:#1A2A2A;}.ministry-sub {font-size:12px; color:#647873; margin-top:3px;}
.top-links {display:flex; gap:28px; justify-content:center; flex:1;}.top-link {display:flex; align-items:center; gap:8px; padding:12px 2px; color:#53656a; border-bottom:2px solid transparent; font-size:14px;}.top-link b {font-weight:850;}.top-link.active {color:#2F8F7B; border-color:#62B69A;}
.admin-mini {display:flex; align-items:center; gap:12px; min-width:240px; justify-content:flex-end; direction:rtl;}.admin-mini .avatar {width:44px;height:44px;border-radius:50%;display:grid;place-items:center;background:#FFF4DC;border:1px solid rgba(243,200,75,.35);}.admin-mini b {display:block;font-size:13px;color:#1A2A2A;}.admin-mini small {display:block;font-size:11px;color:#647873;margin-top:2px;}.bell {position:relative;width:38px;height:38px;border-radius:50%;display:grid;place-items:center;background:#fff;border:1px solid rgba(47,143,123,.12);}.bell span {position:absolute;top:-5px;left:-5px;width:20px;height:20px;border-radius:50%;background:#62B69A;color:white;font-size:11px;display:grid;place-items:center;font-weight:900;}
.upload-page-grid {display:grid !important; grid-template-columns:minmax(0,1fr) 320px; gap:22px; align-items:start; direction:ltr;}.upload-main {direction:rtl;}.upload-card-grid {display:grid !important; grid-template-columns:1fr 1.25fr; gap:26px; align-items:center;}
.upload-hero-scene {min-height:315px; border-radius:28px; position:relative; overflow:hidden; background:linear-gradient(180deg,#F8FFFC,#EEF9F3); display:grid; place-items:end center; padding-bottom:20px; border:1px solid rgba(47,143,123,.08);}.upload-hero-scene .bus-svg {width:285px; height:auto; z-index:2; filter:drop-shadow(0 18px 18px rgba(47,143,123,.16));}.city-bg {position:absolute; inset:0; opacity:.95;}.city-bg .sun {position:absolute; top:45px; right:140px; font-size:32px;}.cloud {position:absolute; font-size:35px; color:#CDEBE2; opacity:.85;}.c1 {top:56px; left:80px}.c2 {top:105px; right:52px}.buildings {position:absolute; bottom:64px; left:26px; right:26px; height:118px; opacity:.65; background:linear-gradient(to top, rgba(98,182,154,.20), rgba(98,182,154,.03)); clip-path:polygon(0 100%,0 55%,8% 55%,8% 35%,16% 35%,16% 62%,23% 62%,23% 25%,32% 25%,32% 72%,40% 72%,40% 42%,49% 42%,49% 100%,59% 100%,59% 52%,67% 52%,67% 30%,77% 30%,77% 68%,86% 68%,86% 45%,100% 45%,100% 100%);}
.upload-box {padding:28px; border-radius:26px; border:1px dashed rgba(94,111,128,.25); background:rgba(255,255,255,.70); text-align:center;}.requirements-box {margin-top:14px; padding:16px; border-radius:18px; background:linear-gradient(135deg, rgba(98,182,154,.08), rgba(243,200,75,.08)); border:1px solid rgba(47,143,123,.15); color:#2F8F7B; font-weight:850; font-size:13px;}
.workflow-card {background:#fff; border:1px solid rgba(47,143,123,.14); padding:18px; direction:rtl; border-radius:22px; box-shadow:0 14px 36px rgba(42,120,105,.09);}.workflow-card h3 {margin:0 0 18px; text-align:center; font-size:20px;color:#1A2A2A;}.workflow-step {display:grid; grid-template-columns:46px 1fr 30px; align-items:center; gap:12px; padding:16px; margin-bottom:12px; border-radius:18px; border:1px solid rgba(94,111,128,.12); background:#fff; box-shadow:0 8px 20px rgba(20,60,54,.04);}.workflow-step.active {border-color:rgba(98,182,154,.45); background:linear-gradient(135deg,rgba(98,182,154,.10),rgba(255,255,255,.95)); box-shadow:0 10px 25px rgba(47,143,123,.10);}.step-icon {font-size:25px;}.workflow-step b {display:block;color:#1A2A2A;font-size:15px;}.workflow-step span {display:block;color:#647873;font-size:12px;margin-top:4px;}.workflow-step em {font-style:normal;width:28px;height:28px;border-radius:50%;background:#8390A3;color:white;display:grid;place-items:center;font-weight:900;font-size:13px;}.workflow-step.active em {background:#2F8F7B;}.side-cute-scene {margin-top:20px; min-height:180px; border-radius:20px; background:linear-gradient(180deg,#F8FFFC,#FFF8E7); position:relative; display:grid; place-items:center; overflow:hidden;}.side-cute-scene .bus-svg {width:210px;}.side-cute-scene .kids {position:absolute;bottom:10px;right:30px;font-size:28px;}
.empty-scene .bus-svg {width:230px;}.bus-result-head {background:transparent !important; border:0 !important; padding:0 4px 18px !important;}.bus-result-grid {grid-template-columns:repeat(auto-fit,minmax(190px,1fr)) !important; gap:20px !important;}.bus-user-card {text-align:center !important; min-height:255px !important; padding:20px 16px 18px !important; border-radius:24px !important; box-shadow:0 12px 26px rgba(42,120,105,.08) !important;}.bus-user-card:before {display:none !important;}.bus-user-card .bus-title {font-size:22px !important; margin:0 0 8px !important; color:#2F8F7B !important;}.bus-user-card .bus-svg {width:172px; max-width:100%; margin:0 auto 6px; filter:drop-shadow(0 14px 14px rgba(42,120,105,.12));}.bus-count {justify-content:center !important; margin-top:0 !important;}.bus-count b {font-size:28px !important; color:#1A2A2A !important;}.bus-count span {font-size:17px !important; color:#1A2A2A !important;}.bus-status-soft {margin:14px auto 0; max-width:150px; border-radius:12px; padding:9px; background:rgba(98,182,154,.14); color:#2F8F7B; font-weight:900;}.bus-status-soft span {display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;background:#62B69A;color:white;font-size:11px;margin-right:5px;}


/* ===== Page 4 route layout fix - compact no scroll ===== */

/* Make the summary and map shorter to reduce scrolling */
.route-text-box,
.route-text-box > div,
.route-text-box .friendly-route-card {
    height: 390px !important;
    min-height: 390px !important;
    max-height: 390px !important;
    overflow: hidden !important;
}

.route-text-box {
    display: flex !important;
    flex-direction: column !important;
    padding: 12px 14px !important;
}

.route-text-box .markdown,
.route-text-box .prose,
.route-text-box .md,
.route-text-box .gr-markdown,
.route-text-box .friendly-route-card {
    flex: 1 !important;
}

.friendly-route-card {
    direction: rtl;
    text-align: right;
    padding: 10px 12px !important;
    border-radius: 24px !important;
    background: rgba(255,255,255,.72);
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    overflow: hidden !important;
}

.route-badge {
    padding: 7px 12px !important;
    font-size: 12px !important;
}

.friendly-route-card h3 {
    font-size: 21px !important;
    margin: 10px 0 12px !important;
}

/* Stack the boxes vertically, but keep them smaller */
.friendly-metrics {
    display: grid !important;
    grid-template-columns: 1fr !important;
    gap: 8px !important;
    margin: 8px 0 10px !important;
}

.friendly-metrics div {
    min-height: 48px !important;
    height: 48px !important;
    border-radius: 16px !important;
    padding: 6px 14px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    text-align: right !important;
}

.friendly-metrics b {
    font-size: 24px !important;
}

.friendly-metrics span {
    font-size: 13px !important;
}

.route-status {
    margin-top: 8px !important;
    padding: 12px 14px !important;
    border-radius: 18px !important;
    font-size: 13px !important;
}

.map-shell iframe,
.map-shell > div {
    min-height: 390px !important;
    height: 390px !important;
    max-height: 390px !important;
    overflow: hidden !important;
}

@media (max-width:980px){.site-navbar{flex-direction:column;align-items:stretch;border-radius:22px}.top-links{justify-content:flex-start;overflow:auto;gap:14px}.admin-mini,.ministry-brand{min-width:0}.upload-page-grid,.upload-card-grid{grid-template-columns:1fr !important}}

/* ===== Formal ministry web style overrides ===== */
.formal-bus-svg {filter: drop-shadow(0 14px 18px rgba(42,120,105,.12));}
.shape-cloud {position:absolute; width:86px; height:24px; border-radius:999px; background:#DDEFEA; opacity:.75;}
.shape-cloud:before {content:''; position:absolute; width:34px; height:34px; border-radius:50%; background:#DDEFEA; left:14px; top:-16px;}
.shape-cloud:after {content:''; position:absolute; width:42px; height:42px; border-radius:50%; background:#DDEFEA; right:14px; top:-22px;}
.shape-sun {position:absolute; top:46px; right:135px; width:42px; height:42px; border-radius:50%; background:#F3C84B; box-shadow:0 0 0 12px rgba(243,200,75,.12);}
.top-icon {display:grid; place-items:center;}
.top-icon svg, .step-icon svg {width:25px; height:25px; stroke:#5E6F80; stroke-width:1.8; fill:none; stroke-linecap:round; stroke-linejoin:round;}
.top-link.active .top-icon svg, .workflow-step.active .step-icon svg {stroke:#2F8F7B;}
.workflow-step {grid-template-columns:46px 1fr 30px !important;}
.step-icon {width:38px; height:38px; display:grid; place-items:center;}
.side-cute-scene:after {content:'وزارة النقل والخدمات اللوجستية'; position:absolute; bottom:14px; left:0; right:0; text-align:center; color:#647873; font-size:12px; font-weight:700;}
.formal-side-scene .bus-svg {width:220px !important; transform:translateY(-12px);}
.bus-user-card {background:#fff !important;}
.bus-user-card .bus-svg {width:178px !important;}
.bus-status-soft {background:#EEF8F4 !important; color:#2F8F7B !important;}
.hero-kicker, .kicker {letter-spacing:0 !important; text-transform:none !important;}
.hero-title {font-weight:900 !important;}
.empty-scene .bus-svg {width:230px;}
.upload-hero-scene .bus-svg {width:300px !important;}
.mot-star {width:20px; height:20px; border-radius:6px; background:rgba(255,255,255,.75); transform:rotate(45deg);}
.admin-mini .bell:before {content:''; width:18px; height:18px; border:2px solid #5E6F80; border-radius:50% 50% 45% 45%; display:block;}
.admin-mini .avatar {background:linear-gradient(135deg,#E8F4EF,#F7E8BE) !important; border:1px solid rgba(47,143,123,.14);}

@media (max-width: 980px) {
    .premium-navbar {flex-direction:column; align-items:flex-start;}
    .hero-grid {grid-template-columns:1fr;}
    .hero-title {font-size:42px;}
    .feature-row, .status-strip, .floating-stat {grid-template-columns:1fr;}
}

/* ===== Hide route summary scrollbar ===== */
.route-text-box,
.route-text-box > div,
.route-text-box .wrap,
.route-text-box .block,
.route-text-box .markdown,
.route-text-box .prose,
.route-text-box .gr-markdown,
.route-text-box .friendly-route-card {
    overflow: hidden !important;
    scrollbar-width: none !important;
}

.route-text-box::-webkit-scrollbar,
.route-text-box *::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
}

#page4,
#page4 * {
    scrollbar-width: none !important;
}

#page4::-webkit-scrollbar,
#page4 *::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
}

"""


premium_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
).set(
    body_background_fill="#F8FFFC",
    body_text_color="#171321",
    block_background_fill="#FFFFFF",
    block_border_color="rgba(47,143,123,.14)",
    block_border_width="1px",
    block_shadow="0 22px 70px rgba(76,29,149,.13)",
    button_primary_background_fill="#62B69A",
    button_primary_background_fill_hover="#2F8F7B",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_text_color="#2F8F7B",
    input_background_fill="#FFFFFF",
    input_border_color="rgba(47,143,123,.20)",
)


with gr.Blocks(theme=premium_theme, css=PREMIUM_CSS, title="SmartBus") as demo:

    df_full_state     = gr.State()
    df_students_state = gr.State()
    df_eta_state      = gr.State()

    with gr.Column(elem_id="premium-shell"):

        # PAGE 1 — WELCOME
        with gr.Group(visible=True) as page1:
            gr.HTML(premium_nav(active_step=1))
            with gr.Group(elem_classes=["hero-premium"]):
                gr.HTML(f"""
                <div class='hero-grid'>
                    <div>
                        <div class='hero-kicker'>منصة النقل المدرسي</div>
                        <div class='hero-title'>نظّم رحلات <span>الباص المدرسي</span> بسهولة.</div>
                        <div class='hero-text'>
                            منصة سهلة تساعدك على رفع بيانات الطلاب، توزيعهم على الباصات، عرض المسارات على الخريطة، وإرسال إشعارات الوصول لأولياء الأمور.
                        </div>
                        <div class='feature-row'>
                            <div class='feature-pill'><b>توزيع الطلاب</b><small>تنظيم الطلاب في مجموعات مناسبة لكل باص.</small></div>
                            <div class='feature-pill'><b>تخطيط المسار</b><small>ترتيب المحطات بطريقة عملية وسهلة.</small></div>
                            <div class='feature-pill'><b>خريطة واقعية</b><small>عرض المسار على الطرق الفعلية.</small></div>
                        </div>
                    </div>
                    <div class='visual-panel'>
                        <div class='visual-orb'></div>
                        <div class='visual-image'><img src='{BUS_IMAGE_PATH}'></div>
                    </div>
                </div>
                """)
                btn_start = gr.Button("ابدأ الآن →", elem_classes=["primary-btn"])

        # PAGE 2 — UPLOAD + PREPARE BUSES
        with gr.Group(visible=False) as page2:
            gr.HTML(premium_nav(active_step=2))
            with gr.Row(elem_classes=["upload-page-grid"]):
                with gr.Column(elem_classes=["upload-main"]):
                    with gr.Group(elem_classes=["glass-card"]):
                        with gr.Row(elem_classes=["upload-card-grid"]):
                            with gr.Column():
                                gr.HTML(cute_upload_scene())
                            with gr.Column():
                                gr.HTML("""
                                    <div class='page-title'>رفع البيانات</div>
                                    <div class='page-subtitle'>قم برفع ملف بيانات الطلاب والمواقع لبدء عملية تجهيز الباصات.</div>
                                """)
                                merged_file = gr.File(label="اختر ملف البيانات", file_types=[".csv"])
                                with gr.Row():
                                    btn_back_to_page1 = gr.Button("← رجوع", elem_classes=["secondary-btn"])
                                    btn_run_clustering = gr.Button("تجهيز الباصات", elem_classes=["primary-btn"])
                                clustering_status = gr.Markdown()
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.HTML("""
                            <div class='card-title'>ملخص الباصات</div>
                            <div class='card-subtitle'>بعد تجهيز البيانات، سيظهر كل باص مع عدد الطلاب بطريقة مبسطة ولطيفة للمستخدم.</div>
                        """)
                        bus_cards_html = gr.HTML(make_bus_cards_html())
                        summary_table = gr.Dataframe(label="ملخص الباصات", wrap=True, visible=False)
                        rank_table = gr.Dataframe(label="جدول داخلي", visible=False)
                        btn_to_page3 = gr.Button("التالي: الحضور والغياب →", elem_classes=["primary-btn"])
                with gr.Column():
                    gr.HTML(side_steps_html(active=1))

        # PAGE 3 — BUS + ATTENDANCE
        with gr.Group(visible=False) as page3:
            gr.HTML(premium_nav(active_step=3))
            gr.HTML(page_header("الخطوة ٢", "اختيار الباص والحضور", "اختَر الباص وحدد الطلاب الحاضرين اليوم، ثم انتقل لعرض المسار."))
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.HTML("""
                            <div class='card-title'>اختيار الباص</div>
                            <div class='card-subtitle'>اختَر الباص المطلوب.</div>
                            <div class='page-cute-illustration'><span class='big'>🚍</span><span>اختاري الباص ثم حددي حضور الطلاب</span></div>
                        """)
                        bus_id_dropdown = gr.Dropdown(label="رقم الباص", choices=[], value=None)
                        with gr.Row():
                            btn_back_to_page2 = gr.Button("← Back", elem_classes=["secondary-btn"])
                            btn_to_page4 = gr.Button("عرض المسار →", elem_classes=["primary-btn"])
                with gr.Column(scale=8):
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.HTML("""
                            <div class='card-title'>حضور الطلاب</div>
                            <div class='card-subtitle'>أزل علامة الصح من الطلاب الغائبين قبل إنشاء المسار.</div>
                            <div class='page-cute-illustration'><span class='big'></span><span>قائمة الطلاب لهذا الباص</span></div>
                        """)
                        students_check = gr.CheckboxGroup(label="الطلاب الحاضرون", choices=[], value=[])

        # PAGE 4 — ROUTING
        with gr.Group(visible=False) as page4:
            gr.HTML(premium_nav(active_step=4))
            gr.HTML(page_header("الخطوة ٣", "عرض مسارات الباصات", "سيظهر المسار على الخريطة بشكل واضح وواقعي للمستخدم."))
            with gr.Group(elem_classes=["glass-card"]):
                with gr.Row():
                    btn_back_to_page3 = gr.Button("← Back", elem_classes=["secondary-btn"])
                    btn_rerun_route = gr.Button("إنشاء / تحديث المسار", elem_classes=["primary-btn"])
                    btn_to_page5 = gr.Button("الإشعارات →", elem_classes=["primary-btn"])
            with gr.Tabs():
                with gr.Tab(" الذهاب للمدرسة"):
                    with gr.Row():
                        with gr.Column(scale=8):
                            morning_plot = gr.HTML(elem_classes=["map-shell"])
                        with gr.Column(scale=4):
                            with gr.Group(elem_classes=["glass-card", "route-text-box"]):
                                gr.HTML("<div class='card-title'>ملخص المسار الصباحي</div><div class='card-subtitle'>عرض مبسط للمستخدم بدون تفاصيل تقنية.</div>")
                                morning_text = gr.Markdown()
                with gr.Tab(" العودة للمنزل"):
                    with gr.Row():
                        with gr.Column(scale=8):
                            afternoon_plot = gr.HTML(elem_classes=["map-shell"])
                        with gr.Column(scale=4):
                            with gr.Group(elem_classes=["glass-card", "route-text-box"]):
                                gr.HTML("<div class='card-title'>ملخص مسار العودة</div><div class='card-subtitle'>عرض مبسط للمستخدم بدون تفاصيل تقنية.</div>")
                                afternoon_text = gr.Markdown()

        # PAGE 5 — NOTIFICATIONS
        with gr.Group(visible=False) as page5:
            gr.HTML(premium_nav(active_step=5))
            gr.HTML(page_header("الخطوة ٤", "إرسال إشعارات الوصول", "اختر نوع الرحلة ثم أرسل إشعارات الوصول لأولياء الأمور."))
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.HTML("""
                            <div class='card-title'>مركز الإشعارات</div>
                            <div class='card-subtitle'>اختر نوع الرحلة قبل الإرسال.</div>
                        """)
                        trip_selector = gr.Radio(["morning", "afternoon"], label="نوع الرحلة")
                        with gr.Row():
                            btn_back_to_page4 = gr.Button("← Back to routing", elem_classes=["secondary-btn"])
                            btn_send_notifications = gr.Button("إرسال الإشعارات", elem_classes=["primary-btn"])
                with gr.Column(scale=7):
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.HTML("<div class='card-title'>سجل الإرسال</div><div class='card-subtitle'>ستظهر نتائج الإرسال هنا.</div>")
                        notify_output = gr.Markdown()

        # EVENT WIRING
        debug_text = gr.Markdown("", visible=False)
        EMPTY_DF   = pd.DataFrame()

        def on_run_clustering(file):
            try:
                df_full, df_students, rank_df, msg, summary_df = run_clustering(file)
                df_students["cluster_ai"] = df_students["cluster_ai"].astype(int)
                bus_ids = [str(int(x)) for x in sorted(df_students["cluster_ai"].unique())]

                dbg = "OK  " + msg
                return (
                    df_full,
                    df_students,
                    rank_df if rank_df is not None else EMPTY_DF,
                    msg,
                    summary_df if summary_df is not None else EMPTY_DF,
                    make_bus_cards_html(summary_df),
                    gr.update(choices=bus_ids),
                    dbg
                )
            except Exception as e:
                return (
                    None, None, EMPTY_DF,
                    f" Error: {str(e)}",
                    EMPTY_DF,
                    gr.update(choices=None),
                    f" ERROR\n{str(e)}"
                )

        btn_run_clustering.click(
            fn=on_run_clustering,
            inputs=[merged_file],
            outputs=[
                df_full_state,
                df_students_state,
                rank_table,
                clustering_status,
                summary_table,
                bus_cards_html,
                bus_id_dropdown,
                debug_text
            ]
        )

        bus_id_dropdown.change(
            fn=get_students_for_bus,
            inputs=[bus_id_dropdown, df_students_state],
            outputs=[students_check]
        )

        def on_run_routing(bus_id, selected_students, df_full, df_students):
            return run_routing(bus_id, selected_students, df_full, df_students)

        btn_rerun_route.click(
            fn=on_run_routing,
            inputs=[bus_id_dropdown, students_check, df_full_state, df_students_state],
            outputs=[morning_plot, morning_text, afternoon_plot, afternoon_text, df_eta_state],
            queue=True
        )

        btn_send_notifications.click(
            fn=send_whatsapp_notifications,
            inputs=[df_eta_state, trip_selector],
            outputs=notify_output
        )

        # Forward navigation
        btn_start.click(fn=lambda: show_page(2), outputs=[page1, page2, page3, page4, page5])
        btn_to_page3.click(fn=lambda: show_page(3), outputs=[page1, page2, page3, page4, page5])
        btn_to_page4.click(fn=lambda: show_page(4), outputs=[page1, page2, page3, page4, page5])
        btn_to_page5.click(fn=lambda: show_page(5), outputs=[page1, page2, page3, page4, page5])

        # Backward navigation
        btn_back_to_page1.click(fn=lambda: show_page(1), outputs=[page1, page2, page3, page4, page5])
        btn_back_to_page2.click(fn=lambda: show_page(2), outputs=[page1, page2, page3, page4, page5])
        btn_back_to_page3.click(fn=lambda: show_page(3), outputs=[page1, page2, page3, page4, page5])
        btn_back_to_page4.click(fn=lambda: show_page(4), outputs=[page1, page2, page3, page4, page5])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        show_error=True
    )
