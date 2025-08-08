# dqn_planner.py
import math, random, collections, pickle, os, torch, torch.nn as nn, torch.optim as optim
from planners import Planner
from problems import ResourceType
from datetime import datetime

# --------------------------------------------------------------------- #
#  1. Replay buffer                                                     #
# --------------------------------------------------------------------- #
Transition = collections.namedtuple('Transition',
                                    ('state', 'action', 'reward', 'next_state', 'done', 'mask'))

class ReplayMemory:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.buffer   = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self): return len(self.buffer)

# --------------------------------------------------------------------- #
#  2. Neural network (dueling, mask-aware)                              #
# --------------------------------------------------------------------- #
class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU())
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, action_dim)

    def forward(self, x, mask=None):
        z  = self.backbone(x)
        V  = self.V(z)
        A  = self.A(z)
        if mask is not None:
            A[mask == 0] = -1e9        # forbid illegal moves
        return V + (A - A.mean(dim=1, keepdim=True))

# --------------------------------------------------------------------- #
#  3. The Planner                                                       #
# --------------------------------------------------------------------- #
class DQNPlanner(Planner):
    """
    A Planner that can *train* (off-line) and later *act* (inference-only)
    using the same definition.  Switch by `train=True/False`.
    
    Enhanced with per-diagnosis queues and structured state representation.
    """
    ## hyper-parameters
    GAMMA      = 0.99
    LR         = 1e-4
    BATCH      = 256
    TARGET_UPD = 2_000
    EPS_START  = 1.0
    EPS_END    = 0.05
    EPS_DECAY  = 400_000     # steps

    def __init__(self,
                 model_path="weights.pt",
                 train=False,
                 max_queue_len=50):
        super().__init__()
        self.model_path     = model_path
        self.train_mode     = train
        self.max_queue_len  = max_queue_len   # cut tail to keep state small
        self.step_counter   = 0

        # ---- bookkeeping for KPIs -------------------------------------
        self.wait_intake     = collections.defaultdict(float)  # hrs waited
        self.wait_in_hosp    = collections.defaultdict(float)
        self.plan_times      = {}                              # case_id -> tp
        self.replan_penalty  = 0
        self.cost            = 0
        self.last_ts         = 0

        # ---- diagnosis tracking and per-diagnosis queues --------------
        self.diagnosis_dict = {}  # case_id -> diagnosis
        self.arrival_times = {}   # case_id -> arrival_time
        self.diagnosis_encoding = {
            'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3,
            'B1': 4, 'B2': 5, 'B3': 6, 'B4': 7,
            'No diagnosis': 8
        }
        
        # Per-diagnosis queues ordered by arrival time (FIFO)
        self.diagnosis_queues = {
            'A1': collections.deque(), 'A2': collections.deque(), 
            'A3': collections.deque(), 'A4': collections.deque(),
            'B1': collections.deque(), 'B2': collections.deque(), 
            'B3': collections.deque(), 'B4': collections.deque(),
            'No diagnosis': collections.deque()
        }
        
        # Configuration for state slots per diagnosis
        # Based on typical distribution: A1(30%), B1(25%), B2(15%), others(30%)
        self.slots_per_diagnosis = {
            'A1': 3, 'A2': 1, 'A3': 1, 'A4': 1,
            'B1': 2, 'B2': 1, 'B3': 1, 'B4': 1,
            'No diagnosis': 1
        }
        
        # Total slots for state representation
        self.total_slots = sum(self.slots_per_diagnosis.values())  # 12 slots

        # ---- delayed reward tracking ----------------------------------
        self.pending_actions = []  # [(timestamp, state, action, mask, case_id, action_type)]
        self.action_outcomes = {}  # case_id -> outcome tracking
        self.simulation_start_time = 0  # Track simulation duration for normalization

        # ---- resources -------------------------------------------------
        self.resources_now   = {rt:0 for rt in ResourceType}   # current plan
        self.future_plan     = collections.defaultdict(dict)   # t -> {rt:n}

        # ---- RL parts --------------------------------------------------
        # New state dimension: slots + diagnosis features + resource info + time features
        self.state_dim = (self.total_slots * 2) + 9 + len(ResourceType) + 9  # slots + diagnosis counts + resources + time
        self.action_dim = (self.total_slots * 24) + 27  # plan actions + schedule actions
        self.net   = Net(self.state_dim, self.action_dim)
        self.tgt   = Net(self.state_dim, self.action_dim)
        self.tgt.load_state_dict(self.net.state_dict())
        self.mem   = ReplayMemory()
        self.opt   = optim.Adam(self.net.parameters(), lr=self.LR)

        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path))
            self.tgt.load_state_dict(self.net.state_dict())
            print("Loaded weights:", model_path)

    # ================================================================ #
    #  REPORT  – bind simulator stream to our private counters         #
    # ================================================================ #
    def report(self, case_id, element, ts, resource, lifecycle, data=None):
        # Track simulation start time for normalization
        if self.simulation_start_time == 0:
            self.simulation_start_time = ts
            
        # Track diagnosis information and arrival times
        if element is not None and element.label in ['emergency_patient', 'patient_referal']:
            if hasattr(element, 'data') and 'diagnosis' in element.data:
                if element.data['diagnosis'] is not None:
                    self.diagnosis_dict[case_id] = element.data['diagnosis']
                else:
                    self.diagnosis_dict[case_id] = 'No diagnosis'
                # Record arrival time
                self.arrival_times[case_id] = ts
                # Add to appropriate diagnosis queue
                diagnosis = self.diagnosis_dict[case_id]
                self._add_case_to_queue(case_id, diagnosis, ts)
        
        # elapsed since last event to accumulate waiting / cost
        dt = ts - self.last_ts
        for c in self.wait_intake:  self.wait_intake[c] += dt
        for c in self.wait_in_hosp: self.wait_in_hosp[c] += dt
        self.last_ts = ts

        # track phase changes
        from problems import HealthcareElements
        if element:
            if element.label == HealthcareElements.TIME_FOR_INTAKE:
                self.wait_intake[case_id] = 0.0
                self.plan_times[(case_id, ts)] = element.occurrence_time
            elif element.label == HealthcareElements.INTAKE and lifecycle.name.endswith("COMPLETE"):
                self.wait_in_hosp[case_id] = 0.0
                self.wait_intake.pop(case_id, None)
            elif element.label in (HealthcareElements.SURGERY,
                                   HealthcareElements.NURSING) and lifecycle.name.endswith("COMPLETE"):
                self.wait_in_hosp.pop(case_id, None)

        # add nervousness penalty when replanning
        if lifecycle.name == "PLAN_UPDATED":   # fictitious name – adapt!
            old_tp, tr = data['old_tp'], ts
            self.replan_penalty += max(0, 336 - (old_tp - tr)) / 336

        # resource cost each SCHEDULE_RESOURCES
        if lifecycle.name == "SCHEDULE_RESOURCES":
            if data and 'cost' in data:
                self.cost += data['cost']  # simulator gives it (see simulator.py)
                # Assign delayed rewards for scheduling actions
                self._assign_delayed_rewards(ts, 'schedule', data.get('cost', 0))
        
        # Assign delayed rewards for planning actions when cases complete
        if element and lifecycle.name.endswith("COMPLETE"):
            if element.label in [HealthcareElements.INTAKE, HealthcareElements.SURGERY, 
                               HealthcareElements.NURSING, HealthcareElements.RELEASING]:
                self._assign_delayed_rewards(ts, 'plan', case_id)
                
        # Track hospital waiting time for completed tasks
        if element and lifecycle.name.endswith("COMPLETE") and element.label in [HealthcareElements.SURGERY, HealthcareElements.NURSING]:
            if case_id in self.wait_in_hosp:
                hospital_wait_time = self.wait_in_hosp[case_id]
                self._assign_delayed_rewards(ts, 'hospital_wait', hospital_wait_time)
                
        # Track nervousness (replanning penalty)
        if lifecycle.name == "PLAN_UPDATED" and data and 'old_tp' in data:
            old_tp, tr = data['old_tp'], ts
            replan_penalty = max(0, 336 - (old_tp - tr)) / 336
            self._assign_delayed_rewards(ts, 'nervousness', replan_penalty)

    # ================================================================ #
    #  PLAN  – always called at arbitrary sim events                   #
    # ================================================================ #
    def plan(self, to_plan, to_replan, now):
        # Update per-diagnosis queues with new cases
        self._update_diagnosis_queues(to_plan, to_replan, now)
        
        # Build state from structured queues
        state, mask, plan_actions, sched_actions = self._build_structured_state(now)
        action = self._select_action(state, mask)
        
        #  if the chosen action is a scheduling one, ignore here
        if action >= len(plan_actions):
            # Track scheduling action for delayed reward
            self.pending_actions.append((now, state, action, mask, None, 'schedule'))
            return []
        
        case_id, adm_time = plan_actions[action]
        self.plan_times[case_id] = adm_time
        self.wait_intake[case_id] = 0.0  # start counting
        
        # Track planning action for delayed reward
        self.pending_actions.append((now, state, action, mask, case_id, 'plan'))
        
        # Remove case from its diagnosis queue
        diagnosis = self.diagnosis_dict.get(case_id, 'No diagnosis')
        if case_id in self.diagnosis_queues[diagnosis]:
            self.diagnosis_queues[diagnosis].remove(case_id)
        
        return [(case_id, adm_time)]

    # ================================================================ #
    #  SCHEDULE – called daily 18:00                                   #
    # ================================================================ #
    def schedule(self, now):
        state, mask, _, sched_actions = self._build_structured_state(now)
        action = self._select_action(state, mask)
        if action < len(mask)-len(sched_actions):  # intake branch chosen
            # Track no-change action for delayed reward
            self.pending_actions.append((now, state, action, mask, None, 'no_change'))
            return []     # no staffing change
        sched_action_idx = action - (self.action_dim - len(sched_actions))
        if 0 <= sched_action_idx < len(sched_actions):
            res_type, t_eff, n = sched_actions[sched_action_idx]
        else:
            # Fallback: no scheduling change
            self.pending_actions.append((now, state, action, mask, None, 'no_change'))
            return []
        self.resources_now[res_type] = n
        self.future_plan[t_eff][res_type] = n
        
        # Track scheduling action for delayed reward
        self.pending_actions.append((now, state, action, mask, None, 'schedule'))
        return [(res_type, t_eff, n)]

    # ================================================================ #
    #  INTERNALS                                                       #
    # ================================================================ #
    def _update_diagnosis_queues(self, to_plan, to_replan, now):
        """Update per-diagnosis queues with new cases, maintaining FIFO order"""
        all_cases = to_plan + to_replan
        
        for case_id in all_cases:
            diagnosis = self.diagnosis_dict.get(case_id, 'No diagnosis')
            arrival_time = self.arrival_times.get(case_id, now)
            
            # Add to appropriate queue if not already there
            if case_id not in self.diagnosis_queues[diagnosis]:
                # Insert in order of arrival time (FIFO)
                self.diagnosis_queues[diagnosis].append(case_id)
                
                # Sort by arrival time (maintain FIFO)
                sorted_queue = sorted(self.diagnosis_queues[diagnosis], 
                                    key=lambda x: self.arrival_times.get(x, 0))
                self.diagnosis_queues[diagnosis] = collections.deque(sorted_queue)
    
    def _add_case_to_queue(self, case_id, diagnosis, arrival_time):
        """Add a case to its appropriate diagnosis queue"""
        if case_id not in self.diagnosis_queues[diagnosis]:
            self.diagnosis_queues[diagnosis].append(case_id)
            
            # Sort by arrival time (maintain FIFO)
            sorted_queue = sorted(self.diagnosis_queues[diagnosis], 
                                key=lambda x: self.arrival_times.get(x, 0))
            self.diagnosis_queues[diagnosis] = collections.deque(sorted_queue)
    
    def _assign_delayed_rewards(self, current_time, action_type, outcome_data):
        """Assign delayed rewards to pending actions based on outcomes"""
        if not self.train_mode:
            return
        
        # Calculate reward based on action type and outcome
        reward = 0
        
        if action_type == 'schedule':
            # Reward based on cost efficiency (aligns with personnel_cost)
            # Cost calculation: planned_ahead(1) + current_scheduled(2) + busy_resources(3)
            cost = outcome_data
            
            # Normalize cost by simulation duration to avoid bias towards shorter runs
            # Expected cost per hour: (5+30+40+4+9) * 1 = 88 base cost per hour
            # Maximum cost per hour: 88 * 3 = 264 (if all resources are busy)
            expected_hourly_cost = 88  # Base cost for all resources
            max_hourly_cost = 264      # Maximum possible cost per hour
            
            # Calculate cost efficiency (lower is better)
            cost_efficiency = max(0, (max_hourly_cost - cost) / max_hourly_cost)
            
            if cost_efficiency > 0.7:  # Very efficient (cost < 79)
                reward = 1.0
            elif cost_efficiency < 0.3:  # Inefficient (cost > 185)
                reward = -1.0
            else:
                reward = 0.0
                
        elif action_type == 'plan':
            case_id = outcome_data
            if case_id in self.wait_intake:
                waiting_time = self.wait_intake[case_id]
                # Reward based on waiting time (aligns with waiting_time_for_admission)
                # Planning constraint: must plan ≥24 hours ahead
                # Good: 24-48 hours (proper planning)
                # Bad: >72 hours (too much delay)
                if 24 <= waiting_time <= 48:  # Good: proper planning window
                    reward = 1.0
                elif waiting_time > 72:  # Bad: excessive delay
                    reward = -1.0
                elif waiting_time < 24:  # Bad: violates planning constraint
                    reward = -0.5
                else:
                    reward = 0.0
                    
        elif action_type == 'nervousness':
            # Reward based on replanning frequency (aligns with nervousness)
            replan_penalty = outcome_data
            # Nervousness penalty: max(0, 336 - (old_tp - tr)) / 336
            # 336 = 2 weeks, penalty increases as replanning gets closer to deadline
            if replan_penalty < 0.1:  # Low nervousness (good)
                reward = 0.5
            elif replan_penalty > 0.5:  # High nervousness (bad)
                reward = -1.0
            else:
                reward = 0.0
                
        elif action_type == 'hospital_wait':
            # Reward based on in-hospital waiting time (aligns with waiting_time_in_hospital)
            hospital_wait_time = outcome_data
            # Typical nursing durations: A1(4h), A2(8h), A3/A4(16h), B1(8h), B2/B3/B4(16h)
            # Good: <2 days (48h), Bad: >5 days (120h)
            if hospital_wait_time < 48:  # Good: efficient hospital flow
                reward = 1.0
            elif hospital_wait_time > 120:  # Bad: excessive hospital stay
                reward = -1.0
            else:
                reward = 0.0
        
        # Find and assign rewards to pending actions
        processed_actions = []
        for i, (timestamp, state, action, mask, case_id, pending_action_type) in enumerate(self.pending_actions):
            if pending_action_type == action_type:
                # Calculate time delay
                delay = current_time - timestamp
                
                # Apply time decay to reward (older actions get less credit)
                # Decay over 1 week (168 hours) to handle long-delay scenarios
                decay_factor = max(0.1, 1.0 - (delay / 168.0))
                adjusted_reward = reward * decay_factor
                
                # Store the experience with delayed reward
                self._remember(state, action, mask, adjusted_reward, None, False)
                processed_actions.append(i)
        
        # Remove processed actions
        for i in reversed(processed_actions):
            self.pending_actions.pop(i)
            
        return len(processed_actions)  # Return number of processed actions

    def _build_structured_state(self, now):
        """Create structured state vector with fixed slots per diagnosis"""
        
        # Initialize state components
        slot_data = []  # case_id and waiting_time for each slot
        diagnosis_counts = [0] * 9  # count of cases per diagnosis
        
        # Fill slots for each diagnosis
        slot_idx = 0
        for diagnosis, num_slots in self.slots_per_diagnosis.items():
            queue = self.diagnosis_queues[diagnosis]
            diagnosis_idx = self.diagnosis_encoding[diagnosis]
            
            # Fill available slots for this diagnosis
            for i in range(num_slots):
                if i < len(queue):
                    case_id = queue[i]
                    waiting_time = now - self.arrival_times.get(case_id, now)
                    slot_data.extend([hash(case_id) % 10000, waiting_time])
                    diagnosis_counts[diagnosis_idx] += 1
                else:
                    # Empty slot
                    slot_data.extend([0, 0])
                slot_idx += 1
        
        # Resource information
        fut = [self.resources_now[rt] for rt in ResourceType]
        
        # Time features
        hour = now % 24
        dow  = (now//24)%7
        clock = [math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24)]
        dow_onehot = [1 if i==dow else 0 for i in range(7)]
        
        # Construct state vector
        state = torch.tensor(slot_data + diagnosis_counts + fut + clock + dow_onehot,
                           dtype=torch.float32).unsqueeze(0)

        # ------------------- ACTION ENCODING -------------------------
        plan_actions  = []   # [(cid, adm_time)]
        sched_actions = []   # [(ResourceType, t, n)]
        mask          = torch.zeros(self.action_dim, dtype=torch.uint8).unsqueeze(0)

        # 1. Plan actions: one action per slot per time window
        idx = 0
        for diagnosis, num_slots in self.slots_per_diagnosis.items():
            queue = self.diagnosis_queues[diagnosis]
            for slot_idx in range(num_slots):
                if slot_idx < len(queue):
                    case_id = queue[slot_idx]
                    earliest = now + 24     # ≥ one day ahead
                    for h in range(24):     # 24 hourly slots
                        adm = earliest + h
                        plan_actions.append((case_id, adm))
                        if adm >= earliest: mask[0, idx] = 1
                        idx += 1
                else:
                    # Empty slot - no valid actions
                    idx += 24

        # 2. Schedule actions (same as before)
        for dr_or in (-1,0,1):
            for dr_ab in (-1,0,1):
                for dr_bb in (-1,0,1):
                    new_or = min(5, max(0, self.resources_now[ResourceType.OR]+dr_or))
                    new_ab = min(30,max(0, self.resources_now[ResourceType.A_BED]+dr_ab))
                    new_bb = min(40,max(0, self.resources_now[ResourceType.B_BED]+dr_bb))
                    t_eff  = now + 14
                    # respect decrease-only-after-1week rule
                    if dr_or<0 or dr_ab<0 or dr_bb<0:
                        t_eff = now + 168
                    sched_actions.append((ResourceType.OR, t_eff, new_or))
                    # the three resource types share the same tuple for simplicity
                    mask[0, idx] = 1
                    idx += 1

        return state, mask, plan_actions, sched_actions

    # ε-greedy --------------------------------------------------------
    def _select_action(self, state, mask):
        if self.train_mode:
            eps = self.EPS_END + (self.EPS_START-self.EPS_END) * \
                  math.exp(-1.*self.step_counter/self.EPS_DECAY)
            self.step_counter += 1
            if random.random() < eps:
                valid = torch.nonzero(mask[0]).squeeze().tolist()
                return random.choice(valid) if valid else 0
        with torch.no_grad():
            q = self.net(state, mask)
            return int(torch.argmax(q).item())

    # store + learn ---------------------------------------------------
    def _remember(self, s, a, m, reward, s2, done):
        if self.train_mode:
            self.mem.push(s, a, reward, s2, done, m)
            if len(self.mem) >= self.BATCH:
                self._learn()

    def _learn(self):
        batch = self.mem.sample(self.BATCH)
        non_final = torch.tensor([s is not None for s in batch.next_state])
        non_next  = torch.cat([s for s in batch.next_state if s is not None], dim=0) \
            if any(non_final) else None
        state     = torch.cat(batch.state)
        action    = torch.tensor(batch.action).unsqueeze(1)
        reward    = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        mask      = torch.cat(batch.mask)

        q_sa      = self.net(state, mask).gather(1, action)
        q_next    = torch.zeros_like(q_sa)
        if any(non_final):
            with torch.no_grad():
                best = torch.argmax(self.net(non_next, None), dim=1, keepdim=True)
                q_next[non_final] = self.tgt(non_next, None).gather(1, best)
        target = reward + self.GAMMA * q_next * (~torch.tensor(batch.done).unsqueeze(1))
        loss   = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        if self.step_counter % self.TARGET_UPD == 0:
            self.tgt.load_state_dict(self.net.state_dict())

    # call this at the end of a long training run
    def save(self):
        if self.train_mode:
            torch.save(self.net.state_dict(), self.model_path)
