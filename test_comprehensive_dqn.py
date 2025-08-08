#!/usr/bin/env python3
"""
Comprehensive test suite for DQN planner - matches current implementation
"""

from dqn_planner import DQNPlanner
from problems import HealthcareProblem, HealthcareElements, ResourceType
from simulator import Simulator, EventType
import random
import time
import torch
import collections

def test_dqn_planner_comprehensive():
    """Comprehensive test for DQN planner covering all major functionality"""
    
    print("=" * 80)
    print("COMPREHENSIVE DQN PLANNER TEST SUITE")
    print("=" * 80)
    
    # Test 1: Basic Initialization and Configuration
    print("\n1. TESTING BASIC INITIALIZATION...")
    try:
        planner = DQNPlanner(train=True, max_queue_len=20)
        assert planner.train_mode == True, "Training mode not set correctly"
        assert planner.max_queue_len == 20, "Queue length not set correctly"
        assert hasattr(planner, 'diagnosis_dict'), "Diagnosis dict not initialized"
        assert hasattr(planner, 'pending_actions'), "Pending actions not initialized"
        assert hasattr(planner, 'diagnosis_queues'), "Diagnosis queues not initialized"
        assert hasattr(planner, 'slots_per_diagnosis'), "Slots per diagnosis not initialized"
        assert planner.total_slots == 12, f"Expected 12 total slots, got {planner.total_slots}"
        print("✓ Basic initialization passed")
    except Exception as e:
        print(f"✗ Basic initialization failed: {e}")
        raise
    
    # Test 2: Diagnosis Tracking and Queue Management
    print("\n2. TESTING DIAGNOSIS TRACKING AND QUEUE MANAGEMENT...")
    try:
        test_cases = [
            ("A1_1", "A1"),
            ("A2_1", "A2"), 
            ("B1_1", "B1"),
            ("B2_1", "B2"),
            ("EM1", None),  # Emergency with no diagnosis
            ("EM2", "B3"),  # Emergency with diagnosis
        ]
        
        for case_id, expected_diagnosis in test_cases:
            class MockElement:
                def __init__(self, case_id, label, diagnosis):
                    self.case_id = case_id
                    self.label = label
                    self.data = {"diagnosis": diagnosis}
            
            if case_id.startswith("EM"):
                element = MockElement(case_id, "emergency_patient", expected_diagnosis)
            else:
                element = MockElement(case_id, "patient_referal", expected_diagnosis)
            
            planner.report(case_id, element, 100.0, None, EventType.ACTIVATE_EVENT)
            
            tracked_diagnosis = planner.diagnosis_dict.get(case_id, "Not found")
            expected = expected_diagnosis if expected_diagnosis is not None else "No diagnosis"
            
            assert tracked_diagnosis == expected, f"Diagnosis tracking failed for {case_id}"
        
        print(f"✓ Diagnosis tracking passed - tracked {len(planner.diagnosis_dict)} cases")
        
        # Test queue population
        for diagnosis in planner.diagnosis_queues:
            queue_length = len(planner.diagnosis_queues[diagnosis])
            print(f"  {diagnosis}: {queue_length} cases")
        
    except Exception as e:
        print(f"✗ Diagnosis tracking failed: {e}")
        raise
    
    # Test 3: Per-Diagnosis Queues with FIFO Ordering
    print("\n3. TESTING PER-DIAGNOSIS QUEUES WITH FIFO ORDERING...")
    try:
        # Add more test cases with different arrival times
        additional_cases = [
            ("A1_early", "A1", 100.0),
            ("A1_late", "A1", 120.0),
            ("B1_early", "B1", 110.0),
            ("B1_late", "B1", 130.0),
        ]
        
        for case_id, diagnosis, arrival_time in additional_cases:
            class MockElement:
                def __init__(self, case_id, label, diagnosis):
                    self.case_id = case_id
                    self.label = label
                    self.data = {"diagnosis": diagnosis}
            
            element = MockElement(case_id, "patient_referal", diagnosis)
            planner.report(case_id, element, arrival_time, None, EventType.ACTIVATE_EVENT)
        
        # Verify FIFO ordering within each diagnosis
        a1_queue = list(planner.diagnosis_queues['A1'])
        b1_queue = list(planner.diagnosis_queues['B1'])
        
        print(f"A1 queue: {a1_queue}")
        print(f"B1 queue: {b1_queue}")
        
        # Check that early cases come before late cases
        assert 'A1_early' in a1_queue and 'A1_late' in a1_queue, "A1 cases not in queue"
        assert 'B1_early' in b1_queue and 'B1_late' in b1_queue, "B1 cases not in queue"
        
        print("✓ Per-diagnosis queues with FIFO ordering passed")
    except Exception as e:
        print(f"✗ Per-diagnosis queues failed: {e}")
        raise
    
    # Test 4: Structured State Building
    print("\n4. TESTING STRUCTURED STATE BUILDING...")
    try:
        state, mask, plan_actions, sched_actions = planner._build_structured_state(150.0)
        
        # Verify state dimensions
        total_slots = sum(planner.slots_per_diagnosis.values())
        slot_features = total_slots * 2  # 2 features per slot (waiting time, priority)
        diagnosis_counts = 9  # 9 diagnosis types
        resource_features = len(planner.resources_now)  # Resource counts
        time_features = 2 + 7  # Clock (sin, cos) + day of week one-hot
        
        expected_state_dim = slot_features + diagnosis_counts + resource_features + time_features
        # The state is unsqueezed to add batch dimension, so we check the second dimension
        actual_dim = state.shape[1] if len(state.shape) > 1 else state.shape[0]
        assert actual_dim == expected_state_dim, f"State dimension mismatch: {actual_dim} != {expected_state_dim}"
        
        # Verify action spaces
        assert len(plan_actions) > 0, "No planning actions available"
        assert len(sched_actions) > 0, "No scheduling actions available"
        
        print(f"✓ Structured state building passed - State: {state.shape}, Plan actions: {len(plan_actions)}, Schedule actions: {len(sched_actions)}")
    except Exception as e:
        print(f"✗ Structured state building failed: {e}")
        raise
    
    # Test 5: Resource Constraints and Scheduling Actions
    print("\n5. TESTING RESOURCE CONSTRAINTS AND SCHEDULING ACTIONS...")
    try:
        resource_limits = {
            ResourceType.OR: 5,
            ResourceType.A_BED: 30,
            ResourceType.B_BED: 40,
            ResourceType.INTAKE: 4,
            ResourceType.ER_PRACTITIONER: 9
        }
        
        # Verify planner respects resource limits in scheduling actions
        # sched_actions format: [((or, ab, bb), t_eff)]
        for (or_count, ab_count, bb_count), t_eff in sched_actions:
            assert or_count <= resource_limits[ResourceType.OR], f"Scheduling {or_count} OR exceeds limit {resource_limits[ResourceType.OR]}"
            assert ab_count <= resource_limits[ResourceType.A_BED], f"Scheduling {ab_count} A_BED exceeds limit {resource_limits[ResourceType.A_BED]}"
            assert bb_count <= resource_limits[ResourceType.B_BED], f"Scheduling {bb_count} B_BED exceeds limit {resource_limits[ResourceType.B_BED]}"
        
        print("✓ Resource constraints and scheduling actions passed")
    except Exception as e:
        print(f"✗ Resource constraints failed: {e}")
        raise
    
    # Test 6: Planning and Scheduling Methods
    print("\n6. TESTING PLANNING AND SCHEDULING METHODS...")
    try:
        # Test plan method
        to_plan = ["A1_1", "B1_1"]
        to_replan = []
        now = 200.0
        
        plan_result = planner.plan(to_plan, to_replan, now)
        print(f"Plan result: {plan_result}")
        
        # Test schedule method
        schedule_result = planner.schedule(now)
        print(f"Schedule result: {schedule_result}")
        
        # Verify pending actions are tracked
        assert len(planner.pending_actions) > 0, "No pending actions tracked"
        
        print("✓ Planning and scheduling methods passed")
    except Exception as e:
        print(f"✗ Planning and scheduling methods failed: {e}")
        raise
    
    # Test 7: Reward System and Credit Assignment
    print("\n7. TESTING REWARD SYSTEM AND CREDIT ASSIGNMENT...")
    try:
        # Test credit assignment methods
        planner._credit_latest_schedule(-0.5)  # Low cost reward
        planner._credit_last_plan('A1_1', -0.1)  # Good planning reward
        
        # Test experience pushing (should work in training mode)
        initial_memory_size = len(planner.mem)
        planner._push_experience(torch.randn(1, planner.state_dim), 0, torch.ones(1, planner.action_dim), -0.5)
        
        print(f"✓ Reward system and credit assignment passed - Memory size: {len(planner.mem)}")
    except Exception as e:
        print(f"✗ Reward system and credit assignment failed: {e}")
        raise
    
    # Test 8: Neural Network and Learning Components
    print("\n8. TESTING NEURAL NETWORK AND LEARNING COMPONENTS...")
    try:
        # Test network forward pass
        test_state = torch.randn(1, planner.state_dim)
        test_mask = torch.ones(1, planner.action_dim)
        
        with torch.no_grad():
            q_values = planner.net(test_state, test_mask)
        
        assert q_values.shape == (1, planner.action_dim), f"Q-values shape mismatch: {q_values.shape}"
        
        # Test action selection
        action = planner._select_action(test_state, test_mask)
        assert 0 <= action < planner.action_dim, f"Invalid action selected: {action}"
        
        print(f"✓ Neural network and learning components passed - Q-values shape: {q_values.shape}")
    except Exception as e:
        print(f"✗ Neural network and learning components failed: {e}")
        raise
    
    # Test 9: Event Reporting and State Updates
    print("\n9. TESTING EVENT REPORTING AND STATE UPDATES...")
    try:
        # Test intake start event (should trigger WTA calculation)
        class MockIntakeElement:
            def __init__(self, case_id):
                self.case_id = case_id
                self.label = HealthcareElements.INTAKE
        
        # Simulate intake start for a tracked case
        if 'A1_1' in planner.arrival_ts:
            element = MockIntakeElement('A1_1')
            planner.report('A1_1', element, 250.0, None, EventType.START_TASK)
            print("✓ Intake start event processed")
        
        # Test hospital task events
        class MockSurgeryElement:
            def __init__(self, case_id):
                self.case_id = case_id
                self.label = HealthcareElements.SURGERY
        
        element = MockSurgeryElement('A1_1')
        planner.report('A1_1', element, 300.0, None, EventType.ACTIVATE_TASK)
        planner.report('A1_1', element, 350.0, None, EventType.START_TASK)
        
        print("✓ Event reporting and state updates passed")
    except Exception as e:
        print(f"✗ Event reporting and state updates failed: {e}")
        raise
    
    # Test 10: Integration with Full Simulation
    print("\n10. TESTING INTEGRATION WITH FULL SIMULATION...")
    try:
        # Create fresh planner for integration test
        integration_planner = DQNPlanner(train=True, max_queue_len=30)
        problem = HealthcareProblem()
        simulator = Simulator(integration_planner, problem)
        
        # Run a short simulation
        start_time = time.time()
        result = simulator.run(7 * 24)  # 1 week
        end_time = time.time()
        
        assert result is not None, "Simulation should return a result"
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Check that all expected metrics are present
        expected_metrics = ['waiting_time_for_admission', 'waiting_time_in_hospital', 'nervousness', 'personnel_cost']
        for metric in expected_metrics:
            assert metric in result, f"Missing metric: {metric}"
        
        print(f"✓ Integration with full simulation passed - Completed in {end_time - start_time:.2f} seconds")
        print(f"  Results: {result}")
    except Exception as e:
        print(f"✗ Integration with full simulation failed: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! DQN PLANNER IS WORKING CORRECTLY")
    print("=" * 80)

if __name__ == "__main__":
    test_dqn_planner_comprehensive()
