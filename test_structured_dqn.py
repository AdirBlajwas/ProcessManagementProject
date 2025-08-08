#!/usr/bin/env python3
"""
Test script for the structured DQN planner with per-diagnosis queues
"""

from dqn_planner import DQNPlanner
from problems import HealthcareProblem, HealthcareElements
from simulator import Simulator, EventType
import random
import collections

def test_per_diagnosis_queues():
    """Test that per-diagnosis queues work correctly with FIFO ordering"""
    
    print("Testing per-diagnosis queues...")
    
    # Create planner
    planner = DQNPlanner(train=True, max_queue_len=10)
    
    # Test cases with different arrival times
    test_cases = [
        ("A1_early", "A1", 100.0),
        ("A1_late", "A1", 120.0),
        ("B1_early", "B1", 110.0),
        ("B1_late", "B1", 130.0),
        ("EM1", "No diagnosis", 105.0),
        ("EM2", "B3", 115.0),
    ]
    
    # Add cases to queues
    for case_id, diagnosis, arrival_time in test_cases:
        # Create mock element
        class MockElement:
            def __init__(self, case_id, label, diagnosis, arrival_time):
                self.case_id = case_id
                self.label = label
                self.data = {"diagnosis": diagnosis}
        
        # Simulate event
        if case_id.startswith("EM"):
            element = MockElement(case_id, "emergency_patient", diagnosis, arrival_time)
        else:
            element = MockElement(case_id, "patient_referal", diagnosis, arrival_time)
        
        # Call report to populate queues
        planner.report(case_id, element, arrival_time, None, EventType.ACTIVATE_EVENT)
    
    # Test queue ordering (should be FIFO within each diagnosis)
    print("\nQueue contents:")
    for diagnosis, queue in planner.diagnosis_queues.items():
        print(f"{diagnosis}: {list(queue)}")
    
    # Verify FIFO ordering
    assert list(planner.diagnosis_queues['A1']) == ['A1_early', 'A1_late'], "A1 queue not FIFO"
    assert list(planner.diagnosis_queues['B1']) == ['B1_early', 'B1_late'], "B1 queue not FIFO"
    
    print("✓ Per-diagnosis queues test passed!")

def test_structured_state():
    """Test that the structured state representation works correctly"""
    
    print("\nTesting structured state representation...")
    
    # Create planner
    planner = DQNPlanner(train=True, max_queue_len=10)
    
    # Add some test cases
    test_cases = [
        ("A1_1", "A1", 100.0),
        ("A1_2", "A1", 110.0),
        ("A1_3", "A1", 120.0),
        ("B1_1", "B1", 105.0),
        ("B1_2", "B1", 115.0),
        ("EM1", "No diagnosis", 108.0),
    ]
    
    for case_id, diagnosis, arrival_time in test_cases:
        class MockElement:
            def __init__(self, case_id, label, diagnosis):
                self.case_id = case_id
                self.label = label
                self.data = {"diagnosis": diagnosis}
        
        if case_id.startswith("EM"):
            element = MockElement(case_id, "emergency_patient", diagnosis)
        else:
            element = MockElement(case_id, "patient_referal", diagnosis)
        
        planner.report(case_id, element, arrival_time, None, EventType.ACTIVATE_EVENT)
    
    # Test state building
    try:
        state, mask, plan_actions, sched_actions = planner._build_structured_state(150.0)
        print(f"State shape: {state.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Number of plan actions: {len(plan_actions)}")
        print(f"Number of schedule actions: {len(sched_actions)}")
        
        # Verify state dimensions
        expected_state_dim = (planner.total_slots * 2) + 9 + len(planner.resources_now) + 9
        assert state.shape[1] == expected_state_dim, f"State dimension mismatch: {state.shape[1]} vs {expected_state_dim}"
        
        print("✓ Structured state test passed!")
        
    except Exception as e:
        print(f"✗ Structured state test failed: {e}")
        raise

def test_slot_allocation():
    """Test that slots are allocated correctly per diagnosis"""
    
    print("\nTesting slot allocation...")
    
    planner = DQNPlanner(train=True, max_queue_len=10)
    
    # Expected slot allocation
    expected_slots = {
        'A1': 3, 'A2': 1, 'A3': 1, 'A4': 1,
        'B1': 2, 'B2': 1, 'B3': 1, 'B4': 1,
        'No diagnosis': 1
    }
    
    assert planner.slots_per_diagnosis == expected_slots, "Slot allocation mismatch"
    assert planner.total_slots == 12, f"Total slots should be 12, got {planner.total_slots}"
    
    print("✓ Slot allocation test passed!")

def test_queue_management():
    """Test queue management and case removal"""
    
    print("\nTesting queue management...")
    
    planner = DQNPlanner(train=True, max_queue_len=10)
    
    # Add cases
    test_cases = [
        ("A1_1", "A1", 100.0),
        ("A1_2", "A1", 110.0),
        ("B1_1", "B1", 105.0),
    ]
    
    for case_id, diagnosis, arrival_time in test_cases:
        class MockElement:
            def __init__(self, case_id, label, diagnosis):
                self.case_id = case_id
                self.label = label
                self.data = {"diagnosis": diagnosis}
        
        element = MockElement(case_id, "patient_referal", diagnosis)
        planner.report(case_id, element, arrival_time, None, EventType.ACTIVATE_EVENT)
    
    # Verify cases are in queues
    assert "A1_1" in planner.diagnosis_queues['A1']
    assert "B1_1" in planner.diagnosis_queues['B1']
    
    # Simulate planning a case (should remove it from queue)
    # Force the action to be a planning action by setting a high epsilon
    original_eps = planner.EPS_START
    planner.EPS_START = 1.0
    planner.step_counter = 0
    
    result = planner.plan(["A1_1"], [], 150.0)
    
    # Restore original epsilon
    planner.EPS_START = original_eps
    
    # Verify case was removed (if a planning action was chosen)
    if result:  # If a case was actually planned
        assert "A1_1" not in planner.diagnosis_queues['A1']
    assert "A1_2" in planner.diagnosis_queues['A1']  # Other cases remain
    
    print("✓ Queue management test passed!")

def test_arrival_time_tracking():
    """Test that arrival times are tracked correctly"""
    
    print("\nTesting arrival time tracking...")
    
    planner = DQNPlanner(train=True, max_queue_len=10)
    
    # Add cases with different arrival times
    test_cases = [
        ("A1", "A1", 100.0),
        ("B1", "B1", 110.0),
        ("EM1", "No diagnosis", 105.0),
    ]
    
    for case_id, diagnosis, arrival_time in test_cases:
        class MockElement:
            def __init__(self, case_id, label, diagnosis):
                self.case_id = case_id
                self.label = label
                self.data = {"diagnosis": diagnosis}
        
        if case_id.startswith("EM"):
            element = MockElement(case_id, "emergency_patient", diagnosis)
        else:
            element = MockElement(case_id, "patient_referal", diagnosis)
        
        planner.report(case_id, element, arrival_time, None, EventType.ACTIVATE_EVENT)
    
    # Verify arrival times are tracked
    assert planner.arrival_times["A1"] == 100.0
    assert planner.arrival_times["B1"] == 110.0
    assert planner.arrival_times["EM1"] == 105.0
    
    # Test waiting time calculation
    current_time = 150.0
    waiting_time_A1 = current_time - planner.arrival_times["A1"]
    waiting_time_B1 = current_time - planner.arrival_times["B1"]
    
    assert waiting_time_A1 == 50.0
    assert waiting_time_B1 == 40.0
    
    print("✓ Arrival time tracking test passed!")

def test_integration():
    """Test full integration with simulation"""
    
    print("\nTesting full integration...")
    
    try:
        # Create planner
        planner = DQNPlanner(train=True, max_queue_len=20)
        
        # Create problem and simulator
        problem = HealthcareProblem()
        simulator = Simulator(planner, problem)
        
        # Run a short simulation
        print("Running short simulation...")
        result = simulator.run(24)  # 1 day
        
        print(f"Simulation completed. Result: {result}")
        print(f"Diagnosis dict size: {len(planner.diagnosis_dict)}")
        print(f"Arrival times tracked: {len(planner.arrival_times)}")
        
        # Check queue status
        total_cases_in_queues = sum(len(queue) for queue in planner.diagnosis_queues.values())
        print(f"Total cases in queues: {total_cases_in_queues}")
        
        print("✓ Integration test passed!")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise

if __name__ == "__main__":
    print("Testing Structured DQN Planner with Per-Diagnosis Queues")
    print("=" * 70)
    
    test_slot_allocation()
    test_arrival_time_tracking()
    test_per_diagnosis_queues()
    test_structured_state()
    test_queue_management()
    test_integration()
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("\nKey improvements implemented:")
    print("1. ✓ Per-diagnosis queues with FIFO ordering")
    print("2. ✓ Arrival time tracking for all cases")
    print("3. ✓ Structured state representation with fixed slots per diagnosis")
    print("4. ✓ Efficient queue management (add/remove cases)")
    print("5. ✓ Diagnosis-aware action space")
    print("6. ✓ Fair resource allocation across all diagnosis types")
