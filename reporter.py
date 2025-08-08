from simulator import EventType
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import os
NO_DIAGNOSIS = 'No diagnosis'

class Reporter:

    def __init__(self):
        # The simulation starts at Monday 2018-01-01 00:00:00.000
        # Simulation time is in hours from the initial time	
        self.initial_time = datetime(2018, 1, 1)
        self.time_format = "%Y-%m-%d %H:%M:%S.%f"
    
    def get_formatted_timestamp(self, timestamp):
         return (self.initial_time + timedelta(hours=timestamp)).strftime(self.time_format)

    def callback(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        t = self.get_formatted_timestamp(timestamp)
        if element is not None:
            tt = element.label
            dt = '\t' + '\t'.join(str(v) for v in element.data.values())
        else:
            tt = str(None)
            dt = ""
            return str(case_id) + "\t" + tt + "\t" + t + "\t" + str(resource) + "\t" + str(lifecycle_state) + dt


class EventLogReporter(Reporter):
    """
    This class is used to log the events of the simulation in a CSV file that can be read in a process mining tool.
    The CSV file has a header with the following columns: case_id, task_id, event_label, resource, start_time, completion_time, data_type1, data_type2, ...
    The constructor takes two arguments: the filename of the CSV file and a list of data types that will be logged in the CSV file.
    """
    def __init__(self, filename, data_types):
        super().__init__()
        self.task_start_times = dict()
        self.data_types = data_types

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.logfile = open(filename, "wt")
        self.logfile.write("case_id,task_id,event_label,resource,start_time,completion_time")
        for data_type in data_types:
            self.logfile.write("," + data_type)        
        self.logfile.write("\n")

    def callback(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        if lifecycle_state==EventType.START_TASK:
            self.task_start_times[(case_id, element.label)] = self.get_formatted_timestamp(timestamp)
        elif lifecycle_state==EventType.COMPLETE_EVENT or lifecycle_state==EventType.COMPLETE_TASK:
            completion_time = self.get_formatted_timestamp(timestamp)
            if element.is_task():
                start_time = self.task_start_times[(case_id, element.label)]
                del self.task_start_times[(case_id, element.label)]
            else:
                start_time = completion_time
            resource = resource
            self.logfile.write(str(case_id) + ",")
            self.logfile.write(str(element.id) + ",")
            self.logfile.write(element.label + ",")
            self.logfile.write(str(resource) + ",")
            self.logfile.write(start_time + ",")
            self.logfile.write(completion_time)
            for data_type in self.data_types:
                if data_type in element.data:
                    self.logfile.write("," + str(element.data[data_type]))
                else:
                    self.logfile.write(",")
            self.logfile.write("\n")
            self.logfile.flush()

    def close(self):
        self.logfile.close()


class ResourceScheduleReporter(Reporter):
    """
    Records EventType.SCHEDULE_RESOURCES events. At each event simply records the number of available resources in double lists to facilitate making a graph of that later.
    Has a function to create a graph of the aavailable resources over time.
    """    
    def __init__(self):
        super().__init__()
        self.available_resources = []
        self.busy_resources = []
        self.away_resources = []
        self.time = []
    
    def callback(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        if lifecycle_state==EventType.SCHEDULE_RESOURCES:
            self.available_resources.append(data['available_resources'])
            self.busy_resources.append(data['busy_resources'])
            self.away_resources.append(data['away_resources'])
            self.time.append(timestamp)
    
    def create_graph(self, start=0, end=None):
        """
        Plot the number of available, busy, and away resources over time.
        The lines must be stacked, so the graph shows the total number of resources at each time.        
        """
        if end is None:
            end = len(self.time)
        plt.fill_between(self.time[start:end], 0, self.busy_resources[start:end], color='blue', alpha=0.5, label='Busy')
        plt.fill_between(self.time[start:end], self.busy_resources[start:end], 
                 [self.available_resources[i]+self.busy_resources[i] for i in range(start, end)], 
                 color='green', alpha=0.5, label='Available')
        plt.fill_between(self.time[start:end], 
                 [self.available_resources[i]+self.busy_resources[i] for i in range(start, end)], 
                 [self.available_resources[i]+self.busy_resources[i]+self.away_resources[i] for i in range(start, end)], 
                 color='red', alpha=0.5, label='Away')
        plt.xlabel('Time')
        plt.ylabel('Number of resources')
        plt.legend()
        plt.show()


class DfReporter(Reporter):
    """
    This class stores all simulation events and resource scheduling information in pandas DataFrames.
    It combines the functionality of both EventLogReporter and ResourceScheduleReporter.
    The data is stored in DataFrame attributes that can be accessed at any time.
    """
    
    def __init__(self, data_types=None):
        super().__init__()
        
        # Initialize data types for event log
        self.data_types = data_types if data_types else []
        
        # Track task start times for event log functionality
        self.task_start_times = dict()
        
        # Initialize diagnosis dictionary to track diagnosis by case_id
        self.diagnosis_dict = {}
        
        # Initialize event log DataFrame
        event_columns = ['case_id', 'task_id', 'event_label', 'resource', 'start_time', 'completion_time'] + self.data_types
        self.event_log_df = pd.DataFrame(columns=event_columns)
        
        # Initialize resource schedule DataFrame
        resource_columns = ['timestamp', 'available_resources', 'busy_resources', 'away_resources']
        self.resource_schedule_df = pd.DataFrame(columns=resource_columns)
    
    def callback(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        # Check for diagnosis information in specific event labels
        if element is not None and element.label in ['emergency_patient', 'patient_referal']:
            if hasattr(element, 'data') and 'diagnosis' in element.data:
                if element.data['diagnosis'] is not None:
                    self.diagnosis_dict[case_id] = element.data['diagnosis']
                else:
                    self.diagnosis_dict[case_id] = NO_DIAGNOSIS
        
        # Handle event log data (similar to EventLogReporter)
        if lifecycle_state == EventType.START_TASK:
            self.task_start_times[(case_id, element.label)] = self.get_formatted_timestamp(timestamp)
        elif lifecycle_state == EventType.COMPLETE_EVENT or lifecycle_state == EventType.COMPLETE_TASK:
            completion_time = self.get_formatted_timestamp(timestamp)
            if element.is_task():
                start_time = self.task_start_times.get((case_id, element.label), completion_time)
                if (case_id, element.label) in self.task_start_times:
                    del self.task_start_times[(case_id, element.label)]
            else:
                start_time = completion_time
            
            # Create event log entry
            event_entry = {
                'case_id': case_id,
                'task_id': element.id,
                'event_label': element.label,
                'resource': resource,
                'start_time': start_time,
                'completion_time': completion_time
            }
            
            # Add data type values
            for data_type in self.data_types:
                if data_type in element.data:
                    event_entry[data_type] = element.data[data_type]
                else:
                    event_entry[data_type] = None
            
            # Add to event log DataFrame
            self.event_log_df = pd.concat([self.event_log_df, pd.DataFrame([event_entry])], ignore_index=True)
        
        # Handle resource schedule data (similar to ResourceScheduleReporter)
        elif lifecycle_state == EventType.SCHEDULE_RESOURCES:
            resource_entry = {
                'timestamp': timestamp,
                'available_resources': data['available_resources'],
                'busy_resources': data['busy_resources'],
                'away_resources': data['away_resources']
            }
            
            # Add to resource schedule DataFrame
            self.resource_schedule_df = pd.concat([self.resource_schedule_df, pd.DataFrame([resource_entry])], ignore_index=True)
    
    def get_event_log_df(self):
        """Return the event log DataFrame"""
        return self.event_log_df.copy()
    
    def get_resource_schedule_df(self):
        """Return the resource schedule DataFrame"""
        return self.resource_schedule_df.copy()
    
    def get_diagnosis_dict(self):
        """Return the diagnosis dictionary"""
        return self.diagnosis_dict.copy()
    
    def create_resource_graph(self, start=0, end=None):
        """
        Plot the number of available, busy, and away resources over time.
        Similar to ResourceScheduleReporter.create_graph() but uses DataFrame data.
        """
        if self.resource_schedule_df.empty:
            print("No resource schedule data available")
            return
        
        df = self.resource_schedule_df.iloc[start:end] if end is not None else self.resource_schedule_df.iloc[start:]
        
        plt.fill_between(df['timestamp'], 0, df['busy_resources'], color='blue', alpha=0.5, label='Busy')
        plt.fill_between(df['timestamp'], df['busy_resources'], 
                        df['available_resources'] + df['busy_resources'], 
                        color='green', alpha=0.5, label='Available')
        plt.fill_between(df['timestamp'], 
                        df['available_resources'] + df['busy_resources'], 
                        df['available_resources'] + df['busy_resources'] + df['away_resources'], 
                        color='red', alpha=0.5, label='Away')
        plt.xlabel('Time')
        plt.ylabel('Number of resources')
        plt.legend()
        plt.show()
    
    def export_event_log_csv(self, filename):
        """Export event log DataFrame to CSV file"""
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        self.event_log_df.to_csv(filename, index=False)
    
    def export_resource_schedule_csv(self, filename):
        """Export resource schedule DataFrame to CSV file"""
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        self.resource_schedule_df.to_csv(filename, index=False)