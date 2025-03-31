# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataProcessor:
    def __init__(self, data_path):
        """Initialize data processor
        
        Args:
            data_path (str): Path to data file
        """
        self.data_path = data_path
        self.df = None
        self.sensor_mapping = {}
        self.activity_mapping = {}
        
    def load_data(self):
        """Load and preprocess data"""
        # Read data
        self.df = pd.read_csv(self.data_path, sep=r'\s+', header=None,
                            names=['timestamp', 'sensor', 'sensor_dup', 'state', 'activity'],
                            dtype=str)
        
        # Remove duplicate sensor columns
        self.df.drop(columns=['sensor_dup'], inplace=True)
        
        # Process timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.dropna(subset=['timestamp'])
        self.df['timestamp'] = self.df['timestamp'].ffill()
        self.df = self.df.sort_values(by='timestamp')
        
        # Create sensor and activity mappings
        self.create_mappings()
        
        return self.df
    
    def create_mappings(self):
        """Create mappings for sensors and activities"""
        # Sensor mapping
        unique_sensors = sorted(self.df['sensor'].unique())
        self.sensor_mapping = {sensor: idx for idx, sensor in enumerate(unique_sensors)}
        
        # Activity mapping
        unique_activities = sorted(pd.unique(self.df['activity'].dropna()))
        self.activity_mapping = {activity: idx for idx, activity in enumerate(unique_activities)}
        
        # Add "Unknown" activity
        self.activity_mapping['Unknown'] = len(self.activity_mapping)
    
    def get_statistics(self):
        """Get basic statistics of the dataset"""
        if self.df is None:
            self.load_data()
            
        stats = {
            'num_sensors': len(self.sensor_mapping),
            'num_activities': len(self.activity_mapping) - 1,  # Subtract Unknown
            'num_data_points': len(self.df),
            'start_time': self.df['timestamp'].min(),
            'end_time': self.df['timestamp'].max(),
            'sensor_list': list(self.sensor_mapping.keys()),
            'activity_list': [k for k in self.activity_mapping.keys() if k != 'Unknown']
        }
        
        return stats
    
    def encode_data(self):
        """Encode data into numeric format"""
        if self.df is None:
            self.load_data()
            
        # Encode sensors
        self.df['sensor_code'] = self.df['sensor'].map(self.sensor_mapping)
        
        # Encode activities, map NaN values to "Unknown"
        self.df['activity_code'] = self.df['activity'].fillna('Unknown').map(self.activity_mapping)
        
        # Encode state (ON/OFF)
        self.df['state_code'] = (self.df['state'] == 'ON').astype(int)
        
        return self.df

if __name__ == '__main__':
    # Test data processing
    processor = DataProcessor('dataset/base_kasteren-m.csv')
    df = processor.load_data()
    stats = processor.get_statistics()
    
    print('\n=== Dataset Statistics ===')
    print('Number of sensors:', stats['num_sensors'])
    print('Number of activities:', stats['num_activities'])
    print('Number of data points:', stats['num_data_points'])
    print('Start time:', stats['start_time'])
    print('End time:', stats['end_time'])
    print('\nSensor list:', stats['sensor_list'])
    print('\nActivity list:', stats['activity_list'])
        
    print('\n=== Encoded Data Example ===')
    encoded_df = processor.encode_data()
    print(encoded_df.head()) 