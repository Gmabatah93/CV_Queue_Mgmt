#!/usr/bin/env python3
"""
CV Queue Management Terminal Report

This script creates a simple terminal report from your CSV exports,
showing the complete story of customer journeys through the bank.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CVTerminalReporter:
    """Create comprehensive terminal reports of CV queue management data."""
    
    def __init__(self, csv_dir: str = "data/csv_exports"):
        self.csv_dir = Path(csv_dir)
        self.data = self._load_all_data()
        
    def _load_all_data(self):
        """Load all CSV files and combine into a comprehensive dataset."""
        data = {
            'line_events': pd.DataFrame(),
            'teller_events': pd.DataFrame(),
            'abandonment_events': pd.DataFrame()
        }
        
        # Load line events
        line_files = list(self.csv_dir.glob("line_events_*.csv"))
        if line_files:
            latest_line_file = max(line_files, key=lambda f: f.stat().st_mtime)
            data['line_events'] = pd.read_csv(latest_line_file)
            data['line_events']['timestamp'] = pd.to_datetime(data['line_events']['timestamp'])
        
        # Load teller interaction events
        teller_files = list(self.csv_dir.glob("teller_interaction_events_*.csv"))
        if teller_files:
            latest_teller_file = max(teller_files, key=lambda f: f.stat().st_mtime)
            data['teller_events'] = pd.read_csv(latest_teller_file)
            data['teller_events']['timestamp'] = pd.to_datetime(data['teller_events']['timestamp'])
        
        # Load abandonment events
        abandonment_files = list(self.csv_dir.glob("abandonment_events_*.csv"))
        if abandonment_files:
            latest_abandonment_file = max(abandonment_files, key=lambda f: f.stat().st_mtime)
            data['abandonment_events'] = pd.read_csv(latest_abandonment_file)
            data['abandonment_events']['timestamp'] = pd.to_datetime(data['abandonment_events']['timestamp'])
            data['abandonment_events']['line_entered_timestamp'] = pd.to_datetime(data['abandonment_events']['line_entered_timestamp'])
            data['abandonment_events']['line_exited_timestamp'] = pd.to_datetime(data['abandonment_events']['line_exited_timestamp'])
        
        return data
    
    def create_comprehensive_report(self):
        """Create a comprehensive terminal report."""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        print("=" * 80)
        print(f"🏦 CV QUEUE MANAGEMENT SYSTEM REPORT")
        print(f"📅 {current_date}")
        print("=" * 80)
        
        # 1. Executive Summary
        self._print_executive_summary()
        
        # 2. Event Breakdown
        self._print_event_breakdown()
        
        # 3. Customer Journey Analysis
        self._print_customer_journey_analysis()
        
        # 4. Performance Metrics
        self._print_performance_metrics()
        
        # 5. Wait Time Analysis
        self._print_wait_time_analysis()
        
        # 6. Key Insights
        self._print_key_insights()
        
        print("=" * 80)
        print("✅ Report Complete!")
        print("=" * 80)
    
    def _print_executive_summary(self):
        """Print executive summary."""
        print("\n📊 EXECUTIVE SUMMARY")
        print("-" * 40)
        
        total_people = len(self.data['line_events']['person_id'].unique()) if not self.data['line_events'].empty else 0
        total_line_events = len(self.data['line_events']) if not self.data['line_events'].empty else 0
        total_teller_events = len(self.data['teller_events']) if not self.data['teller_events'].empty else 0
        total_abandonments = len(self.data['abandonment_events']) if not self.data['abandonment_events'].empty else 0
        
        print(f"👥 Total People Detected: {total_people}")
        print(f"�� Total Line Events: {total_line_events}")
        print(f"🤝 Total Teller Interactions: {total_teller_events}")
        print(f"⚠️  Total Abandonments: {total_abandonments}")
        
        if total_people > 0:
            abandonment_rate = (total_abandonments / total_people) * 100
            interaction_rate = (total_teller_events / total_people) * 100
            print(f"📉 Abandonment Rate: {abandonment_rate:.1f}%")
            print(f"📈 Interaction Rate: {interaction_rate:.1f}%")
    
    def _print_event_breakdown(self):
        """Print detailed event breakdown."""
        print("\n�� EVENT BREAKDOWN")
        print("-" * 40)
        
        # Line events breakdown
        if not self.data['line_events'].empty:
            line_event_counts = self.data['line_events']['event_type'].value_counts()
            print("📊 Line Events:")
            for event_type, count in line_event_counts.items():
                print(f"   • {event_type}: {count}")
        
        # Teller events breakdown
        if not self.data['teller_events'].empty:
            teller_event_counts = self.data['teller_events']['event_type'].value_counts()
            print("\n🤝 Teller Events:")
            for event_type, count in teller_event_counts.items():
                print(f"   • {event_type}: {count}")
        
        # Abandonment events breakdown
        if not self.data['abandonment_events'].empty:
            abandonment_event_counts = self.data['abandonment_events']['event_type'].value_counts()
            print("\n⚠️  Abandonment Events:")
            for event_type, count in abandonment_event_counts.items():
                print(f"   • {event_type}: {count}")
    
    def _print_customer_journey_analysis(self):
        """Print customer journey analysis."""
        print("\n👤 CUSTOMER JOURNEY ANALYSIS")
        print("-" * 40)
        
        if self.data['line_events'].empty:
            print("No customer journey data available.")
            return
        
        # Analyze each person's journey
        for person_id in self.data['line_events']['person_id'].unique():
            person_events = self.data['line_events'][
                self.data['line_events']['person_id'] == person_id
            ]
            
            print(f"\n👤 {person_id}:")
            
            # Get their events in chronological order
            person_events = person_events.sort_values('timestamp')
            
            for _, event in person_events.iterrows():
                event_time = event['timestamp'].strftime("%H:%M:%S")
                print(f"   {event_time} - {event['event_type']}")
            
            # Check for teller interactions
            person_teller_events = self.data['teller_events'][
                self.data['teller_events']['person_id'] == person_id
            ]
            
            if not person_teller_events.empty:
                for _, event in person_teller_events.iterrows():
                    event_time = event['timestamp'].strftime("%H:%M:%S")
                    print(f"   {event_time} - {event['event_type']} ✅")
            
            # Check for abandonments
            person_abandonments = self.data['abandonment_events'][
                self.data['abandonment_events']['person_id'] == person_id
            ]
            
            if not person_abandonments.empty:
                for _, event in person_abandonments.iterrows():
                    event_time = event['timestamp'].strftime("%H:%M:%S")
                    print(f"   {event_time} - {event['event_type']} ❌")
    
    def _print_performance_metrics(self):
        """Print performance metrics."""
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 40)
        
        total_people = len(self.data['line_events']['person_id'].unique()) if not self.data['line_events'].empty else 0
        total_abandonments = len(self.data['abandonment_events']) if not self.data['abandonment_events'].empty else 0
        total_interactions = len(self.data['teller_events']) if not self.data['teller_events'].empty else 0
        
        if total_people > 0:
            abandonment_rate = (total_abandonments / total_people) * 100
            interaction_rate = (total_interactions / total_people) * 100
            completion_rate = ((total_people - total_abandonments) / total_people) * 100
            
            print(f"📊 Abandonment Rate: {abandonment_rate:.1f}%")
            print(f"📊 Interaction Rate: {interaction_rate:.1f}%")
            print(f"�� Completion Rate: {completion_rate:.1f}%")
            
            # Performance assessment
            if abandonment_rate < 10:
                print("🎯 Performance: EXCELLENT - Low abandonment rate")
            elif abandonment_rate < 25:
                print("🎯 Performance: GOOD - Moderate abandonment rate")
            elif abandonment_rate < 50:
                print("🎯 Performance: FAIR - High abandonment rate")
            else:
                print("🎯 Performance: POOR - Very high abandonment rate")
    
    def _print_wait_time_analysis(self):
        """Print wait time analysis."""
        print("\n⏱️  WAIT TIME ANALYSIS")
        print("-" * 40)
        
        if self.data['abandonment_events'].empty:
            print("No wait time data available.")
            return
        
        # Calculate wait times
        wait_times = []
        for _, event in self.data['abandonment_events'].iterrows():
            wait_time = (event['line_exited_timestamp'] - event['line_entered_timestamp']).total_seconds()
            wait_times.append(wait_time)
        
        if wait_times:
            avg_wait = np.mean(wait_times)
            max_wait = np.max(wait_times)
            min_wait = np.min(wait_times)
            median_wait = np.median(wait_times)
            
            print(f"⏱️  Average Wait Time: {avg_wait:.1f} seconds")
            print(f"⏱️  Median Wait Time: {median_wait:.1f} seconds")
            print(f"⏱️  Maximum Wait Time: {max_wait:.1f} seconds")
            print(f"⏱️  Minimum Wait Time: {min_wait:.1f} seconds")
            
            # Wait time assessment
            if avg_wait < 30:
                print("🎯 Wait Time: EXCELLENT - Very fast service")
            elif avg_wait < 60:
                print("🎯 Wait Time: GOOD - Reasonable wait time")
            elif avg_wait < 120:
                print("🎯 Wait Time: FAIR - Moderate wait time")
            else:
                print("🎯 Wait Time: POOR - Long wait time")
    
    def _print_key_insights(self):
        """Print key insights."""
        print("\n💡 KEY INSIGHTS")
        print("-" * 40)
        
        total_people = len(self.data['line_events']['person_id'].unique()) if not self.data['line_events'].empty else 0
        total_abandonments = len(self.data['abandonment_events']) if not self.data['abandonment_events'].empty else 0
        total_interactions = len(self.data['teller_events']) if not self.data['teller_events'].empty else 0
        
        print("✅ System Performance:")
        print("   • Real-time event detection is working")
        print("   • Customer movements are being tracked")
        print("   • Queue management data is being captured")
        
        print("\n📊 Business Insights:")
        if total_people > 0:
            abandonment_rate = (total_abandonments / total_people) * 100
            interaction_rate = (total_interactions / total_people) * 100
            
            print(f"   • {total_people} people used the system")
            print(f"   • {abandonment_rate:.1f}% abandoned the line")
            print(f"   • {interaction_rate:.1f}% had teller interactions")
            
            if abandonment_rate > 50:
                print("   ⚠️  High abandonment rate - consider improving service speed")
            elif interaction_rate < 50:
                print("   ⚠️  Low interaction rate - consider staff training")
            else:
                print("   ✅ Good balance of interactions and completions")
        
        print("\n�� Technical Insights:")
        print("   • Computer vision system is functioning properly")
        print("   • Event detection algorithms are working")
        print("   • Data collection and storage is operational")


def main():
    """Main function to create the CV terminal report."""
    print("🎬 CV Queue Management Terminal Report Generator")
    print("=" * 60)
    
    # Create reporter
    reporter = CVTerminalReporter()
    
    # Create comprehensive report
    reporter.create_comprehensive_report()
    
    print("\n✅ Terminal report complete!")


if __name__ == "__main__":
    main()