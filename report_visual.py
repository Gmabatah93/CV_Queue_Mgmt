#!/usr/bin/env python3
"""
Complete CV Queue Management Graph Generator
Shows ALL event types including abandonments and teller interactions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

def create_complete_graph():
    """Create a complete graph showing all event types."""
    
    # Load the latest CSV files
    csv_dir = Path("data/csv_exports")
    
    # Find latest files
    line_files = list(csv_dir.glob("line_events_*.csv"))
    teller_files = list(csv_dir.glob("teller_interaction_events_*.csv"))
    abandonment_files = list(csv_dir.glob("abandonment_events_*.csv"))
    
    if not line_files:
        print("❌ No CSV files found!")
        return
    
    # Load all data
    latest_line_file = max(line_files, key=lambda f: f.stat().st_mtime)
    df_line = pd.read_csv(latest_line_file)
    
    df_teller = pd.DataFrame()
    if teller_files:
        latest_teller_file = max(teller_files, key=lambda f: f.stat().st_mtime)
        df_teller = pd.read_csv(latest_teller_file)
    
    df_abandonment = pd.DataFrame()
    if abandonment_files:
        latest_abandonment_file = max(abandonment_files, key=lambda f: f.stat().st_mtime)
        df_abandonment = pd.read_csv(latest_abandonment_file)
    
    # Combine all event types
    all_events = []
    
    # Add line events
    for _, event in df_line.iterrows():
        all_events.append(event['event_type'])
    
    # Add teller interactions
    for _, event in df_teller.iterrows():
        all_events.append(event['event_type'])
    
    # Add abandonment events
    for _, event in df_abandonment.iterrows():
        all_events.append(event['event_type'])
    
    # Count all events
    event_counts = pd.Series(all_events).value_counts()
    
    # Get current date for title
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Create the graph
    plt.figure(figsize=(12, 8))
    
    # Create bar chart with custom colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95']
    bars = plt.bar(event_counts.index, event_counts.values, color=colors[:len(event_counts)])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Customize the graph with date in title
    plt.title(f'CV Queue Management - Complete Event Summary\n{current_date}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Event Type', fontsize=12)
    plt.ylabel('Number of Events', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add summary text
    total_people = len(df_line['person_id'].unique())
    total_events = len(all_events)
    
    plt.figtext(0.5, 0.02, 
                f'Total People: {total_people} | Total Events: {total_events}',
                ha='center', fontsize=10, style='italic')
    
    # Save the graph
    plt.tight_layout()
    plt.savefig('complete_cv_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Complete graph saved as 'complete_cv_summary.png'")
    
    # Show the graph
    plt.show()
    
    # Print detailed summary
    print(f"\n📊 Complete Summary:")
    print(f"👥 Total People: {total_people}")
    print(f"📈 Total Events: {total_events}")
    print(f"\n📋 Event Breakdown:")
    for event_type, count in event_counts.items():
        print(f"   • {event_type}: {count}")
    
    # Calculate abandonment rate
    if not df_abandonment.empty:
        abandonment_rate = (len(df_abandonment) / max(total_people, 1)) * 100
        print(f"\n⚠️  Abandonment Rate: {abandonment_rate:.1f}%")

if __name__ == "__main__":
    create_complete_graph()