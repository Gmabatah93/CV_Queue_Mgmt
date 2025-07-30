
## 🎛️ Configuration

### Zone Configuration
The system uses configurable zones for different banking environments:

```python
# Teller Access Line (Queue Detection)
line_zone = {
    'y_position': 780,  # Horizontal line position
    'x_range': (300, 800),  # Line width
}

# Teller Interaction Zone (Service Detection)
teller_zone = {
    'x_range': (400, 700),  # Zone boundaries
    'y_range': (100, 300),
    'dwell_time': 2.0,  # Minimum interaction time (seconds)
}
```

### Environment Variables
```bash
# Required: Video file path
VIDEO_PATH=videos/bank_sample.MOV

# Optional: Output directory
OUTPUT_CSV_DIR=data/csv_exports

# Optional: Logging level
LOG_LEVEL=INFO
```

## 📈 Data Output & Analytics

### �� **CSV Export Files**
The system generates comprehensive event logs:

- **`line_events_YYYYMMDD_HHMMSS.csv`**
  - Line entry/exit events
  - Timestamp, person_id, event_type

- **`teller_interaction_events_YYYYMMDD_HHMMSS.csv`**
  - Teller service interactions
  - Dwell time analysis

- **`abandonment_events_YYYYMMDD_HHMMSS.csv`**
  - Customer abandonment events
  - Line entry/exit timestamps

### 🎯 **Key Metrics Tracked**
- **Queue Length**: Real-time line occupancy
- **Wait Times**: Individual customer wait durations
- **Service Times**: Teller interaction durations
- **Abandonment Rate**: Percentage of customers who leave without service
- **Throughput**: Customers served per time period

## 🔧 Development & Extension

### 🛠️ **Adding New Features**

1. **Custom Zone Detection**
   ```python
   # Add new zones in config/zone_config.py
   new_zone = Zone(
       zone_id="custom_zone",
       zone_type=ZoneType.CUSTOM,
       coordinates=[(x1,y1), (x2,y2), ...]
   )
   ```

2. **New Event Types**
   ```python
   # Extend event detection in inference/zone_detector.py
   def detect_custom_events(self, detections):
       # Add your custom event logic
       pass
   ```

3. **Dashboard Enhancements**
   ```python
   # Add new metrics in interface/streamlit_dashboard.py
   st.metric("Custom Metric", value)
   ```

### 🧪 **Testing & Quality**
```bash
# Run linting
ruff check --fix

# Run type checking
mypy premise_cv_platform/

# Run tests
pytest tests/
```

## 🎯 **Business Value**

### 🏦 **For Banking Operations**
- **Reduce Wait Times**: Identify bottlenecks and optimize staffing
- **Improve Service**: Monitor teller efficiency and customer satisfaction
- **Prevent Abandonments**: Early detection of service issues
- **Data-Driven Decisions**: Historical analysis for operational improvements

### 📊 **Key Insights Provided**
- **Peak Hours**: Identify busy periods for staffing optimization
- **Service Efficiency**: Monitor teller performance and interaction times
- **Customer Behavior**: Understand queue dynamics and abandonment patterns
- **Operational Metrics**: Real-time KPIs for management decisions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **YOLO**: For robust person detection capabilities
- **OpenCV**: For video processing and computer vision
- **Streamlit**: For interactive dashboard interface
- **Pydantic**: For data validation and type safety

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example configurations in `examples/`

---

**PREMISE CV Platform** - Transforming banking operations through intelligent computer vision analytics.

*From MVP to Enterprise: Building the future of queue management, one frame at a time.*