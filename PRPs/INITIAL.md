## OBJECTIVE:

**Objective:** Create a comprehensive Computer Vision Solution. We will start with a foundational MVP using sample `bank simulation` data and structure the project to scale up to the full banking solution later.

---

1.  **Project Vision & MVP Scope**

    * **Project Title:** "PREMISE" - A Scalable Computer Vision Analytics Platform
    * **Vision:** To create a modular computer vision system that translates raw video into actionable business intelligence, starting with general people-tracking and designed to evolve for specialized industry use cases like banking.
    * **MVP Goal:** To build a functional, end-to-end pipeline that can ingest a pre-recorded video of `bank simulation (someone leaving without interacting with teller)`, perform inference to identify and track individuals and their interactions within defined zones, store the resulting event data, and present key insights on a simple web dashboard. This will validate the core architecture.

---

2.  **Phase 1: MVP Requirements (Using Sample Data)**

    **US-1: Ingestion Layer**

    * As a developer, I can provide a local video file (e.g., `.MOV or .mpg`) to the system so that it can be processed frame by frame.
        * **Acceptance Criteria:** The system uses OpenCV to read the video file. The process is logged with the video's name and duration.

    **US-2: Inference Engine (MVP)**

    * As a developer, I can process the video to identify and track individuals.
    * As a system, I can identify when someone leaves the line without interacting with the teller
        * **Acceptance Criteria:** 
            1. *Person Detection:* The system successfully uses a pre-trained object detection model (e.g., `YOLO`) to detect individual people within the video frames.
            2. *Unique ID Assignment:* The system assigns and maintains a unique tracking ID for each detected person across consecutive frames, even with minor occlusions or movements.
            3. *Line Definition (Configurable):* The system allows for the configuration of a "line zone" (e.g., a defined polygonal area or horizontal line segment) within the video frame, representing the waiting line.
            4. *Teller Interaction Zone (Configurable):* The system allows for the configuration of a "teller interaction zone" (e.g., a defined polygonal area near the teller counter) within the video frame.
            5. *Line Entry/Exit Detection:* 
                - The system logs a "line_entered" event with a timestamp and unique person ID when a person's bounding box centroid enters the defined "line zone".
                - The system logs a "line_exited" event with a timestamp and unique person ID when a person's bounding box centroid leaves the defined "line zone".
            6. *Teller Interaction Detection:* The system logs a "teller_interacted" event with a timestamp and unique person ID when a person's bounding box centroid enters and remains within the "teller interaction zone" for a minimum configurable duration (e.g., 2 seconds).
            7. *"Left Line Without Interaction" Identification:* The system identifies and logs an event (e.g., "left_line_no_teller_interaction") with a timestamp and unique person ID when:
                - A "line_entered" event has been logged for that person, AND
                - A "line_exited" event is subsequently logged for that person, AND
                - No "teller_interacted" event has been logged for that person within the duration of their presence in the "line zone" or "teller interaction zone" (up to a configurable grace period after exiting the line zone)..

    **US-3: Data Storage (MVP)**

    * As a system, I can store the inference results in a simple, queryable format.
        * **Acceptance Criteria:** 
            1. *Storage Format:* Inference results are stored in one or more `.csv` files.
            2. *File Naming Convention:* `.csv` file(s) are named clearly (e.g., `events_YYYYMMDD_HHMMSS.csv` or separate files like `line_events.csv, abandonment_events.csv`).
            3. *Data Schema (CSV Headers):* Each `.csv` file includes appropriate headers defining the data columns. For example:
                - `person_tracking.csv`: `timestamp`, `event_type`, `person_id`, `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `frame_number`.
                - `line_events.csv`: `timestamp`, `event_type` (`line_entered`, `line_exited`), `person_id`, `line_zone_id` (if multiple zones).
                - `teller_interaction_events.csv`: `timestamp`, `event_type` (`teller_interacted`), `person_id`, `teller_zone_id` (if multiple tellers).
                - `abandonment_events.csv`: `timestamp`, `event_type` (`left_line_no_teller_interaction`), `person_id`, `line_entered_timestamp`,`line_exited_timestamp`.
            4. *Data Recording:* For each identified event (person detection, line entry/exit, teller interaction, abandonment), a corresponding row of data is successfully appended to the relevant CSV file.
            5. *Data Accessibility:* The stored data is directly accessible for viewing and manual analysis (e.g., can be opened in a spreadsheet program like Excel or Google Sheets).

    **US-4: Reporting Layer (MVP)**

    * As a user, I can view key operational insights on a simple, interactive web dashboard at the end of the video processing.
        * **Acceptance Criteria:**
            1.  *Dashboard Framework:* A Python-based web dashboard (using `Streamlit`) can be launched after video processing is complete to display the summary report.
            2.  *Data Source:* The dashboard reads the necessary data from the stored inference results (e.g., the CSV files from US-3).
            3.  *Dashboard Content - Key Metrics:* The dashboard accurately displays the following key operational insights:
                -   **Total number of unique individuals** detected and tracked throughout the entire video.
                -   **Total number of individuals who entered the defined "line zone"** at least once.
                -   **Total number of individuals who successfully interacted with a teller** (i.e., entered the "teller interaction zone" for the defined duration).
                -   **Total number of individuals who left the line without interacting with a teller** (i.e., had a 'left_line_no_teller_interaction' event logged).
            4.  *Interactivity (Basic):* The dashboard provides basic interactivity (e.g., displaying the metrics clearly, potentially a reset button for a new video analysis).
            5.  *Accessibility:* The dashboard is accessible via a local web browser after launching the Streamlit application.
            6.  *Example Dashboard Layout:* The dashboard's presentation of metrics should resemble a clear, concise summary:

                ```
                PREMISE - Video Processing Report
                ---------------------------------

                Video Processed: [Video File Name or ID]
                Processing Date: [Current Date and Time]

                Overall Metrics:
                ----------------
                Unique Individuals Tracked: [X unique individuals]
                Individuals Who Entered Line: [Y individuals]

                Line & Interaction Analysis:
                ----------------------------
                Successfully Interacted with Teller: [A individuals]
                Left Line Without Teller Interaction: [B individuals]
                ```

---

3. **Future Phases: Full Banking Solution Architecture**

    This section outlines the full-scale architecture to be built upon the MVP foundation. Use the high-level project details to fill this out.

    * **Ingestion Layer:** Describe the transition from local files to a robust system using RTSP streams, Kafka/MQTT, and secure edge servers for live bank camera feeds.
    * **Inference Engine:** Detail the evolution from simple group detection to the complex banking use cases (queue length, service time, etc.) using PyTorch and TensorRT for optimization.
    * **RAG Database:** Plan the transition from SQLite to a scalable system using PostgreSQL for structured data and a vector database (like FAISS or Pinecone) for unstructured insights.
    * **VLM Reporting Layer:** Design the final reporting system using LangGraph to orchestrate a VLM (like GPT-4) that queries the RAG database to produce insightful EOD reports and answer natural language queries via a web dashboard.

---

4. **Technical Stack & Project Structure**

* **Initial Stack (MVP):** `Python`, `OpenCV`, a pre-trained `YOLO` model (via a library like `ultralytics`), `streamlit` (for the frontend) **uv (as package handler)**.
* **Some Tech Stack:** 
    ### 🧰 🔧 Some Tech Stack Summary (By Layer)

    | Layer               | Tools/Frameworks                                          |
    | :------------------ | :-------------------------------------------------------- |
    | **Ingestion** | `RTSP/RTP`, `OpenCV`, `GStreamer`, `Kafka`, `MQTT`, Edge Servers, `VPN` |
    | **Inference Engine**| `Python/C++`, `OpenCV`, `supervision`, `PyTorch`, `TensorRT`, `NVIDIA GPUs`, `SQL` |
    | **RAG Database** | `PostgreSQL`, `InfluxDB`, `Pinecone`, `FAISS`, `Milvus` |
    | **VLM Reporting Layer** | `GPT-4`, `Llama 2`, `LangGraph`, `React`, `Dash`, `Streamlit` |


* **Project File Structure:**
    ```
    /premise_cv_platform
    |-- /data_ingestion
    |   |-- process_video.py
    |-- /inference
    |   |-- track_people.py
    |   |-- models/ # For the YOLO model
    |-- /storage
    |   |-- something.csv
    |-- /reporting
    |   |-- generate_summary.py
    |-- app.py # Orchestrates the pipeline
    |-- requirements.txt
    ```

---

5. **Action Plan**

    Generate the PRD based on the above structure. Focus heavily on detailing the **Phase 1 MVP requirements** first. For **Phase 2**, expand on my high-level notes to create a detailed architectural plan that we can follow after the MVP is complete. The goal is a document that guides our iterative development process clearly.


## EXAMPLES:

- Video Ingestion Log: `examples/ingestion_log.txt`
    - This example shows the expected format of the console output or log file after a video has been successfully ingested by the `process_video.py` script.
- Sample: `examples/line_events.csv`
    - Illustrates the structure and content of the CSV file storing line entry/exit events.
- Sample: `examples/teller_interaction_events.csv`
    - Illustrates the structure and content of the CSV file storing teller interaction events.
- Sample: `examples/abandonment_events.csv`
    - Illustrates the structure and content of the CSV file storing instances of people leaving the line without interacting with a teller.
- Final Report Summary: `examples/summary_report_sample.txt`
    - This example shows the expected text-based summary generated by the `generate_summary.py` script.

## DOCUMENTATION:

```
- file: examples/Project Management Feasibility_ Banking Computer Vision Solution.pdf
  why: This comprehensive research document provides in-depth context, market validation, detailed architectural considerations for future phases, data governance strategies, risk analysis, and operational planning specific to banking computer vision solutions. It will serve as the primary reference for expanding Section 3 and other high-level considerations.
- url:  https://docs.opencv.org/
  why: Specific modules for video I/O (VideoCapture) and drawing (rectangle, polylines): Refer to the Python API tutorials.
- url: https://docs.ultralytics.com/
  why: Guidance on object detection inference and tracking integration.
- url: https://huggingface.co/docs
  why: Datasets library documentation if any pre-existing datasets are used for local testing or validation.
- url: https://docs.astral.sh/uv/
  why: This resource provides comprehensive guides on uv's commands, virtual environment management, dependency resolution, and best practices.
```

## OTHER CONSIDERATIONS:
This section outlines additional requirements, best practices, and potential "gotchas" that are important for the project's success and maintainability, especially when working with AI coding assistants.

- **Environment Variables** (`.env.example` and `python-dotenv`):
    - A `.env.example` file must be included at the root of the project to clearly indicate expected environment variables. This includes:
        - **VIDEO_PATH:** Path to the local video file for processing.
        - **MODEL_NAME:** Name or path of the YOLO model to be used (e.g., yolov8n.pt).
        - **OUTPUT_CSV_DIR:** Directory where generated CSV files should be stored.
        - **LOG_LEVEL:** For configuring logging verbosity (e.g., INFO, DEBUG).
        - **Future Banking Specific:** (Examples for later phases) `RTSP_STREAM_URL`, `KAFKA_BROKER`, `POSTGRES_DB_URL`, `OPENAI_API_KEY`.
    - The `main.py` script and other modules should use `from dotenv import load_dotenv; load_dotenv()` to load these variables, ensuring sensitive information or configuration paths are not hardcoded.

- **README.md Instructions:**
    - The `README.md` file at the root of the project must be comprehensive. It should include:
        - **Project Overview:** A brief description of PREMISE and its MVP goal.
        - **Setup Instructions:**
            - How to clone the repository.
            - How to create and activate the virtual environment using `uv`:
                - Create venv: `uv venv`
                - Activate venv (macOS): `source ./.venv/bin/activate`
            - How to install dependencies using `uv`: `uv pip install -r requirements.txt`
            - **Crucially, instructions on how to set up the `.env` file from `.env.example.`**
        - **Running the MVP:** Step-by-step instructions on how to execute `main.py` and observe the results.
        - **Project File Structure:** Include the provided project file structure explicitly in the README for quick reference.
        
- **Virtual Environment & Dependencies:**
    - The project will exclusively use `uv` for all package management operations (installation, updates, dependency resolution).
 
- **Error Handling & Robustness (MVP Consideration):**
    - While an MVP, basic error handling should be in place (e.g., handling FileNotFoundError for video files, logging errors during inference).
    - Consider graceful shutdowns or recovery mechanisms for video processing if an error occurs mid-stream.

- **Logging Strategy:**
    - Implement consistent logging throughout the application (using Python's logging module) at appropriate levels (INFO, DEBUG, WARNING, ERROR). This aids in debugging and monitoring.
    - Ensure logs indicate progress (e.g., "Processing frame X of Y").

