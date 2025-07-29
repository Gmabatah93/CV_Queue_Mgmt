## FEATURE: Video Upload and On-Demand Analysis (Dashboard Integration)

**User Story:**
* As a user, I can upload a local video file directly through the web dashboard, initiate its processing, and then view the analysis results within the dashboard once complete.

**Acceptance Criteria:**
1.  **Video Upload Interface:**
    * The `streamlit_dashboard.py` interface includes a clearly visible file uploader component (e.g., `st.file_uploader`) that accepts common video formats (e.g., `.MOV`, `.mp4`, `.mpg`).
    * The uploader validates the file type to ensure it's a supported video format.
2.  **Processing Initiation:**
    * Upon successful file upload, a prominent "Start Analysis" button (or similar) becomes available.
    * Clicking this button triggers the entire video processing pipeline defined in `main.py` (Ingestion, Inference, Data Storage).
3.  **Real-time Feedback (Processing Status):**
    * During video processing, the dashboard provides visual feedback to the user (e.g., a spinner, progress bar, or status messages like "Processing Frame X of Y") indicating that the analysis is in progress.
    * Logs from the processing pipeline (as defined in US-1 and US-2) should be displayed in real-time or near real-time on the dashboard or accessible via an expander.
4.  **Result Display after Processing:**
    * Once video processing is complete, the dashboard automatically updates and displays the analysis results using the metrics and visualizations defined in the existing `streamlit_dashboard.py` code.
    * The dashboard should clearly indicate that the displayed data corresponds to the newly uploaded and processed video.
5.  **Error Handling (Basic):**
    * If an error occurs during file upload or video processing, an informative error message is displayed on the dashboard (e.g., "Error: Video processing failed. Please check logs.").
6.  **Integration with Existing Pipeline:**
    * The dashboard's "Start Analysis" functionality seamlessly integrates with the existing `main.py` entry point or relevant pipeline functions, passing the uploaded video file as input.

## EXAMPLES:

* **`examples/sample_app.py`:** Use this as a conceptual reference for integrating a file upload widget and showing processing status within a Streamlit application. **Do NOT copy its backend logic (e.g., YouTube API calls, ChromaDB vectorstore creation) directly, as the PREMISE project has its own distinct processing pipeline and data storage.** Focus on the Streamlit frontend patterns for user interaction and dynamic content updates.
* **Screenshot Mockup (Optional, but highly recommended):** If possible, provide a simple wireframe or mockup image (`examples/dashboard_upload_mockup.png`) showing how the file upload section and processing feedback would appear on the `streamlit_dashboard.py` interface.

## DOCUMENTATION:
* **Streamlit File Uploader:** [https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader)
* **Streamlit Displaying Progress:** [https://docs.streamlit.io/library/api-reference/status/st.progress](https://docs.streamlit.io/library/api-reference/status/st.progress)
* **Streamlit Displaying Logs:** [https://docs.streamlit.io/library/api-reference/utilities/st.experimental_show_tokens](https://docs.streamlit.io/library/api-reference/utilities/st.experimental_show_tokens) (or similar for displaying dynamic text/logs)

## OTHER CONSIDERATIONS:
* **Temporary File Handling:** The uploaded video file should be saved to a temporary location on disk for processing by the `data_ingestion` layer, and ideally cleaned up after processing is complete.
* **Performance:** Given this is an MVP, initial focus is on functionality. Performance optimizations for large video files can be addressed in future iterations, but basic responsiveness should be maintained.
* **Backend Communication:** Consider how the Streamlit frontend will trigger and monitor the (potentially long-running) backend video processing task. This might involve threading, async operations, or simple process calls, ensuring the Streamlit app doesn't freeze.
* **Resource Management:** Ensure the processing of uploaded videos doesn't lead to excessive memory consumption or disk usage, especially with concurrent users (though MVP is likely single-user).