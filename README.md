<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Real-Time Object Detection Web App</h3>

  <p align="center">
    An interactive deep learning application that performs real-time object detection and bounding box annotation on live camera streams using PyTorch, MobileNetV3, OpenCV, and Streamlit.
    <br />
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#deployment--cloud-note">Deployment & Cloud Note</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project delivers a real-time computer vision pipeline wrapped in a clean web UI. Powered by PyTorch's **SSDLite320 MobileNet V3 Large** architecture, the application captures video frames, processes them through object detection neural networks, and overlays bounding boxes and confidence scores in real time.

Key features include:
* **Pre-trained SSD MobileNet V3 Architecture:** Lightweight, fast object detection optimized for edge devices and local hardware.
* **Streamlit Model Caching:** Uses `@st.cache_resource` to keep neural network weights loaded in memory, eliminating re-initialization lag.
* **Interactive Controls:** Adjustable confidence score threshold slider to dynamically filter low-probability detections.
* **Hardware Acceleration:** Automatic fallback supporting Apple Silicon Metal Performance Shaders (`mps`) and standard `cpu` processing.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python.org]][Python-url]
* [![PyTorch][PyTorch.org]][PyTorch-url]
* [![OpenCV][OpenCV.org]][OpenCV-url]
* [![Streamlit][Streamlit.io]][Streamlit-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow these steps to set up and run the application locally on your machine.

### Prerequisites

Ensure you have **Python 3.10 or 3.11** installed on your system.

* **macOS Dependency Note:** If using `pyenv` on macOS, ensure `xz` is installed via Homebrew to prevent `_lzma` module errors during PyTorch initialization:
  ```sh
  brew install xz
  ```

### Installation

1. **Clone the repository:**
   ```sh
   git clone [https://github.com/rabraham360/realtime-object-detection.git](https://github.com/rabraham360/realtime-object-detection.git)
   cd realtime-object-detection
   ```

2. **Install project dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Option 1: Run the Interactive Web Application (Local Streamlit)
Launch the interactive web dashboard on your local machine to access real-time webcam feed controls and confidence threshold sliders:
```sh
streamlit run app.py
```

### Option 2: Run Standalone OpenCV Script
Execute the raw computer vision script directly in your terminal using native OpenCV windowing (`cv2.imshow`):
```sh
python3 Main.py
```
*Press `q` on your keyboard to stop the camera stream.*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Baseline OpenCV real-time detection pipeline
- [x] Local Streamlit Web UI integration
- [x] Model caching via `@st.cache_resource`
- [x] Dynamic confidence score slider
- [ ] Map numeric COCO class IDs to human-readable labels (`person`, `car`, etc.)
- [ ] Add live FPS (Frames Per Second) performance overlay counter
- [ ] Implement image file uploading for offline batch analysis

See the [open issues](https://github.com/rabraham360/realtime-object-detection/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions make the open-source community an incredible place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Project Link: [https://github.com/rabraham360/realtime-object-detection](https://github.com/rabraham360/realtime-object-detection)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [PyTorch Torchvision Models](https://pytorch.org/vision/stable/models.html)
* [OpenCV Documentation](https://docs.opencv.org/)
* [Streamlit Documentation](https://docs.streamlit.io/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch.org]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[OpenCV.org]: https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[Streamlit.io]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
