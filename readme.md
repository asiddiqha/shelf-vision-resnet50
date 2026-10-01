# ShelfVision: Automated Retail Stock Ingestion and Analysis Engine

ShelfVision is a computer vision application that processes retail inventory environments using a deep learning backbone. The service ingests camera snapshots of supermarket tiers to dynamically classify row capacity into fully stocked, low stock, or out-of-stock states, enabling automated inventory auditing.

## Interface Overview

### Initial Diagnostic State
The control panel initializes with a side-by-side management view. The left side handles standard image data streams, while the right console tracks system diagnostics.
![Initial Control Panel State](docs/initial_dashboard.png)

### Live Telemetry Extraction
Upon executing an image analysis loop, the system extracts the evaluated frame and renders real-time inventory signals alongside evaluation metrics.
![Live Telemetry Extraction](docs/telemetry_report.png)

## Architecture Evolution

### Phase 1: Deep Learning Backbone Transition
Initially drafted using MobileNetV2, the system was upgraded to a deeper ResNet50 model to enhance complex feature mapping. The network leverages transfer learning parameters from ImageNet, freezing lower layers while training the top dense layers on real-world retail textures to increase edge-case detection accuracy.

### Phase 2: Web Framework & Memory Architecture
The front-end utility is built using the FastAPI ecosystem paired with Jinja2 HTML rendering. To bypass high latency spikes caused by compiling massive weights on every network request, the application utilizes an internal lifespan event listener. This warms up and locks the 100MB+ model directly inside system RAM on boot, cutting prediction delays down to milliseconds.

## Directory Layout

```text
shelf_vision/
  ├── docs/                      # UI documentation assets
  │    ├── initial_dashboard.png
  │    └── telemetry_report.png
  ├── shelf_dataset/            # Image batches sorted by state
  │    ├── train/
  │    └── validation/
  ├── templates/
  │    └── index.html            # Minimalist pastel control center
  ├── .gitignore
  ├── app.py                     # High-speed memory-cached FastAPI server
  ├── download_data.py           # Programmatic Kaggle downloader script
  └── train_shelf_model.py       # Custom ResNet50 training pipeline
```

## Setup and Activation

1. Initialize your isolated Python runtime sandbox environment:
   python3 -m venv .venv
   source .venv/bin/activate

2. Install the target machine learning dependencies and web utilities:
   pip install tensorflow fastapi uvicorn python-multipart pillow numpy jinja2 scipy kaggle

3. Feed your Kaggle credentials into the automated download file and fetch the imagery batches:
   python download_data.py

4. Trigger the custom ResNet50 training pipeline to generate the neural net weights:
   python train_shelf_model.py

5. Initialize the high-speed local deployment framework:
   python app.py

Open your browser and navigate to http://127.0.0.1:8080 to access the visual panel.
