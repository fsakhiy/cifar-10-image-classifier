# CIFAR-10 CNN Image Classifier Web UI

A Streamlit web application that lets you upload or select external images to be classified by your trained Convolutional Neural Network (CNN) `.keras` model.

---

## 🚀 Quick Start

### Using Make (Universal - Windows, macOS, Linux)

If you have `make` installed on your machine, it will automatically detect your OS:

```bash
# Start the app (auto-creates venv & installs dependencies if needed)
make run

# Other handy commands:
make setup     # Install/update dependencies
make samples   # Generate test sample images
make clean     # Clean python cache
make info      # Show detected OS and paths
```

---

### Manual Commands (Without Make)

#### Windows (PowerShell / Command Prompt)

```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run the Streamlit app
streamlit run app.py
```
*(Or in one command without activating: `.\.venv\Scripts\streamlit.exe run app.py`)*

---

### macOS & Linux (Terminal)

```bash
# 1. Create virtual environment (first time only)
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install requirements (first time only)
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Placing Your Trained Model (`cnn_cifar10.keras`)

When your Google Colab training finishes, you can load the model into the app in any of these 3 ways:

1. **Option A (Recommended)**: Copy `cnn_cifar10.keras` into the `model/` folder:
   ```text
   TK2-AI/
   └── model/
       └── cnn_cifar10.keras
   ```
2. **Option B (Automatic Downloads Detection)**: Since Google Colab automatically saves downloads to `C:\Users\fairuz\Downloads\cnn_cifar10.keras`, the app will detect and load it automatically if it's there!
3. **Option C (Web UI Uploader)**: Open the sidebar in the running Streamlit web app and drag-and-drop `cnn_cifar10.keras` into the model uploader.

---

## 🎨 Features

- **Side-by-Side Resolution Inspector**: Real-world images are resized to 32×32 pixels to match the CIFAR-10 CNN input. The app displays both the full-resolution uploaded image and the 32×32 pixelated view that the CNN evaluates.
- **Top Prediction Metric**: Highlights the predicted category with emoji and confidence percentage.
- **Interactive Probabilities Breakdown**: Displays a real-time horizontal bar chart of all 10 CIFAR-10 classes (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
- **Sample Images Included**: Pre-loaded external images in `samples/` for one-click testing without having to search for images online.

---

## 📁 Project Structure

```text
TK2-AI/
├── .venv/                         # Virtual environment (Python 3.12)
├── model/
│   └── cnn_cifar10.keras          # Model weights (place your trained model here)
├── samples/                       # Sample external images for testing
├── app.py                         # Streamlit web application
├── create_placeholder_model.py    # Generates initial model structure for testing
├── create_sample_images.py        # Generates sample test images
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```
