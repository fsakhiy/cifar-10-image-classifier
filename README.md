# CIFAR-10 CNN Image Classifier Web UI

---

## Quick Start

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
