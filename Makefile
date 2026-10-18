# ==============================================================================
# Cross-Platform Makefile (Windows, macOS, Linux)
# Auto-detects host OS and sets correct python/venv paths accordingly
# ==============================================================================

# OS Detection
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
	SYS_PYTHON := python
	VENV_DIR := .venv
	VENV_BIN := $(VENV_DIR)/Scripts
	PYTHON := $(VENV_BIN)/python.exe
	PIP := $(VENV_BIN)/pip.exe
	RM := del /Q /F
	RMDIR := rmdir /S /Q
else
	DETECTED_OS := $(shell uname -s)
	SYS_PYTHON := python3
	VENV_DIR := .venv
	VENV_BIN := $(VENV_DIR)/bin
	PYTHON := $(VENV_BIN)/python
	PIP := $(VENV_BIN)/pip
	RM := rm -f
	RMDIR := rm -rf
endif

.PHONY: all help run setup install clean info samples

# Default target
all: run

help:
	@echo ==========================================================
	@echo  CIFAR-10 Streamlit Classifier Makefile
	@echo  Host Detected: $(DETECTED_OS)
	@echo ==========================================================
	@echo  make run       - Run the Streamlit web application
	@echo  make setup     - Create venv and install all dependencies
	@echo  make samples   - Generate test sample images in samples/
	@echo  make clean     - Clean temporary Python cache files
	@echo  make info      - Show detected OS and environment paths
	@echo ==========================================================

info:
	@echo Detected OS: $(DETECTED_OS)
	@echo Python Path: $(PYTHON)
	@echo Venv Dir:    $(VENV_DIR)

# Create virtual environment if it does not exist
$(PYTHON):
	@echo ==> Creating virtual environment for $(DETECTED_OS)...
	$(SYS_PYTHON) -m venv $(VENV_DIR)

# Install dependencies into virtual environment
setup: $(PYTHON)
	@echo ==> Installing requirements for $(DETECTED_OS)...
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install: setup

# Generate sample images
samples: $(PYTHON)
	@echo ==> Generating sample images...
	$(PYTHON) create_sample_images.py

# Run the Streamlit web application
run: $(PYTHON)
	@echo ==> Starting Streamlit app on $(DETECTED_OS)...
	$(PYTHON) -m streamlit run app.py

# Clean cache files
clean:
ifeq ($(OS),Windows_NT)
	@if exist __pycache__ $(RMDIR) __pycache__
	@echo Cleaned cache.
else
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo Cleaned cache.
endif
