# SatDamage Assessment HUD

A Streamlit prototype for visualizing building damage assessment from satellite imagery. Displays pre/post disaster images with AI-powered damage classification and Grad-CAM activation maps.

## Features

- Pre-disaster vs post-disaster image comparison
- Grad-CAM activation map visualization
- Interactive damage overlay with hover tooltips
- Real-time threat level assessment
- Damage distribution statistics
- Multi-model metrics display

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment

1. Push this code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in and click "New app"
4. Select your repository and branch
5. Click "Deploy"

## Usage

1. Upload a **post-disaster satellite image** (required)
2. Optionally upload a **pre-disaster image** for comparison
3. Click "RUN ASSESSMENT" to analyze damage
4. Hover over buildings to see individual damage classifications

## Tech Stack

- **Frontend:** Streamlit
- **Visualization:** Custom HTML/CSS/JS HUD
- **ML Backend:** TensorFlow (integration ready)
- **Dataset:** xView2 format

## Team

Developed for satellite damage assessment pipeline.
