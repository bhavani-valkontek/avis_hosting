# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import os
# import time
# import tempfile
# import shutil
# from datetime import datetime
# import requests

# # ---------------------------#
# # ✅ Streamlit Page Config
# # ---------------------------#
# st.set_page_config(page_title="Upload Inspection Image", layout="centered")
# st.title("📷 Upload Inspection Image")

# # ---------------------------#
# # ✅ Define Model URLs (Hugging Face)
# # ---------------------------#
# MODEL_URLS = {
#     "Scratch": "https://huggingface.co/mahigodike/AVIS/resolve/main/scratch_model_v2.pt",
#     "Dent": "https://huggingface.co/mahigodike/AVIS/resolve/main/best_model_dentw.pt",
#     "Corrosion": "https://huggingface.co/mahigodike/AVIS/resolve/main/corrosion.pt",
#     "Windshield": "https://huggingface.co/mahigodike/AVIS/resolve/main/windshield.pt"
# }

# MODEL_DIR = "models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# # ---------------------------#
# # ✅ Download all models (cached)
# # ---------------------------#
# @st.cache_resource(show_spinner=True)
# def download_all_models():
#     downloaded_models = {}
    

#     for name, url in MODEL_URLS.items():
#         model_path = os.path.join(MODEL_DIR, f"{name.lower()}.pt")

#         if not os.path.exists(model_path):
#             try:
                
#                 with requests.get(url, stream=True) as r:
#                     r.raise_for_status()
#                     with open(model_path, "wb") as f:
#                         for chunk in r.iter_content(chunk_size=8192):
#                             if chunk:
#                                 f.write(chunk)
                
#             except Exception as e:
#                 st.sidebar.error(f"❌ Failed to download {name}: {e}")
#                 continue
#         else:
#             pass

#         downloaded_models[name] = model_path

#     st.sidebar.success("🎯 All models are ready!")
#     return downloaded_models


# # ✅ Preload models when app starts
# MODEL_PATHS = download_all_models()

# # ---------------------------#
# # ✅ Module setup
# # ---------------------------#
# active_modules = ["Scratch", "Dent", "Corrosion", "Windshield"]
# upcoming_modules = ["Tire", "Interior", "Underbody"]

# if "selected_modules" not in st.session_state:
#     st.warning("⚠️ Please go back and select modules first.")
#     st.stop()

# selected_modules = st.session_state.selected_modules

# if any(m in upcoming_modules for m in selected_modules):
#     upcoming = [m for m in selected_modules if m in upcoming_modules]
#     st.warning(f"⚙️ Coming soon: {', '.join(upcoming)}")
#     st.stop()

# # ---------------------------#
# # 📤 File Uploader
# # ---------------------------#
# uploaded_file = st.file_uploader("Upload vehicle image for inspection", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image")

#     temp_dir = tempfile.mkdtemp()
#     temp_path = os.path.join(temp_dir, uploaded_file.name)
#     image.save(temp_path)

#     if os.path.exists("runs"):
#         shutil.rmtree("runs")

#     if st.button("🔍 Run Detection"):
#         progress_bar = st.progress(0)
#         results = {}
#         os.makedirs("runs", exist_ok=True)

#         for i, module in enumerate(selected_modules):
#             progress_bar.progress((i + 1) / len(selected_modules))
#             model_path = MODEL_PATHS.get(module)

#             if not model_path or not os.path.exists(model_path):
#                 st.error(f"❌ Model not found for {module}")
#                 continue

#             model = YOLO(model_path)

#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             result_name = f"{module.lower()}_{timestamp}"
#             results_dir = f"runs/{result_name}"

#             preds = model.predict(
#                 source=temp_path,
#                 save=True,
#                 project="runs",
#                 name=result_name,
#                 exist_ok=False,
#                 verbose=False
#             )

#             detections = sum(len(r.boxes) for r in preds)
#             files = os.listdir(results_dir)
#             result_image_path = os.path.join(results_dir, files[0]) if files else None

#             results[module] = {
#                 "Detections": detections,
#                 "Result Path": result_image_path
#             }

#             time.sleep(0.2)

#         progress_bar.empty()
#         st.success("✅ Detection completed successfully!")

#         st.session_state.inspection_results = results
#         st.session_state.uploaded_image = image

#         time.sleep(2)
#         st.switch_page("pages/report.py")

import streamlit as st
import os

# ✅ Fix Ultralytics writable directory issue
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import tempfile
from datetime import datetime
import requests

# ---------------------------#
# ✅ Streamlit Page Config
# ---------------------------#
st.set_page_config(page_title="Upload Inspection Image", layout="centered")
st.title("📷 Upload Inspection Image")

# ---------------------------#
# ✅ Define Model URLs
# ---------------------------#
MODEL_URLS = {
    "Scratch": "https://huggingface.co/mahigodike/AVIS/resolve/main/scratch_model_v2.pt",
    "Dent": "https://huggingface.co/mahigodike/AVIS/resolve/main/best_model_dentw.pt",
    "Corrosion": "https://huggingface.co/mahigodike/AVIS/resolve/main/corrosion.pt",
    "Windshield": "https://huggingface.co/mahigodike/AVIS/resolve/main/windshield.pt"
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------#
# ✅ Download Models
# ---------------------------#
@st.cache_resource(show_spinner=True)
def download_all_models():
    downloaded_models = {}

    for name, url in MODEL_URLS.items():

        model_path = os.path.join(
            MODEL_DIR,
            f"{name.lower()}.pt"
        )

        if not os.path.exists(model_path):

            try:
                with requests.get(url, stream=True) as r:

                    r.raise_for_status()

                    with open(model_path, "wb") as f:

                        for chunk in r.iter_content(
                            chunk_size=8192
                        ):
                            if chunk:
                                f.write(chunk)

            except Exception as e:
                st.error(f"❌ Failed to download {name}: {e}")
                continue

        downloaded_models[name] = model_path

    return downloaded_models


MODEL_PATHS = download_all_models()

# ---------------------------#
# ✅ Module Setup
# ---------------------------#
active_modules = [
    "Scratch",
    "Dent",
    "Corrosion",
    "Windshield"
]

upcoming_modules = [
    "Tire",
    "Interior",
    "Underbody"
]

if "selected_modules" not in st.session_state:
    st.warning("⚠️ Please select modules first.")
    st.stop()

selected_modules = st.session_state.selected_modules

if any(
    m in upcoming_modules
    for m in selected_modules
):
    upcoming = [
        m for m in selected_modules
        if m in upcoming_modules
    ]

    st.warning(
        f"⚙️ Coming soon: {', '.join(upcoming)}"
    )

    st.stop()

# ---------------------------#
# 📤 Upload Image
# ---------------------------#
uploaded_file = st.file_uploader(
    "Upload vehicle image for inspection",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ✅ Save temp image
    temp_dir = tempfile.mkdtemp()

    temp_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    image.save(temp_path)

    # ---------------------------#
    # 🔍 Run Detection
    # ---------------------------#
    if st.button("🔍 Run Detection"):

        progress_bar = st.progress(0)

        results = {}

        for i, module in enumerate(selected_modules):

            progress_bar.progress(
                (i + 1) / len(selected_modules)
            )

            model_path = MODEL_PATHS.get(module)

            if (
                not model_path or
                not os.path.exists(model_path)
            ):
                st.error(
                    f"❌ Model not found for {module}"
                )
                continue

            # ✅ Load YOLO model
            model = YOLO(model_path)

            # ✅ Unique folder name
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            result_name = (
                f"{module.lower()}_{timestamp}"
            )

            # ✅ Run prediction
            preds = model.predict(
                source=temp_path,
                save=True,
                project="runs",
                name=result_name,
                exist_ok=True,
                verbose=False
            )

            # ✅ REAL output directory
            results_dir = preds[0].save_dir

            # ✅ Count detections
            detections = sum(
                len(r.boxes)
                for r in preds
            )

            # ✅ Get result image safely
            image_extensions = (
                ".jpg",
                ".jpeg",
                ".png"
            )

            image_files = [

                f for f in os.listdir(results_dir)

                if f.lower().endswith(
                    image_extensions
                )
            ]

            result_image_path = None

            if image_files:

                result_image_path = os.path.join(
                    results_dir,
                    image_files[0]
                )

            # ✅ Store results
            results[module] = {

                "Detections": detections,

                "Result Path": result_image_path
            }

            time.sleep(0.2)

        progress_bar.empty()

        st.success(
            "✅ Detection completed successfully!"
        )

        # ✅ Store session data
        st.session_state.inspection_results = results
        st.session_state.uploaded_image = image

        time.sleep(2)

        st.switch_page("pages/report.py")
