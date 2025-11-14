# 🎥 Jumbled Video Frame Reconstruction using Computer Vision & Deep Learning

This project reconstructs a **completely shuffled video** back into its **correct temporal order** using a combination of:

✔ Deep Learning (ResNet-18 embeddings)  
✔ Feature Matching (ORB keypoints)  
✔ Optical Flow  
✔ k-NN graph construction  
✔ Beam Search  
✔ 2-opt optimization  
✔ Simulated Annealing  
✔ Motion-flow–based orientation correction  
✔ Optional post-processing (reverse/rotate)  

This repository is designed to demonstrate strong skills in:

- Computer Vision  
- Optimization  
- Video Processing  
- Python development  
- Practical ML system design  

---

# 📌 Project Overview

Reconstructing video from shuffled frames is a challenging problem because:

- Consecutive frames may look very similar  
- Motion may be small or large  
- Scenes may contain static or dynamic regions  
- Frames must be arranged **globally**, not just pair-wise  

This project solves the problem **robustly** by combining multiple visual cues and optimization strategies.

---

# 🧠 Core Algorithm Pipeline

### 1️⃣ **Frame Extraction**
The script extracts all frames from the shuffled video into a temporary directory.

### 2️⃣ **Frame Embedding (Appearance Similarity)**
Two methods supported:

- **ResNet-18 pretrained (recommended)** → 512-dimensional embeddings  
- **HSV Histogram fallback** → used if PyTorch is unavailable  

Embeddings capture semantic similarity between frames.

---

### 3️⃣ **k-Nearest Neighbors Graph**
Using cosine similarity, each frame keeps its **top-k most similar neighbors**.  
This reduces the problem complexity from **O(N²)** to **O(N·k)**.

---

### 4️⃣ **Directional + Motion Cues**

#### ✔ ORB Keypoints  
Keypoint displacements estimate **direction of movement**.

#### ✔ Optical Flow  
Quantifies **motion smoothness** between frames.

These signals help distinguish between forward/backward adjacency.

---

### 5️⃣ **Directed Edge Cost**

For every candidate edge (i → j):

Lower = more likely to be the next frame.

---

### 6️⃣ **Global Ordering Optimization**

The ordering is solved as a **minimum-cost Hamiltonian path problem**, optimized through:

#### ✔ Spectral initialization  
Rough global ordering prediction using graph Laplacian.

#### ✔ Beam Search  
Explores only promising paths.

#### ✔ 2-Opt  
Local swap optimization used in TSP solvers.

#### ✔ Simulated Annealing  
Escapes local minima and smoothens ordering.

---

### 7️⃣ **Automatic Orientation Detection**
Using optical-flow smoothness:

- If reversed order is smoother → **auto-reverse**  
- If not → keep order

This ensures final playback is always forward-moving.

---

### 8️⃣ **Optional Video Reversal Script**
A second script allows manual final reversal:


---

# 📂 Repository Structure

video-reconstruction/
│
├── reconstruct_optimal.py # Main reconstruction algorithm
├── reverse_video_by_frames.py # Utility to reverse a video
├── requirements.txt # Dependencies
├── README.md # Documentation
├── jumbled_video.mp4 # (Input-video)

---

# ⚙️ Installation

## 1️⃣ Clone the repo
```bash
git clone https://github.com/VOIDxGOJO/video-reconstruction
cd video-reconstruction
```

2️⃣ Create virtual environment (Windows)
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

(Optional) Install PyTorch for ResNet embeddings
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

▶️ Usage
🧩 1. Reconstruct the jumbled video
```bash
python reconstruct_optimal.py --video jumbled_video.mp4 --out reconstructed_optimal.mp4
```

🔁 2. Reverse video using second script
```bash
python reverse_video_by_frames.py --input reconstructed_optimal.mp4 --out final_video.mp4
```

🧪 Example Result

Input: Completely shuffled frame order
Output: Smooth reconstructed, forward-moving video

✔ Frames globally sorted
✔ Motion continuity preserved
✔ Temporal consistency restored


🧭 Why This Approach Works
This solution combines:
Deep feature similarity
Motion information
Graph-based reasoning
Global optimization
Local refinement
Automatic direction correction
Which makes it accurate across many types of videos — indoor, outdoor, fast/slow motion, static scenes, etc.








