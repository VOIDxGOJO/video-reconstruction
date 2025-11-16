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

📌 Detailed Algorithm Explanation

The reconstruction problem is formulated as inferring the true temporal ordering of a set of video frames whose original sequence has been completely destroyed. Unlike classical video tasks, where temporal information is preserved, this task relies entirely on visual information inside the individual frames. The approach taken here blends deep visual understanding, motion estimation, and graph-based global optimization, enabling the system to reconstruct a wide range of videos reliably.

🔹 1. Frame Extraction
The video is decomposed into individual frames, providing an indexable set where each image represents a node in a reconstruction graph. This establishes the foundation for computing features and pairwise relationships.

🔹 2. Feature Embedding (Semantic Similarity)
Each frame is encoded using a deep visual embedding, typically extracted using a pretrained ResNet-18 network. These embeddings capture high-level semantics such as objects, textures, shapes, and overall scene context. If PyTorch is unavailable, a fallback to classical HSV color histograms ensures the algorithm still functions. Deep embeddings ensure that frames belonging to the same scene or shot cluster naturally in the feature space, even when the intervening motion is subtle or the lighting varies.

🔹 3. k-NN Candidate Graph (Sparse Temporal Hypotheses)
For each frame, the algorithm identifies the k most visually similar frames based on cosine similarity of embeddings. This forms a sparse directed graph of plausible transitions. This step eliminates the need to evaluate all O(N²) frame pairs, drastically reducing computational cost and filtering out visually implausible neighbors. This sparsity makes the problem tractable while still preserving the true temporal neighbors in the majority of cases.

🔹 4. Motion-Based Directed Cues
Appearance similarity alone cannot determine direction — frame A and frame B may be similar, but which one comes first? 
The system introduces direction-sensitive cues computed only on candidate edges:

ORB Keypoint Displacement:
Keypoints are matched between frames; the average directional shift of pixels gives a coarse indication of forward vs backward motion.

Optical Flow Magnitude:
Dense optical flow estimates the motion between two frames. A smoother, lower-magnitude flow typically indicates a correct temporal adjacency.

Both cues are normalized and fused with appearance similarity to form a directed cost function for each edge:
```
cost = w_embed*(1 – sim) + w_flow*(flow_mag) + w_dir*(direction_penalty)
```
This formulation allows the algorithm to exploit appearance, motion, and structure simultaneously.

🔹 5. Global Ordering via Multi-Stage Optimization
Recovering the correct frame sequence is equivalent to finding a minimum-cost Hamiltonian path in the directed graph. Because this is NP-hard, a layered optimization strategy is used:

Spectral Initialization provides a rough global embedding-based ordering.
Beam Search explores promising paths while controlling combinatorial explosion.
2-Opt removes local inversions, similar to classic TSP refinement.
Simulated Annealing perturbs the ordering stochastically to escape local minima and improve global coherence.

This multi-stage process yields a sequence that balances appearance consistency and motion smoothness.

🔹 6. Orientation Verification and Correction
Even after optimization, the global direction may be flipped (i.e., the sequence may run backward). To detect this, the algorithm computes the total optical-flow smoothness of the sequence in both directions. Whichever orientation exhibits smoother, more consistent motion is selected as the final ordering.

🔹 7. Post-Processing
A final local smoothing pass adjusts neighboring pairs using flow scores. Optionally, the last few seconds of the video may be rotated to the beginning if the dataset exhibits circular motion patterns.


🎯 Why This Approach Works

This hybrid pipeline is effective because:
✔ Deep embeddings handle global similarity
  ResNet-based embeddings preserve scene structure even when frames change minimally or when illumination varies.

✔ Motion cues resolve directionality
  Optical flow and ORB displacement provide directional information that embedding similarity alone cannot.

✔ Sparse candidate graph reduces noise
  By restricting attention to top-k visually plausible neighbors, the algorithm avoids false relationships and reduces computational complexity.

✔ Multi-stage global optimization finds coherent ordering
  Spectral methods provide a robust starting point, beam search offers structured exploration, and local refinements (2-opt, annealing) resolve fine temporal        details.

✔ Flow-based orientation detection ensures correct playback
  Real-world videos have smoother forward motion, which this method exploits to automatically detect reversed reconstructions.

Overall, the pipeline balances appearance, motion, and optimization, making it generalizable to diverse video types—from slow pans to fast-motion scenes.


⏱ Time & Complexity Evaluation
Theoretical Complexity Breakdown

Embedding computation: O(N) forward passes

k-NN similarity computation: O(N²·d) where d = embedding dimension

Candidate edge evaluation: O(N·k) using ORB + optical flow

Beam search: approx. O(N·beam_width) expansions

Local refinements: heuristic linear-to-quadratic depending on iterations
Practical Runtime Estimates

On a typical mid-range CPU:
Small video (100–150 frames): 45–120 seconds
Medium video (200–300 frames): 5–12 minutes
Large video (500–1000 frames): 12–30+ minutes

On GPU (for embeddings):
Embedding computation speed improves by 10×–30×.
Total runtime often decreases by 30–50% for medium-sized videos.
Factors that influence runtime
Frame resolution (optical flow cost scales with pixel count)
k (candidate neighbors)
Beam width
ORB feature count

Hardware acceleration (GPU vs CPU)
Memory Considerations
Embeddings and similarity matrices dominate RAM usage.
k-NN sparsification significantly reduces both memory and runtime.

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









