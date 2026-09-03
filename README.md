# Custom Real-Time Computer Vision Pipeline (PyTorch & MPS)

A low-level computer vision implementation demonstrating raw tensor operations, hardware acceleration, and object detection on Apple Silicon without relying on high-level wrappers (e.g., YOLO/Ultralytics).

---

## Highlights

* **Raw Tensor Control:** Implemented custom image-to-tensor preprocessing, normalization, channel permutation `(H, W, C) -> (C, H, W)`, and batching operations from scratch.
* **Apple Silicon (MPS) Optimization:** Configured execution on Apple’s Metal Performance Shaders backend, managing explicit CPU $\leftrightarrow$ GPU tensor memory transfers for low-latency inference.
* **Zero High-Level Abstractions:** Utilized foundational `torchvision` Faster R-CNN primitives to maintain granular control over inference modes, score filtering, and class masking.
* **Vectorized Tensor Filtering:** Built mask-based logic using PyTorch bitwise operators `&` to filter detections directly on the GPU before transferring outputs to NumPy.

---

## Tech Stack & Concepts

* **Frameworks:** PyTorch, Torchvision, OpenCV, NumPy
* **Hardware Acceleration:** Apple Silicon MPS (`torch.device("mps")`)
* **Core Algorithms:** Faster R-CNN (ResNet-50 + FPN), Region Proposal Networks (RPN), Non-Maximum Suppression (NMS), Anchor Boxes

---

## Low-Level Pipeline Architecture

```text
[ Raw Frame ] ──> [ BGR to RGB ] ──> [ Float32 / 255.0 ] ──> [ Permute (C, H, W) ]
                                                                      │
                                                                      ▼
[ Bounding Boxes ] <── [ Bitwise Tensor Masking ] <── [ MPS GPU Inference ]
