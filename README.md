
# AdaBoost Image Projects

This repository contains step-by-step projects demonstrating **AdaBoost** applied to images:

1. **MNIST Digit Classification** (AdaBoost with decision stumps)
2. **Face Detection** (OpenCV Haar Cascade, trained with AdaBoost internally) 
3. **Face vs Non-Face Classification** using **HOG features + AdaBoost** 

--- 
 
## Requirements   
 
Install dependencies:

```bash
pip install numpy matplotlib scikit-learn joblib opencv-python scikit-image
```

---

## Projects

### 1. MNIST Digit Classification

Trains AdaBoost with decision stumps on the MNIST dataset.

Run:

```bash
python adaboost_mnist.py
```

* Loads MNIST (70k images, 28x28)
* Normalizes pixels
* Trains AdaBoost
* Evaluates accuracy, confusion matrix
* Shows example predictions

---

### 2. Face Detection (Haar Cascade)

Uses OpenCV’s built-in Haar cascades (trained with AdaBoost) to detect faces.

#### Run on an image:

```bash
python face_detection.py path/to/image.jpg
```

#### Run with webcam:

```bash
python face_detection.py --webcam
```

* Press `q` to quit webcam mode.

---

### 3. Face vs Non-Face Classification with HOG + AdaBoost

Prepare dataset:

```
data/
  faces/      # cropped face images
  nonfaces/   # cropped non-face patches
```

Run:

```bash
python adaboost_hog_faces.py
```

* Extracts **HOG features**
* Trains AdaBoost classifier (face vs non-face)
* Prints accuracy and classification report

---

## Notes

* If using **scikit-learn ≥ 1.2**, the correct parameter is `estimator=` (already fixed in this code).
* For older scikit-learn (<1.2), replace with `base_estimator=` if needed.

---

##  Files

* `adaboost_mnist.py` → MNIST digit classification
* `face_detection.py` → Face detection with Haar cascades (image/webcam)
* `adaboost_hog_faces.py` → HOG features + AdaBoost face vs non-face

---

##  References

* [Scikit-learn AdaBoost Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [OpenCV Haar Cascades](https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html)
* [HOG Features (scikit-image)](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)


