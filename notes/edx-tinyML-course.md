### **Model Compression Techniques**

##### **A. Pruning**

Removing less important connections (weights) or entire neurons from a trained network

-   **Process:**
    1.  Train a full-sized model to a good accuracy
    2.  Identify and remove (prune) weights/neurons based on a criterion, for instance, smallest magnitudes
-   **Benefits:**
    -   Reduced Model Size
    -   Faster Inference

##### **B. Quantization**

Discretizing the values (weights and activations) from high-precision floating-point numbers to a smaller set of lower-precision numbers.

-   **Process:** Mapping a range of FP32 values to a much smaller number of INT8 levels.
-   **Benefits:**
    -   Reduced Model Size
    -   Faster & More Efficient Inference

##### **C. Knowledge Distillation**

Training a small and efficient (student) model to replicate the behavior of a large, powerful (teacher) model

-   **Process:**
    1.  Train the large, accurate teacher model first
    2.  The student model is trained not just on the original "hard" labels, but also on the "soft" probabilities (knowledge) output by the teacher
-   **Benefits:**
    -   The small student model can often achieve higher accuracy than if it were trained on the data alone, as it learns the generalizing patterns from the teacher

In practice, these techniques are often combined to achieve the extreme levels of compression required for TinyML
