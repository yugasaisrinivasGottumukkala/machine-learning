# Lesson 01: Vectors and Operations on them

> **Why this is important:** Any data in machine learning is represented as vectors. A 28x28 image is represented as a vector of 784 numbers. A user is a vector of preferences. A word is an embedding vector. Without understanding vector operations,it's hard to understand what's happening inside any model.

---

## 1. What is a Vector?

In ML, a vector is simply an ordered list of numbers. perod. No need for philosophy about "directed line segments."

```
x = [3.0, -1.5, 2.7, 0.0, 5.1] # a 5-dimentional vector
```

The dimention of a vector is the number of elements in it. in real-world tasks, dimentions range from 2 (a point on a plane) to 4096(BERT embeddings) and beyond.

```python
import numpy as np
x = np.array([3.0, -1.5, 2.7, 0.0, 5.1])
print(x.shape) #(5,)
print(x.ndim) # 1  - one dimentional array
```

```

