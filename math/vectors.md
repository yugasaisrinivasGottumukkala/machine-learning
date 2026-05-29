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

---

## 2. Addition and Scalar multiplication

**Vector Addition** is element-wise. Dimentons must match.

$$ a + b = (a_1 + b_1, a_2 + b_2, \dots, a_n + b_n) $$

**Scalar Multiplication** multiplies each element by a number or constant.

$$ \alpha \cdot a = (\alpha a_1, \alpha a_2, \dots, \alpha a_n)$$

```python 
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)       #[5 7 9]
print(3 * a)       #[3 6 9]
print(a - 2 * b)   #[-7 -8 -9]
```

**Intuition:** Addition is like "gluing" movements. if `a` is "a 3-meter step east" and `b` is "a 4-meter step north", then`a + b` is a diagonal step.

---

## 3. Dot Product

The most important operation in ML. Without exaggeration.

$$a \cdot b = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \dots +a_n b_n $$

Result -- **one number**(scalar).

```python
a =np.array([1, 2, 3])
b =np.array([4, 5, 6])

#Three equivalent ways
print(np.dot(a, b))  #32
print(a @b)          #32 -- operator @
print((a) *(b).sum()) #32
```

**Where is it in ML:**
-One neuron calculates `w . x + b` - is the dot product of the weights and the input.
-In attention you count `Q . K` - proximity of request to each key.
-In recommendation systems, user item gives an estimate.

---
