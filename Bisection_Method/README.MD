# 📌 Bisection Method in Python

## 📖 Overview

This project implements the **Bisection Method**, a classical numerical technique used to find **roots of nonlinear equations** within a given interval.

The program allows users to input a mathematical function and an interval where the root is expected. Using iterative halving of the interval, the algorithm approximates the root until the specified tolerance is reached.

This implementation was developed to support a Civil Engineering colleague in their Numerical Methods coursework. 

The program serves as a computational complement to the manual calculations typically performed when applying the Bisection Method, enabling faster verification of results, automated iteration tracking, and graphical analysis of the method's convergence.

---

## 🧠 About the Bisection Method

The **Bisection Method** is a root-finding algorithm that works on the **Intermediate Value Theorem**.

If a continuous function f(x) satisfies:

f(a) * f(b) < 0

then there exists at least one root in the interval [a,b].

The method repeatedly divides the interval in half and selects the subinterval where the sign change occurs, gradually converging to the root.

---

## ⚙️ Features

This implementation supports a wide variety of mathematical expressions, including:

- Polynomial functions
- Trigonometric functions
- Exponential functions
- Logarithmic functions
- Complex combinations of the above

The program asks the user to provide:

1. The function **f(x)** written in Python syntax
2. The interval **[a, b]** where the root is expected
3. The **tolerance** for the root approximation

---

## 🚀 Improvements Implemented

Beyond the basic algorithm, several improvements were added to make the program **more robust and user-friendly**:

- ✅ **Function validation** with clear error messages if the expression is invalid
- 📈 **Graph of the function** showing the located root
- ⚠️ **Handling special cases**, such as exact roots located at interval boundaries
- 🧾 **Detailed guidance** on how to write mathematical expressions correctly in Python
- 📊 **Improved output formatting**, including a well-structured iteration table
- 💾 **Automatic CSV export** storing:
  - general information about the problem
  - a complete iteration table
- 📉 **Convergence plot** showing approximate error vs iterations

These features allow users not only to compute the root but also to **analyze the convergence behavior of the method**.

---

## 📊 Outputs

The program generates the following outputs:

### 1️⃣ Iteration Table (Console)
Displays detailed iteration data including:

- iteration number
- interval bounds
- midpoint
- function evaluation
- approximate error

### 2️⃣ CSV File
A CSV file containing:

- problem information
- parameters used
- full iteration table

### 3️⃣ Function Plot
A graph of the function f(x) in the selected interval, including:

- the function curve
- the root location highlighted

### 4️⃣ Convergence Plot
A graph showing:

Error vs Iteration

This helps visualize how the method converges to the root.

---

## 📂 Example Problems

The implementation can solve problems such as:

- Polynomial equations
- Trigonometric equations
- Rational expressions
- Mixed transcendental functions

Examples tested include functions combining:

- logarithms
- exponentials
- sine functions
- polynomial expressions

---

## 🛠️ Technologies Used

- **Python**
- `matplotlib`
- `pandas`
- `csv`
- `numpy`

---

## 📈 Educational Purpose

This project was developed to reinforce concepts from Numerical Methods, combining:

- Mathematical theory

- Algorithmic implementation

- visualization of numerical convergence

It demonstrates how classical numerical algorithms can be implemented in Python to solve nonlinear equations efficiently.

