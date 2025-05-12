# Chipotle Portion Analysis

## Overview & Model Setup

This project investigates the potential capability of the fast-food chain Chipotle to control portion sizes at a certain location with only limited data. We assume that the data available to a Chipotle location is an employee schedule (i.e. a matrix of which employees have worked on which day) and an inventory (i.e. a vector of an ingredient's consumption on each day). For this project, we are interested in chicken consumption. Chipotle's website indicates that each chicken bowl should contain 113 grams of chicken. Suppose each of $n$ employees gives a portion $p_j \sim \mathcal{N}(113, 25)$, where $j = 1, 2, ..., n$. Every day $m \leq n$ employees are scheduled to work.

The schedule matrix $S$ is a Markov matrix where

$$
S_{ij} = \begin{cases}
    \frac{1}{m} & \text{if employee } j ${ works on day } i
    0 & \text{otherwise}
\end{cases}
$$

Then we can compute each day's expected portion size $y = Sp$ where $p = (p_1, p_2, ..., p_n)$.
