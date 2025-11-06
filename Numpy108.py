import numpy as np

scores = np.array([
    [85, 78, 92, 88, 91],
    [76, 83, 80, 75, 77],
    [90, 88, 94, 91, 89],
    [65, 70, 72, 68, 66],
    [80, 79, 85, 87, 82],
    [72, 69, 74, 70, 76]
])
print("Shape:", scores.shape)
print("Data Type:", scores.dtype)
print("Dimensions:", scores.ndim)
print("Student 3 scores:", scores[2])
print("Computer Science scores:", scores[:, 4])

student_avg = np.mean(scores, axis=1)
print("Student-wise averages:", student_avg)

subject_max = np.max(scores, axis=0)
subject_min = np.min(scores, axis=0)
print("Max per subject:", subject_max)
print("Min per subject:", subject_min)

ranking = np.argsort(-student_avg)
print("Ranking (Best to Worst):", ranking + 1)

std_subject = np.std(scores, axis=0)
print("Standard Deviation per Subject:", std_subject)

weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
weighted_scores = np.dot(scores, weights)
print("Weighted scores:", weighted_scores)

mean = np.mean(scores, axis=0)
std = np.std(scores, axis=0)
normalized_scores = (scores - mean) / std
print("Normalized Scores:\n", normalized_scores)

A = np.array([
    [85, 78, 92],
    [76, 83, 80],
    [90, 88, 94]
])

B = np.array([255, 239, 270])  # Target total score
x = np.linalg.solve(A, B)
print("New Weight Vector (x):", x)

B = np.array([
    [1, 0.85, 0.78],
    [0.85, 1, 0.82],
    [0.78, 0.82, 1]
])

B_inv = np.linalg.inv(B)
print("Inverse of correlation matrix:\n", B_inv)

cov_matrix = np.cov(scores.T)
print("Covariance Matrix:\n", cov_matrix)


