import numpy as np


def print_matrix(mat):
    for line in mat:
        print('  '.join(map(str, line)))
    print()


def input_matrix(matrix, n):
    return np.reshape(matrix, (n, n))


def read_data_from_txt_file(filename):
    file = open(filename, "r")
    tmp = file.readlines()
    TMPdata = []
    for line in tmp:
        TMPdata.append(int(line.strip()))
    return TMPdata


def conditional(mat, norm):
    inv = np.linalg.inv(np.array(mat))
    if norm == "fro":
        print(f"A norm:{norm} = {np.linalg.norm(mat, 'fro')}\nA^-1 norm:{norm} = {np.linalg.norm(inv, 'fro')}")
        return np.linalg.norm(mat, "fro") * np.linalg.norm(inv, "fro")

    if norm == "nuc":
        print(f"A norm:{norm} = {np.linalg.norm(mat, 'nuc')}\nA^-1 norm:{norm} = {np.linalg.norm(inv, 'nuc')}")
        return np.linalg.norm(mat, "nuc") * np.linalg.norm(inv, "nuc")

    if norm == "inf":
        print(f"A norm:{norm} = {np.linalg.norm(mat, np.inf)}\nA^-1 norm:{norm} = {np.linalg.norm(inv, np.inf)}")
        return np.linalg.norm(mat, np.inf) * np.linalg.norm(inv, np.inf)

    if norm == "-inf":
        print(f"A norm:{norm} = {np.linalg.norm(mat, -np.inf)}\nA^-1 norm:{norm} = {np.linalg.norm(inv, -np.inf)}")
        return np.linalg.norm(mat, -np.inf) * np.linalg.norm(inv, -np.inf)

    if norm == "1":
        print(f"A norm:{norm} = {np.linalg.norm(mat, 1)}\nA^-1 norm:{norm} = {np.linalg.norm(inv, 1)}")
        return np.linalg.norm(mat, 1) * np.linalg.norm(inv, 1)

    if norm == "-1":
        print(f"A norm:{norm} = {np.linalg.norm(mat, -1)}\nA^-1 norm:{norm} = {np.linalg.norm(inv, -1)}")
        return np.linalg.norm(mat, -1) * np.linalg.norm(inv, -1)

    if norm == "2":
        print(f"A norm:{norm} = {np.linalg.norm(mat, 2)}\nA^-1 norm:{norm} = {np.linalg.norm(inv, 2)}")
        return np.linalg.norm(mat, 2) * np.linalg.norm(inv, 2)

    if norm == "-2":
        print(f"A norm:{norm} = {np.linalg.norm(mat, -2)}\nA^-1 norm:{norm} = {np.linalg.norm(inv, -2)}")
        return np.linalg.norm(mat, -2) * np.linalg.norm(inv, -2)
    return "Error"


data = read_data_from_txt_file("data.txt")
A = input_matrix(data, int(np.sqrt(len(data))))
A_inv = np.linalg.inv(A)
print_matrix(A_inv)
B = np.array([1 for i in range(int(np.sqrt(len(data))))])
X = np.linalg.inv(A).dot(B)
print_matrix(A)
eig = np.linalg.eig(A.dot(A.transpose()))
print(eig)
print(X)
t = np.array([[65, 28], [28, 32]])
print_matrix(t)
norm = '2'
print(f"cond(A){norm}={conditional(A, norm)}")
