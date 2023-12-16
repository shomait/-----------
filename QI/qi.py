
import numpy as np
from sample_matrix import MatrixSampler

def qi_solver(A, b, d, n_iter):
    """
    Solves for x in the least squares problem: argmin_x ||Ax - b|| based on arXiv: 2103.10309.
    
    This function seeks to find a vector 'y' such that 'x = A^T y' provides the solution. 
    
    Parameters:
    - A: Matrix involved in the least squares problem.
    - b: Vector representing the target values in the least squares problem.
    - d: The number of samples used for approximating inner products. 
         Increasing 'd' can improve the approximation accuracy.
         d = O(kappa_F^2/epsilon^2) where kappa_F = ||A||_F/||A^+|| and epsilon is error is recommended in the reference.
    - n_iter: The number of iterations to perform in the algorithm.
              More iterations can lead to a more accurate solution but take longer to compute.
              n = O(kappa_F^2 log(1/epsilon)) is recommended.
    """
    sampler = MatrixSampler(A)
    y0 = np.zeros(A.shape[0]) #はじめは0ベクトルから始める
    y_history = [y0]

    for i in range(n_iter):
        row_index = sampler.sample_row_index() #確率的に1個の行成分をサンプリングする
        e = np.zeros_like(y0)
        """
        $$
        \mathbf{y}_{k+1}=\mathbf{y}_k+\frac{1}{\left\|A_{r_k *}\right\|}\left(\tilde{b}_{r_k}-\frac{1}{d} \sum_{j \in S} \tilde{A}_{r_k, j}\left\langle A_{* j} \mid \mathbf{y}_k\right\rangle \frac{\|A\|_F^2}{\left\|A_{* j}\right\|^2}\right) \mathbf{e}_{r_k}
        $$
        """
        e[row_index] = 1.
        if d != np.inf:
            col_indices = sampler.sample_col_index(d) #確率的にd個の列成分をサンプリングする
            #解を反復的に更新する
            y_new = y_history[-1] + \
                1/sampler.row_norms[row_index] * \
            (
                b[row_index] -  \
                (
                    np.sum(
                        A[row_index, col_indices]*
                        y_history[-1].dot(A[:,col_indices])*
                        sampler.frobenius_norm / sampler.col_norms[col_indices]
                    )
                ) / d
            ) * e
        else:
            y_new = y_history[-1] + \
                1/sampler.row_norms[row_index] * \
            (
                b[row_index] -
                        A[row_index, :].dot(np.transpose(A) @ y_history[-1])
            ) * e
        # print(row_index, y_new)
        y_history.append(y_new)

    return y_history

def generate_row_sparse_matrix(rows, cols, sparsity):
    matrix = np.zeros((rows, cols))
    for i in range(rows):
        non_zero_elements = np.random.choice(range(cols), size=int(sparsity * cols), replace=False)
        matrix[i, non_zero_elements] = np.random.rand(len(non_zero_elements))
    return matrix

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import quantum_inspired as qi
    # Step 1: Create test matrix A and vector b
    # np.random.seed(0)  # for reproducibility
    rng = np.random.default_rng(seed=1)
    #A = rng.random((100, 50))  # example dimensions
    A = generate_row_sparse_matrix(100, 50, 0.1)  # 100行50列、各行の10%の要素だけが非ゼロ
    b = rng.random(100)

    # Step 2: Find the true solution using pseudo-inverse
    start_time = time.time()  # 計算開始時間
    x_true = np.linalg.pinv(A) @ b
    end_time = time.time()  # 計算終了時間
    print("Time for exact solution:", end_time - start_time)

    # Step 3: Run qi_solver to get approximate solutions
    d = 1000 # Example parameter, adjust as needed
    n_iter = 1000  # Example number of iterations
    start_time = time.time()  # 計算開始時間
    y_history = qi_solver(A, b, d, n_iter)
    end_time = time.time()  # 計算終了時間
    print("Time for qi_solver:", end_time - start_time)

    # Step 4: Calculate errors over iterations
    errors = []
    for y in y_history:
        x_approx = np.transpose(A) @ y
        error = np.linalg.norm(x_true - x_approx)
        errors.append(error)
    print("true solution", x_true)
    print("qi single trial solution", np.transpose(A) @ y_history[-1])
    # Step 5: Plot the error evolution
    plt.plot(range(len(y_history)), errors, label="single trial")

    n_trials = 100
    y_history_list = np.zeros((n_trials, len(y_history), len(y_history[0])))
    for i in range(n_trials):
        y_history_list[i] = np.array(qi_solver(A,b,d,n_iter))
    averaged_y_history = np.sum(y_history_list, axis=0)/n_trials
    # Step 4: Calculate errors over iterations
    errors = []
    for y in averaged_y_history:
        x_approx = np.transpose(A) @ y
        error = np.linalg.norm(x_true - x_approx)
        errors.append(error)
    print(f"qi {n_trials} trials solution", np.transpose(A) @ averaged_y_history[-1])
    # Step 5: Plot the error evolution
    plt.plot(range(len(y_history)), errors, label=f"average over {n_trials} trials")
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("test.png")