import ee

ee.Initialize()


# Factory to create functions for row, column and total array sums
def sum_factory(axes):
    def axis_sums(arr):
        return arr.reduce(ee.Reducer.sum(), axes)

    return axis_sums


# Row, columns and array sums
row_sums = sum_factory([1])
col_sums = sum_factory([0])
arr_sum = sum_factory([0, 1])


# Weighted environmental means
def weighted_means(arr, weights):
    return weights.transpose().matrixMultiply(arr)


# Weighted centers
def weight_center(arr, weights):
    centers = weighted_means(arr, weights)
    arr_dim = arr.length()
    x_diff = arr.subtract(centers.repeat(0, arr_dim.get([0])))
    return x_diff.multiply(weights.pow(0.5).repeat(1, arr_dim.get([1])))


# Canonical correspondence analysis (ter Braak, 1986)
# Ported from R vegan
def cca(spp, env):
    # Get matrix dimensions
    n_cols_env = env.length().get([1])

    # Divide each element in the spp array by the array sum
    total = arr_sum(spp).get([0, 0])
    spp_transformed = spp.divide(total)

    # Get matrix sums and the outer product
    rs = row_sums(spp_transformed)
    cs = col_sums(spp_transformed)
    rc = rs.matrixMultiply(cs)

    # Chi-squared contributions
    X_bar = spp_transformed.subtract(rc).divide(rc.pow(0.5))

    centers = weighted_means(env, rs)
    Y_r = weight_center(env, rs)

    # Perform QR decomposition on the Y_r matrix
    QR = Y_r.matrixQRDecomposition()
    Q_T = ee.Array(QR.get("Q")).transpose()
    R = ee.Array(QR.get("R"))
    right = Q_T.matrixMultiply(X_bar)
    LS = R.matrixSolve(right)

    # Perform SV decomposition on the fitted env scores
    Y = Y_r.matrixMultiply(LS)
    svd = Y.matrixSingularValueDecomposition()
    U = ee.Array(svd.get("U"))
    S = ee.Array(svd.get("S"))

    # Subset these matrices down to the rank of the matrix
    U_raw = U.slice(1, 0, n_cols_env)
    S = S.slice(0, 0, n_cols_env).slice(1, 0, n_cols_env).pow(2.0)
    eig_sum = arr_sum(S).get([0, 0])
    eig_matrix = S.divide(eig_sum).pow(0.5)

    ones = rs.multiply(0).add(1)
    weights = ones.divide(rs.pow(0.5)).matrixToDiag()
    U = weights.matrixMultiply(U_raw)

    coeff = R.matrixSolve(Q_T.matrixMultiply(U_raw))
    plot_scores = U.matrixMultiply(eig_matrix).toList()
    num_cca_axes = coeff.length().toList().get(1)

    return ee.Dictionary(
        {
            "coeff": coeff,
            "centers": centers,
            "eig_matrix": eig_matrix,
            "plot_scores": plot_scores,
            "num_cca_axes": num_cca_axes,
        }
    )
