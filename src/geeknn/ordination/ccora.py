import ee

ee.Initialize()

from . import utils


def ccora(X, Y):
    # Run QR decomposition on both X and Y matrices
    QRX = X.matrixQRDecomposition()
    QRY = Y.matrixQRDecomposition()

    n_rows = X.length().get([0])
    n_cols_x = X.length().get([1])
    n_cols_y = Y.length().get([1])

    # Create an identity matrix, but n_rows x n_cols_y
    diag = ee.Array.identity(n_cols_y)
    empty = Y.multiply(0).slice(0, 0, n_rows.subtract(n_cols_y))
    diag = ee.Array.cat([diag, empty])

    A = (
        ee.Array(QRX.get("Q"))
        .transpose()
        .matrixMultiply(ee.Array(QRY.get("Q")).matrixMultiply(diag))
        .slice(0, 0, n_cols_x)
    )

    Z = A.matrixSingularValueDecomposition()

    # Solve Rx = U for QRX
    R = ee.Array(QRX.get("R")).slice(0, 0, n_cols_x).slice(1, 0, n_cols_x)
    QR = R.matrixQRDecomposition()
    x_coef = ee.Array(QR.get("R")).matrixSolve(ee.Array(Z.get("U")))

    # Solve Rx = V for QRY
    R = ee.Array(QRY.get("R")).slice(0, 0, n_cols_y).slice(1, 0, n_cols_y)
    QR = R.matrixQRDecomposition()
    y_coef = ee.Array(QR.get("R")).matrixSolve(ee.Array(Z.get("V")))

    # Derive the cscal scalar
    col = X.matrixMultiply(x_coef.slice(1, 0, 1))
    sds = utils.column_sds(col)
    cscal = ee.Number(1.0).divide(ee.Number(sds.get([0, 0])))
    x_coef = x_coef.multiply(cscal)

    # TODO: Need to do ftest.cor to get this value
    n_vec = 12
    x_coef = x_coef.slice(1, 0, n_vec)
    D = ee.Array(Z.get("S")).slice(0, 0, n_vec).slice(1, 0, n_vec)
    return x_coef.matrixMultiply(D)
