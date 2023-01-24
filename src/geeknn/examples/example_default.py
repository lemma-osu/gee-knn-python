from geeknn.test_data import setup as sample_data
from geeknn.ordination.models import GNN


# Different NN methods
# var euc = require("users/gregorma/gee-knn:ordination/models.js").Euclidean;
# var raw = require("users/gregorma/gee-knn:ordination/models.js").Raw;
# var mah = require("users/gregorma/gee-knn:ordination/models.js").Mahalanobis;
# var msn = require("users/gregorma/gee-knn:ordination/models.js").MSN;

# 1 - Initialize the classifier.
# RAW, EUC, MAH, and MSN API - using EUC as an example.  k is
# the only option to pass to the initializer.  You can use any
# of the three ways to call it.
# var euc_model;
# euc_model = euc();         // Default, k=1
# euc_model = euc(5);        // Integer argument, k=5
# euc_model = euc({k: 5});   // Object argument, k=5
# print("EUC model", euc_model);

# GNN - three options
#   k: integer
#   spp_transform: string, one of "SQRT", "LOG", "NONE"
#   num_cca_axes: integer, number of CCA axes to use in prediction
# You can use any of the three ways to call it.
# (The "LOG" option is untested and giving some errors, so I'd
# stick with "SQRT" or "NONE" for now)

# Default, k=1, spp_transform="SQRT", num_cca_axes=8
gnn_model = GNN()

# Positional arguments
gnn_model = GNN(5, "LOG", 4)

# Object argument
gnn_model = GNN(k=5, spp_transform="NONE", num_cca_axes=8)

# 2 - Train the classifier.  MSN and GNN take the spp_columns arguments,
# all others only take ["fc", "id_field", and "env_columns"] arguments.
# Call using either signature
# euc_model = euc_model.train(
#   sample_data.fc,
#   sample_data.id_field,
#   sample_data.env_columns
# );
# euc_model = euc_model.train({
#   fc: sample_data.fc,
#   id_field: sample_data.id_field,
#   env_columns: sample_data.env_columns,
# });
# print("Trained EUC model", euc_model);

gnn_model = gnn_model.train(
    sample_data.fc,
    sample_data.id_field,
    sample_data.spp_columns,
    sample_data.env_columns,
)
gnn_model = gnn_model.train(
    fc=sample_data.fc,
    id_field=sample_data.id_field,
    spp_columns=sample_data.spp_columns,
    env_columns=sample_data.env_columns,
)
# print("Trained GNN model", gnn_model)

# 3 - Run (predict) in spatial mode.  Same signature for all methods.
# Call using either signature.  Note that all env_columns *must*
# be present as band names in env_img
# var euc_nn;
# var euc_nn = euc_model.predict(sample_data.env_img);
# var euc_nn = euc_model.predict({env_image: sample_data.env_img});
# print("EUC NN image", euc_nn);

gnn_nn = gnn_model.predict(sample_data.env_img)
gnn_nn = gnn_model.predict(env_image=sample_data.env_img)
# print("GNN NN image", gnn_nn.getInfo())

# 4 - Run (predict) in feature collection mode.  Same signature for
# all methods.  Call using either signature.  I'm running into some
# memory issues, so I'm limiting to the first ten records right now.
# Also, I'm returning the plot itself as the first neighbor.  We
# will want to filter this out when we do AA, but I think it's better
# to leave it as the first true neighbor.
# var euc_nn_fc;
# euc_nn_fc = euc_model.predict_fc(sample_data.fc.limit(10));
# euc_nn_fc = euc_model.predict_fc({fc: sample_data.fc.limit(10)});
# print("EUC NN FC prediction:", euc_nn_fc);

gnn_nn_fc = gnn_model.predict_fc(sample_data.fc.limit(10))
gnn_nn_fc = gnn_model.predict_fc(fc=sample_data.fc.limit(10))
print("GNN NN FC prediction:", gnn_nn_fc.getInfo())
