from pprint import pprint
from geeknn.test_data import setup as sample_data
from geeknn.ordination import Euclidean, GNN

# This script details the default use of two of the five NN
# classifiers in this package: Euclidean and GNN.  Raw and
# Mahalanobis have a very similar API to Euclidean, and MSN
# is similar to GNN.

# -------------------
# Euclidean workflow
# -------------------

# 1 - Initialize the classifier
# Euclidean (and Raw, Mahalanobis, and MSN) currently only have
# a single initialization option
#
#   k: integer
#
# You can use any of the three ways to call it

# Default, k=1
model = Euclidean()

# Integer argument, k=5
model = Euclidean(5)

# Keyword argument, k=5
model = Euclidean(k=5)
print("EUC model", model)

# 2 - Train the classifier.  Raw, Euclidean, and Mahalanobis all
# take ["fc", "id_field", and "env_columns"] arguments. You can call
# using either arguments or keywords.  Note that the train method
# returns the object itself so it can be chained with the initializer

# Positional arguments
model = model.train(
    sample_data.fc, sample_data.id_field, sample_data.env_columns
)

# Keyword arguments
model = model.train(
    fc=sample_data.fc,
    id_field=sample_data.id_field,
    env_columns=sample_data.env_columns,
)
print("Trained EUC model", model)

# 3 - Run (predict) in spatial mode.  Same signature for all methods.
# Call using either signature.  Note that all env_columns *must*
# be present as band names in env_img

# Positional argument
nn = model.predict(sample_data.env_img)

# Keyword argument
nn = model.predict(env_image=sample_data.env_img)

# 4 - Run (predict) in feature collection mode.  Same signature for
# all methods.  Call using either signature.  Here we're limiting to
# the first ten records of the feature collection. Note that this
# is returning the plot itself as the first neighbor.

# Positional argument
nn_fc = model.predict_fc(sample_data.fc.limit(10))

# Keyword argument
nn_fc = model.predict_fc(fc=sample_data.fc.limit(10))
print("EUC NN FC prediction:")
pprint(nn_fc.getInfo())

# -------------------
# GNN workflow
# -------------------

# 1 - Initialize the classifier
# GNN has three initialization options
#
#   k: integer
#   spp_transform: string, one of "SQRT", "LOG", "NONE"
#   num_cca_axes: integer, number of CCA axes to use in prediction
#
# You can use any of the three ways to call it.
# (The "LOG" option is untested and giving some errors, so I'd
# stick with "SQRT" or "NONE" for now)

# Default, k=1, spp_transform="SQRT", num_cca_axes=8
model = GNN()

# Positional arguments
model = GNN(5, "LOG", 4)

# Keyword arguments
model = GNN(k=5, spp_transform="NONE", num_cca_axes=8)
print("GNN model", model)

# 2 - Train the classifier.  MSN and GNN both take four arguments -
# ["fc", "id_field", "spp_columns",and "env_columns"]. You can call
# using either arguments or keywords.  Note that the train method
# returns the object itself so it can be chained with the initializer

# Positional arguments
model = model.train(
    sample_data.fc,
    sample_data.id_field,
    sample_data.spp_columns,
    sample_data.env_columns,
)

# Keyword arguments
model = model.train(
    fc=sample_data.fc,
    id_field=sample_data.id_field,
    spp_columns=sample_data.spp_columns,
    env_columns=sample_data.env_columns,
)
print("Trained GNN model", model)

# 3 - Run (predict) in spatial mode.  Same signature for all methods.
# Call using either signature.  Note that all env_columns *must*
# be present as band names in env_img

# Positional argument
nn = model.predict(sample_data.env_img)

# Keyword argument
nn = model.predict(env_image=sample_data.env_img)

# 4 - Run (predict) in feature collection mode.  Same signature for
# all methods.  Call using either signature.  Here we're limiting to
# the first ten records of the feature collection. Note that this
# is returning the plot itself as the first neighbor.

# Positional argument
nn_fc = model.predict_fc(sample_data.fc.limit(10))

# Keyword argument
nn_fc = model.predict_fc(fc=sample_data.fc.limit(10))
print("GNN NN FC prediction:")
pprint(nn_fc.getInfo())
