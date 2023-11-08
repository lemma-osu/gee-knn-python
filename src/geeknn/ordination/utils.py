from __future__ import annotations

import ee
from pydantic import BaseModel, ConfigDict


class Colocation(BaseModel):
    """Class for identifying colocated features in a feature collection"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fc: ee.FeatureCollection
    location_field: str
    plot_field: str

    def get_colocation_fc(self) -> ee.FeatureCollection:
        """Return a feature collection that returns a list of colocated features
        for each feature in the original feature collection.  Output features
        will have a property called 'colocated' that is a list of feature IDs
        that are colocated with the feature in question.
        """
        join = ee.Join.saveAll(
            matchesKey="matches",
            ordering=self.plot_field,
            ascending=True,
        )
        fltr = ee.Filter.equals(
            leftField=self.location_field,
            rightField=self.location_field,
        )
        applied = join.apply(self.fc, self.fc, fltr)

        def get_colocated_list(f: ee.Feature) -> ee.Feature:
            def get_colocated_feature(f2: ee.Feature) -> ee.Feature:
                return ee.Feature(f).get(self.plot_field)

            colocated = ee.List(f.get("matches")).map(get_colocated_feature)
            return (
                ee.Feature(None)
                .set(self.plot_field, f.get(self.plot_field))
                .set("colocated", colocated)
            )

        return applied.map(get_colocated_list)


def scores_to_fc(
    ids: list[int], scores: list[list[float]], id_field: str
) -> ee.Dictionary:
    """Convert transformed scores to a feature collection with IDs."""

    def axis_name(i: int) -> ee.String:
        return ee.String("AXIS").cat(ee.Number(i).int().format())

    def axis_score(i: int) -> ee.Feature:
        id_ = ee.Dictionary().set(id_field, ee.List(ids).get(i))
        score = ee.Dictionary.fromLists(axis_names, ee.List(scores).get(i))
        return ee.Feature(None, id_.combine(score))

    n_rows = ee.List(scores).size()
    n_cols = ee.List(ee.List(scores).get(0)).size()
    axis_names = ee.List.sequence(1, n_cols).map(axis_name)

    fc = ee.FeatureCollection(ee.List.sequence(0, n_rows.subtract(1)).map(axis_score))
    return ee.Dictionary({"fc": fc, "axis_names": axis_names})


def crosswalk_to_ids(
    neighbor_fc: ee.FeatureCollection, ids: ee.List, id_field: ee.String
) -> ee.FeatureCollection:
    """Add the IDs to the neighbor feature collection."""
    zipped = neighbor_fc.toList(neighbor_fc.size()).zip(ids)

    def zip_with_id(t):
        t = ee.List(t)
        f = ee.Feature(t.get(0))
        id_ = ee.Number(t.get(1))
        return (
            ee.Feature(None)
            .set("neighbors", ee.Array(f.get("neighbors")).toList())
            .set(id_field, id_)
        )

    return ee.FeatureCollection(zipped.map(zip_with_id))


def filter_neighbors(
    neighbor_fc: ee.FeatureCollection, colocation_obj: Colocation, id_field: str
):
    """For a given feature collection of neighbors, filter out the neighbors that
    are colocated with the feature in question.
    """
    colocation_fc = colocation_obj.get_colocation_fc()
    fltr = ee.Filter.equals(
        leftField=id_field,
        rightField=colocation_obj.plot_field,
    )
    applied = ee.Join.inner().apply(neighbor_fc, colocation_fc, fltr)

    def get_colocated(f):
        neighbors = ee.List(ee.Feature(f.get("primary")).get("neighbors"))
        colocated = ee.List(ee.Feature(f.get("secondary")).get("colocated"))
        return ee.Feature(None).set(
            {
                "id_field": ee.Feature(f.get("primary")).get(id_field),
                "neighbors": neighbors.filter(
                    ee.Filter.inList("item", colocated).Not()
                ),
            }
        )

    return applied.map(get_colocated)


def get_k_neighbors(neighbor_fc: ee.FeatureCollection, k: int):
    """Return the k nearest neighbors for each feature in the feature collection
    as an array.
    """

    def get_k_neighbor_for_feature(lst):
        return ee.List(lst).slice(0, ee.Number(k))

    return ee.Array(
        neighbor_fc.aggregate_array("neighbors").map(get_k_neighbor_for_feature)
    )
