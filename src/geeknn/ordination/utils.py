from typing import Any

import ee
from pydantic import BaseModel


class Colocation(BaseModel):
    fc: ee.FeatureCollection
    location_field: str
    plot_field: str

    class Config:
        arbitrary_types_allowed = True

    def get_colocation_fc(self) -> ee.FeatureCollection:
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


# Extract arrays from feature classes for the specified columns
def fc_to_array(fc, columns):
    columns = ee.List(columns)
    reducer = ee.Reducer.toList().repeat(columns.size())
    fc_array = fc.reduceColumns(reducer, columns)
    return ee.Array(fc_array.get("list")).transpose()


# Convert transformed scores to a feature class with IDs
def scores_to_fc(ids, scores, id_field):
    def axis_name(i):
        return ee.String("AXIS").cat(ee.Number(i).int().format())

    def axis_score(i):
        id_ = ee.Dictionary().set(id_field, ee.List(ids).get(i))
        score = ee.Dictionary.fromLists(axis_names, ee.List(scores).get(i))
        return ee.Feature(None, id_.combine(score))

    n_rows = ee.List(scores).size()
    n_cols = ee.List(ee.List(scores).get(0)).size()
    axis_names = ee.List.sequence(1, n_cols).map(axis_name)

    fc = ee.FeatureCollection(
        ee.List.sequence(0, n_rows.subtract(1)).map(axis_score)
    )
    return ee.Dictionary({"fc": fc, "axis_names": axis_names})


def column_means(arr):
    return arr.reduce(ee.Reducer.mean(), [0])


def column_sds(arr):
    return arr.reduce(ee.Reducer.stdDev(), [0])


def normalize_arr(arr):
    n_rows = arr.length().get([0])
    return arr.subtract(column_means(arr).repeat(0, n_rows)).divide(
        column_sds(arr).repeat(0, n_rows)
    )


def filter_neighbors(
    neighbor_fc: ee.FeatureCollection, colocation_obj: Colocation, id_field: str
):
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


def return_k_neighbors(
    neighbor_fc: ee.FeatureCollection, k: int
) -> Any:  # want ee.Array
    def return_k_neighbor_for_feature(lst):
        return ee.List(lst).slice(0, ee.Number(k))

    return ee.Array(
        neighbor_fc.aggregate_array("neighbors").map(
            return_k_neighbor_for_feature
        )
    )
