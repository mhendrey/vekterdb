"""
Copyright (C) 2023 Matthew Hendrey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import faiss
import numpy as np
import logging
from pathlib import Path
import pytest
import sqlalchemy as sa
from typing import Dict, List

# Bit of a hack, but this way I can run pytests in main directory, but still have IDE
# aware of VekterDB
try:
    from ..vekterdb.vekterdb import VekterDB
except ImportError:
    from vekterdb.vekterdb import VekterDB


def make_data(
    idx_name: str,
    id_name: str,
    vector_name: str,
    idx_start: int = 0,
    normalize: bool = True,
    n: int = 15_000,
    d: int = 64,
    seed: int = None,
) -> List[Dict]:
    rng = np.random.default_rng(seed=seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    if normalize:
        faiss.normalize_L2(X)

    for idx in range(idx_start, idx_start + n):
        yield {idx_name: idx, id_name: str(idx), vector_name: X[idx - idx_start]}


@pytest.fixture(scope="session")
def test_db(tmp_path_factory: Path):
    """
    Create a tmp directory to store the sqlite database to be used during the testing
    session. You can then pass this fixture into various test functions.

    Parameters
    ----------
    tmp_path_factory : Path
        Session-scoped fixture which can be used to create arbitrary temporary
        directories from any other fixture or test

    Returns
    -------
    test_db : Path
        Path to the sqlite database that will be used for testing
    """
    dbname = tmp_path_factory.mktemp("db") / "test.db"
    n = 100_000
    d = 64
    records = make_data("idx", "id", "vector", n=n, d=d, seed=930)
    # Create table using
    vekter_db = VekterDB(
        "test_table",
        columns_dict={
            "id": {
                "type": sa.types.Text,
                "unique": True,
                "nullable": False,
                "index": True,
            }
        },
        url=f"sqlite:///{dbname}",
    )
    n_records = vekter_db.insert(records)

    # Make a query record that is summing the first two vectors
    records = list(vekter_db.fetch_records("idx", [0, 1]))
    q_vec = 1.05 * records[0]["vector"] + records[1]["vector"]
    faiss.normalize_L2(q_vec.reshape(1, -1))
    query_record = {"idx": n_records, "id": "query", "vector": q_vec}
    vekter_db.insert([query_record])

    return dbname


def test_init_basic():
    # Create basic table using in-memory SQLite
    # Create a basic table with just idx: BigInt and vectors: LargeBinary
    vekter_db = VekterDB("my_table")

    # Asserts for the idx column
    assert vekter_db.idx_name == "idx", "Default value for idx_name != 'idx'"
    idx_col = vekter_db.columns["idx"]
    assert isinstance(idx_col.type, sa.types.BigInteger)
    assert idx_col.primary_key, "idx column is supposed to be primary key for the table"
    assert not idx_col.nullable, "idx column should have nullable = False"

    # Asserts for the vector column
    assert vekter_db.vector_name == "vector"
    vector_col = vekter_db.columns["vector"]
    assert isinstance(vector_col.type, sa.types.LargeBinary)
    assert not vector_col.nullable, "vector column should have nullable = False"

    # Asserts for the table
    assert vekter_db.Record.__table__.name == "my_table"
    assert len(vekter_db.columns) == 2
    assert len(vekter_db.Record.__table__.columns) == 2


def test_init_custom():
    vekter_db = VekterDB(
        "my_table",
        idx_name="i",
        vector_name="v",
        columns_dict={
            "id": {
                "type": sa.types.Text,
                "unique": True,
                "nullable": False,
                "index": True,
            },
            "product_category": {"type": sa.types.Text},
        },
    )

    # Asserts for the idx column
    assert vekter_db.idx_name == "i", f"{vekter_db.idx_name=:} != 'i'"
    idx_col = vekter_db.columns["i"]
    assert isinstance(idx_col.type, sa.types.BigInteger)
    assert idx_col.primary_key, "idx column is supposed to be primary key for the table"
    assert not idx_col.nullable, "idx column should have nullable = False"

    # Asserts for the vector column
    assert vekter_db.vector_name == "v", f"{vekter_db.vector_name=:} != 'v'"
    vector_col = vekter_db.columns["v"]
    assert isinstance(vector_col.type, sa.types.LargeBinary)
    assert not vector_col.nullable, "vector column should have nullable = False"

    # Assert other column exists too
    assert (
        len(vekter_db.columns) == 4
    ), f"Should have 4 columns in db table. Instead there are {len(vekter_db.columns)}"
    assert vekter_db.Record.__table__.name == "my_table"

    # Asserts for id column
    id_col = vekter_db.columns["id"]
    assert isinstance(id_col.type, sa.types.Text), f"{id_col.type=:} should be 'Text'"
    assert id_col.unique, f"{id_col.unique=:} should be True"
    assert not id_col.nullable, f"{id_col.nullable=:} should be False"
    assert id_col.index, f"{id_col.index=:} should be True"

    # Asserts for the product category column
    product_cat_col = vekter_db.columns["product_category"]
    assert isinstance(
        product_cat_col.type, sa.types.Text
    ), f"{product_cat_col.type=:} should be 'Text'"


def test_insert():
    records = list(
        make_data(
            "idx", "id", "vector", idx_start=0, normalize=True, n=15_000, d=64, seed=828
        )
    )

    # Create table using in-memory SQLite
    vekter_db = VekterDB(
        "my_table",
        columns_dict={
            "id": {
                "type": sa.types.Text,
                "unique": True,
                "nullable": False,
                "index": True,
            }
        },
    )
    records_gen = make_data(
        "idx", "id", "vector", idx_start=0, normalize=True, n=15_000, d=64, seed=828
    )
    n_records = vekter_db.insert(records_gen, batch_size=3_000, serialize_vectors=True)

    assert n_records == len(records), f"{n_records=:} does not equal {len(records):=}"

    # Retrieve a record
    with vekter_db.Session() as session:
        stmt = sa.select(vekter_db.Record).where(vekter_db.columns["idx"] == 1234)
        row = session.scalar(stmt)
        v = vekter_db.deserialize_vector(row.vector)
        assert row.id == "1234", f"{row.id=:} should be '1234'"
        assert row.idx == 1234, f"{row.idx=:} should be 1234"
        assert np.all(v == records[1234]["vector"]), f"vector retrieved mismatches"

    new_records = list(
        make_data(
            "idx",
            "id",
            "vector",
            idx_start=15_000,
            normalize=True,
            n=1_000,
            d=64,
            seed=853,
        )
    )
    records = records + new_records
    records_gen = make_data(
        "idx", "id", "vector", idx_start=15_000, normalize=True, n=1_000, d=64, seed=853
    )
    n_new_records = vekter_db.insert(
        records_gen, batch_size=990, serialize_vectors=True
    )
    assert n_new_records == 1000, f"{n_new_records:=} should be 1,000"

    with vekter_db.Session() as session:
        stmt = sa.select(vekter_db.Record).where(vekter_db.columns["idx"] == 15_500)
        row = session.scalar(stmt)
        v = vekter_db.deserialize_vector(row.vector)
        assert row.id == "15500", f"{row.id=:} should be '15500'"
        assert row.idx == 15500, f"{row.idx=:} should be 15500"
        assert np.all(v == records[15500]["vector"]), f"vector retrieved mismatches"

        count = session.scalar(
            sa.select(sa.sql.functions.count(vekter_db.columns["idx"]))
        )
        assert count == len(records), f"{count=:,} should equal {len(records):,}"


def test_insert_bytes():
    records = []
    for record in make_data(
        "idx", "id", "vector", idx_start=0, normalize=True, n=15_000, d=64, seed=828
    ):
        record["vector"] = record["vector"].tobytes()
        records.append(record)

    # Create table using in-memory SQLite
    vekter_db = VekterDB(
        "my_table",
        columns_dict={
            "id": {
                "type": sa.types.Text,
                "unique": True,
                "nullable": False,
                "index": True,
            }
        },
    )
    n_records = vekter_db.insert(records, serialize_vectors=False)
    assert n_records == len(records), f"{n_records=:} does not equal {len(records):=}"

    # Retrieve a record
    with vekter_db.Session() as session:
        stmt = sa.select(vekter_db.Record).where(vekter_db.columns["idx"] == 1234)
        row = session.scalar(stmt)
        assert row.id == "1234", f"{row.id=:} should be '1234'"
        assert row.idx == 1234, f"{row.idx=:} should be 1234"
        assert row.vector == records[1234]["vector"], f"vector retrieved mismatches"


def check_against_ground_truth(
    neighbors: List[Dict],
    self_included: bool,
):
    if not neighbors:
        raise AssertionError(f"{neighbors=:} is empty")

    q_idx = 100_000  # Taken from test_db
    if self_included:
        for i, neighbor in enumerate(neighbors):
            n_idx = neighbor["idx"]
            similarity = neighbor["metric"]
            if i == 0:
                assert n_idx == q_idx, f"{i=:}, {n_idx=:} but should be {q_idx=:}"
                assert similarity == pytest.approx(
                    1.0
                ), f"{similarity=:.4} but should be 1.0"
            elif i == 1:
                assert n_idx == 0, f"{i=:}, {n_idx=:} but should be 0"
                assert similarity == pytest.approx(
                    0.70678645
                ), f"{i=:}, {similarity=:.4f} but should be 0.7068"
            elif i == 2:
                assert n_idx == 1, f"{i=:}, {n_idx=:} but should be 1"
                assert similarity == pytest.approx(
                    0.6695153
                ), f"{i=:}, {similarity=:.4f} but should be 0.6695"
    else:
        for i, neighbor in enumerate(neighbors):
            n_idx = neighbor["idx"]
            similarity = neighbor["metric"]
            if i == 0:
                assert n_idx == 0, f"Nearest neighbor should be 0, got {n_idx=:}"
                assert similarity == pytest.approx(
                    0.70678645
                ), f"{i=:}, {similarity=:.4f} but should be 0.7068"

            elif i == 1:
                assert n_idx == 1, f"Second neighbor should be 1, got {n_idx=:}"
                assert similarity == pytest.approx(
                    0.6695153
                ), f"{i=:}, {similarity=:.4f} but should be 0.6695"


def test_flat_index(tmp_path_factory, test_db):
    faiss_file = str(tmp_path_factory.getbasetemp() / "flat.index")

    # Connect to the existing SQLite Database
    vekter_db = VekterDB(
        "test_table",
        url=f"sqlite:///{test_db}",
    )
    # Create the FAISS index
    faiss_factory_string = "Flat"
    vekter_db.create_index(faiss_file, faiss_factory_string, metric="inner_product")

    # Get the test query vector
    query = next(vekter_db.fetch_records("id", ["query"]))

    # Use search() without threshold. This includes yourself
    neighbors = vekter_db.search(query["vector"], 5, "idx", "id")[0]
    check_against_ground_truth(neighbors, self_included=True)
    assert (
        len(neighbors) == 5
    ), f"search(threshold=None) has {len(neighbors)=:} should be 5"

    # Use search() with threshold
    neighbors = vekter_db.search(query["vector"], 5, "idx", "id", threshold=0.65)[0]
    check_against_ground_truth(neighbors, self_included=True)
    assert (
        len(neighbors) == 3
    ), f"search(threshold=0.65) has {len(neighbors)=:} should be 3 when using threshold"

    # Use the nearest_neighbors() without threshold
    neighbors = vekter_db.nearest_neighbors("idx", [query["idx"]], 5, "idx", "id")[0][
        "neighbors"
    ]
    check_against_ground_truth(neighbors, self_included=False)
    assert (
        len(neighbors) == 5
    ), f"nearest_neighbors(threshold=None) has {len(neighbors)=:} should still be 5"

    # Use nearest_neighbors() with threshold
    neighbors = vekter_db.nearest_neighbors(
        "idx", [query["idx"]], 5, "idx", "id", threshold=0.65
    )[0]["neighbors"]
    check_against_ground_truth(neighbors, self_included=False)
    assert (
        len(neighbors) == 2
    ), f"nearest_neighbors(threshold=0.65) has {len(neighbors)=:} should still be 2"


def test_ivf_index(tmp_path_factory, test_db):
    faiss_file = str(tmp_path_factory.getbasetemp() / "ivf.index")
    # Connect to the existing SQLite Database
    vekter_db = VekterDB(
        "test_table",
        url=f"sqlite:///{test_db}",
    )
    # Create the FAISS index
    faiss_factory_string = "IVF300,PQ16"
    vekter_db.create_index(faiss_file, faiss_factory_string, sample_size=50_000)

    # Get the test query vector
    query = next(vekter_db.fetch_records("id", ["query"]))

    # Use search(threshold=None) and default nprobe=1
    neighbors = vekter_db.search(query["vector"], 5, "idx")[0]
    # It is most likely that this will fail to find everyone
    try:
        check_against_ground_truth(neighbors, self_included=True)
    except AssertionError as exc:
        logging.warning(
            f"As expected, IVF300,PQ16 search() default params failed ground truth check"
        )

    # Use search(threshold=None), but with nprobe=100
    neighbors = vekter_db.search(
        query["vector"],
        5,
        "idx",
        k_extra_neighbors=50,
        search_parameters=faiss.SearchParametersIVF(nprobe=100),
    )[0]
    check_against_ground_truth(neighbors, self_included=True)
    assert (
        len(neighbors) == 5
    ), f"search(threshold=None) has {len(neighbors)=:} should be 5"
    neighbor_idxs_no_extra = set([n["idx"] for n in neighbors])

    # Use search(threshold=None, k_extra_neighbors=100) with nprobe=100
    # This should give a different set of neighbors, but still have the top 3
    neighbors = vekter_db.search(
        query["vector"],
        5,
        "idx",
        k_extra_neighbors=100,
        search_parameters=faiss.SearchParametersIVF(nprobe=100),
    )[0]
    check_against_ground_truth(neighbors, self_included=True)
    neighbor_idxs_extra = set([n["idx"] for n in neighbors])
    n_overlap = len(neighbor_idxs_no_extra.intersection(neighbor_idxs_extra))
    try:
        assert n_overlap < 5, (
            "search(k_extra_neighbors) should get different set of "
            + f"neighbors but {neighbor_idxs_extra} {neighbor_idxs_no_extra}"
        )
    except:
        logging.warning(
            f"Expected {n_overlap=:} < 5, but guess both sets were the same this time"
        )

    # Use search(threshold=0.65) with nprobe=100
    neighbors = vekter_db.search(
        query["vector"],
        5,
        "idx",
        k_extra_neighbors=50,
        threshold=0.65,
        search_parameters=faiss.SearchParametersIVF(nprobe=100),
    )[0]
    check_against_ground_truth(neighbors, self_included=True)
    assert (
        len(neighbors) == 3
    ), f"search(threshold=0.65) has {len(neighbors)=:} but should be 3"

    # Set nprobe for this entire runtime to 100
    vekter_db.set_faiss_runtime_parameters("nprobe=100")

    # Use nearest_neighbor(). You don't have to set search_params now
    result = vekter_db.nearest_neighbors("idx", [query["idx"]], 5, "idx", "id")[0]
    assert (
        result["idx"] == query["idx"]
    ), f"nearest_neighbors() {result['idx']} should equal {query['idx']}"
    assert (
        result["id"] == query["id"]
    ), f"nearest_neighbors() {result['id']} should equal {query['id']}"
    neighbors = result["neighbors"]
    check_against_ground_truth(neighbors, self_included=False)

    # Do the search with setting nprobe=1 should be wrong
    neighbors = vekter_db.nearest_neighbors(
        "idx",
        [query["idx"]],
        5,
        "idx",
        "id",
        search_parameters=faiss.SearchParametersIVF(nprobe=1),
    )[0]["neighbors"]
    # It is most likely that this will fail to find everyone
    try:
        check_against_ground_truth(neighbors, self_included=False)
    except AssertionError as exc:
        logging.warning(
            f"As expected, IVF300,PQ16 nearest_neighbors with nprobe=1 "
            + "failed ground truth check"
        )

    # Use nearest_neighbors(threshold=0.65)
    neighbors = vekter_db.nearest_neighbors(
        "idx",
        [query["idx"]],
        5,
        "idx",
        "id",
        k_extra_neighbors=50,
        threshold=0.65,
    )[0]["neighbors"]
    check_against_ground_truth(neighbors, self_included=False)
    assert (
        len(neighbors) == 2
    ), f"search(threshold=0.65) has {len(neighbors)=:} but should be 2"


def test_hnsw_index(tmp_path_factory, test_db):
    faiss_file = str(tmp_path_factory.getbasetemp() / "hnsw.index")

    # Connect to the existing SQLite Database
    vekter_db = VekterDB(
        "test_table",
        url=f"sqlite:///{test_db}",
    )
    # Create the FAISS index
    faiss_factory_string = "HNSW32"
    vekter_db.create_index(faiss_file, faiss_factory_string)

    # Get the test query vector
    query = next(vekter_db.fetch_records("id", ["query"]))

    # Quick Test
    neighbors = vekter_db.nearest_neighbors(
        "idx",
        [query["idx"]],
        5,
        "idx",
        "id",
        k_extra_neighbors=20,
        threshold=0.65,
        search_parameters=faiss.SearchParametersHNSW(efSearch=60),
    )[0]["neighbors"]
    check_against_ground_truth(neighbors, self_included=False)


def test_opaque_index(tmp_path_factory, test_db):
    faiss_file = str(tmp_path_factory.getbasetemp() / "ivf_hnsw.index")

    # Connect to the existing SQLite Database
    vekter_db = VekterDB(
        "test_table",
        url=f"sqlite:///{test_db}",
    )
    # Create the FAISS index
    faiss_factory_string = "OPQ8,IVF300_HNSW32,PQ8"
    vekter_db.create_index(faiss_file, faiss_factory_string, sample_size=20_000)

    query = next(vekter_db.fetch_records("id", ["query"]))

    # Default search parameters, nprobe=1 & efSearch=16
    neighbors = vekter_db.nearest_neighbors(
        "idx",
        [query["idx"]],
        5,
        "idx",
        "id",
        threshold=0.65,
    )[0]["neighbors"]
    try:
        check_against_ground_truth(neighbors, self_included=False)
    except Exception as exc:
        logging.warning(f"As expected, ivf_hnsw with default failed: {exc}")

    # Setting search_parameters per query
    neighbors = vekter_db.nearest_neighbors(
        "idx",
        [query["idx"]],
        5,
        "idx",
        "id",
        threshold=0.65,
        k_extra_neighbors=30,
        search_parameters=faiss.SearchParametersPreTransform(
            index_params=faiss.SearchParametersIVF(
                nprobe=100,
                quantizer_params=faiss.SearchParametersHNSW(efSearch=200),
            )
        ),
    )[0]["neighbors"]
    check_against_ground_truth(neighbors, self_included=False)

    # Setting Faiss Runtime Search Parameters
    vekter_db.set_faiss_runtime_parameters("nprobe=100,quantizer_efSearch=200")
    neighbors = vekter_db.nearest_neighbors(
        "idx",
        [query["idx"]],
        5,
        "idx",
        "id",
        threshold=0.65,
        k_extra_neighbors=30,
    )[0]["neighbors"]
    check_against_ground_truth(neighbors, self_included=False)


def test_serialization(seed: int = None, d: int = 16):
    """Test that serializing a vector into bytes and deserializing from bytes give
    correct values. Start in each direction.

    Parameters
    ----------
    seed : int, optional
        Seed to use to generate the vectors. Default is None.
    d : int, optional
        Vector dimension to use, by default 16
    """
    rng = np.random.default_rng(seed)

    # Starting with a numpy array
    v1 = rng.normal(size=d).astype(np.float32)
    v1_bytes = VekterDB.serialize_vector(v1)
    assert isinstance(v1_bytes, bytes)
    v1_roundtrip = VekterDB.deserialize_vector(v1_bytes)
    assert np.all(v1 == v1_roundtrip)

    # Starting with bytes
    v2_bytes = rng.normal(size=d).astype(np.float32).tobytes()
    v2 = VekterDB.deserialize_vector(v2_bytes)
    assert isinstance(v2, np.ndarray)
    assert v2.shape == (d,)
    assert v2.dtype == np.float32
    v2_roundtrip = VekterDB.serialize_vector(v2)
    assert v2_bytes == v2_roundtrip


def test_similarity():
    v1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    v2 = np.array([-1, 2, 3, 4, 5, -6, -7, -8, 9, 10], dtype=np.float32)

    records = [
        {"idx": 0, "vector": v1.copy()},
        {"idx": 1, "vector": v2.copy()},
    ]

    ## Test inner_product metric
    vekter_db = VekterDB("inner_product")
    vekter_db.insert(records)
    vekter_db.create_index("tmp.index", "Flat", "inner_product")

    # true_inner_product = 85.0
    true_inner_product = np.sum([v1i * v2i for v1i, v2i in zip(v1, v2)])
    similarity = vekter_db.similarity(v1, v2, threshold=80.0)
    assert true_inner_product == pytest.approx(
        similarity
    ), f"{similarity=:} but should be 85.0"

    # Test that if similarity is below the threshold, then None is returned
    similarity = vekter_db.similarity(v1, v2, threshold=100.0)
    assert similarity is None, f"{similarity=:}, but should be below 100.0"

    records = [
        {"idx": 0, "vector": v1.copy()},
        {"idx": 1, "vector": v2.copy()},
    ]

    ## Test L2 metric
    vekter_db = VekterDB("l2")
    vekter_db.insert(records)
    vekter_db.create_index("tmp.index", "Flat", "l2")

    # True L2 norm = 24.494898
    true_l2 = np.sqrt(np.square(v1 - v2).sum())
    similarity = vekter_db.similarity(v1, v2, threshold=30.0)
    assert true_l2 == pytest.approx(
        similarity
    ), f"{similarity=:.4f} but should be {true_l2:.4f}"

    # Test that if similarity is above the threshold, then None is returned
    similarity = vekter_db.similarity(v1, v2, threshold=20.0)
    assert similarity is None, f"{similarity=:}, but should be above 20.0"
