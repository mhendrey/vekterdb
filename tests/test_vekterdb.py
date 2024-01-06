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
    records = make_data("idx", "id", "vector", n=160_000, seed=930)
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
    vekter_db.insert(records)
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


def test_init_custom(tmp_path: Path):
    dbname = tmp_path / "test.db"

    vekter_db = VekterDB(
        "custom_table",
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
        url=f"sqlite:///{dbname}",
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
    assert vekter_db.Record.__table__.name == "custom_table"

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
        "insert_test",
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
        "insert_bytes_test",
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


def test_flat_index(test_db):
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
        url=f"sqlite:///{test_db}",
    )
    faiss_factory_string = "Flat"
    vekter_db.create_index(
        "test_flat.index", faiss_factory_string, metric="inner_product"
    )
    assert (
        vekter_db.index.ntotal == 160_000
    ), f"{vekter_db.index.ntotal=:,} should be 160,000"


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
