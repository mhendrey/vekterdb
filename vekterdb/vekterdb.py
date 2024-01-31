"""
VekterDB turns any SQLAlchemy compliant database into a vector database using the FAISS
library to index the vectors.

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

from datetime import datetime
import json
import logging
import numcodecs
import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy.sql.functions as sa_funcs
from typing import Dict, Iterable, Iterator, List

try:
    import faiss
except:
    logging.error("Failed to import FAISS. Install from conda")


class VekterDB:
    """Turn any SQLAlchemy compliant database into a vector database using the FAISS
    library to index the vectors.

    VekterDB uses a minimum of two columns in the database table: idx_name
    (BigInteger, default 'idx') and vector_name (LargeBinary, default 'vector').
    Vectors are numpy arrays of np.float32, serialized to bytes using tobytes(),
    to comply with FAISS requirements.

    Additional database table columns can be specified with ``columns_dict`` using
    SQLAlchemy's ``Column`` arguments. For example, include a unique, indexed id
    field (str) and a non-unique product_category field (str) with:

    ::

        my_db = VekterDB(
            "my_table",
            columns_dict={
                "id": {"type": Text, "unique": True, "nullable": False, "index": True},
                "product_category": {"type": Text},
            }
        )

    Attributes
    ----------
    idx_name : str
        Column name in the database table that stores integer [0, N) of the primary key.
    vector_name : str
        Column name in the database table that stores the vectors.
    Session : sa.orm.session.sessionmaker
        Session factory that gives you a Session object to database table.
    Record : sa.orm.decl_api.DeclarativeMeta
        ORM-mapped class for the database table
    d : int
        Dimensionality of the vectors
    index : faiss.Index
        FAISS index that indexes the ``vector_name`` vectors for similarity search
    metric : str
        Specifies the metric used to determine similarity.


    Parameters
    ----------
    table_name : str
        Database table name, either existing or new.
    idx_name : str, optional
        Column name that stores the FAISS index integer ID and is the primary key
        for the database table. It must be unique and consecutive from [0, n_records).
        The default name is "idx".
    vector_name : str, optional
        Column name that stores the vector information. Default is "vector".
    columns_dict : Dict[str, Dict], optional
        Names (key) of additional columns to include in the table. The values are
        arguments that will be passed to SQLAlchemy's ``Column``. Default is {}.

        When connecting to an existing database table, this argument is not necessary.
    url : str, optional
        URL string to connect to the database. Passed to SQLAlchemy's
        ``create_engine``. Default is "sqlite:///:memory"; an in-memory database.
    connect_args: Dict, optional
        Any connection arguments to pass to SQLAlchemy's ``create_engine``.
        Default is {}.
    faiss_index : str, optional
        If given, then load an existing FAISS index saved by that name. Default
        is None.
    """

    def __init__(
        self,
        table_name: str,
        idx_name: str = "idx",
        vector_name: str = "vector",
        columns_dict: Dict[str, Dict] = {},
        url: str = "sqlite:///:memory:",
        connect_args: Dict = {},
        faiss_index: str = None,
    ) -> None:
        """
        Initialize a VekterDB
        """
        self.logger = logging.getLogger(__name__)
        self.idx_name = idx_name
        self.vector_name = vector_name
        self.d = None

        self.engine = sa.create_engine(url, **connect_args)
        self.Session = sessionmaker(bind=self.engine)

        metadata_obj = sa.MetaData()
        try:
            metadata_obj.reflect(bind=self.engine, only=[table_name])
        except:
            table_exists = False
        else:
            table_exists = True

        # Create the table if it doesn't exist
        if not table_exists:
            self.logger.info(f"Creating {table_name} table in the database")
            # Build up the columns
            self.columns = {
                idx_name: sa.Column(idx_name, sa.types.BigInteger, primary_key=True)
            }
            for column_name, column_args in columns_dict.items():
                column_type = column_args.pop("type")
                self.columns[column_name] = sa.Column(
                    column_name, column_type, **column_args
                )
            # Vectors will be serialized to bytes for storage
            self.columns[vector_name] = sa.Column(
                vector_name, sa.types.LargeBinary, nullable=False
            )
            table = sa.Table(
                table_name,
                metadata_obj,
                *self.columns.values(),
            )
        else:
            self.logger.warning(
                f"{table_name} already exists in the database. Skipping table creation"
            )
            # Load existing table to get column information
            table = sa.Table(
                table_name,
                metadata_obj,
            )
            # Reflection doesn't add in Column.index = True if indexed.
            indexed_columns = set()
            for table_index in table.indexes:
                indexed_columns.update(table_index.columns.keys())
            self.columns = {}
            for col in table.columns:
                self.columns[col.name] = col
                if col.name in indexed_columns:
                    self.columns[col.name].index = True

            assert (
                idx_name in self.columns
            ), f"{idx_name} column not found in {table_name}"
            assert (
                vector_name in self.columns
            ), f"{vector_name} column not found in {table_name}"

        # Following the example of how to generate mappings from an existing MetaData
        # which also shows how you can add new mappings too.
        # https://docs.sqlalchemy.org/en/20/orm/extensions/automap.html#generating-mappings-from-an-existing-metadata
        Base = automap_base(metadata=metadata_obj)
        Base.prepare()
        self.Record = Base.classes[table_name]
        metadata_obj.create_all(self.engine, tables=[table])

        # Set self.d if there is already some data in existing table
        max_idx = -1
        if table_exists:
            with self.Session() as session:
                try:
                    # Pull a record from the database
                    stmt = sa.select(self.columns[self.vector_name]).limit(1)
                    vector_bytes = session.scalar(stmt)
                    vector = self.deserialize_vector(vector_bytes)
                except:
                    self.logger.warning(
                        f"{table_name} table exists in the database but appears empty."
                    )
                else:
                    self.d = vector.shape[0]
                stmt = sa.select(sa.func.max(self.columns[self.idx_name]))
                max_idx = session.scalar(stmt)

        if faiss_index:
            self.index = faiss.read_index(faiss_index)
            self.faiss_index = faiss_index
            assert self.d == self.index.d, (
                f"{table_name} has {self.d} dimensional vectors, "
                + f"but FAISS index has {self.index.d}. They must be equal."
            )
            if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                self.metric = "inner_product"
            elif self.index.metric_type == faiss.METRIC_L2:
                self.metric = "l2"
            else:
                raise TypeError(
                    "FAISS index's metric type does not match inner_product or L2"
                )
            if self.index.ntotal < (max_idx + 1):
                self.logger.warning(
                    f"{table_name} has {max_idx+1:,} {self.idx_name} values, but only "
                    + f"{self.index.ntotal:,} vectors in FAISS index. Call "
                    + "sync_index_to_db() to add missing vectors into FAISS index."
                )
        else:
            self.faiss_index = None
            self.index = None
            self.metric = None

    @staticmethod
    def serialize_vector(vector: np.ndarray, level=1) -> bytes:
        """
        Static method to serialize numpy vector to Python bytes. Uses zlib to
        compress the bytes.

        Parameters
        ----------
        vector : np.ndarray
            1-d numpy array of type np.float32
        level : int
            Zstandard compression level [1, 22]. Default is 1 (fastest)

        Returns
        -------
        bytes
        """
        return numcodecs.zstd.compress(vector, level)

    @staticmethod
    def deserialize_vector(vector_bytes: bytes) -> np.ndarray:
        """
        Static method to deserialize Python bytes to 1-d numpy vector.

        Parameters
        ----------
        vector_bytes : bytes
            Bytes representation of a vector.

        Returns
        -------
        np.ndarray
            1-d numpy array of type np.float32
        """
        return np.frombuffer(numcodecs.zstd.decompress(vector_bytes), dtype=np.float32)

    def save(self, config_file: str = None):
        """
        Saves configuration info to a JSON file, excluding the URL string for security.
        If ``set_faiss_runtime_parameters()`` has been called, it also saves and applies
        that setting when loading with ``VekterDB.load()``

        Parameters
        ----------
        config_file : str, optional
            JSON file name to save to disk. If not provided, saves the file as
            ``table_name``.json. The default is None.
        """
        table_name = self.Record.__table__.name
        if config_file is None:
            config_file = f"{table_name}.json"

        if self.faiss_index is None:
            self.logger.warning(
                "No FAISS index has been created. Saving to disk anyway."
            )

        config = {
            "table_name": table_name,
            "idx_name": self.idx_name,
            "vector_name": self.vector_name,
            "faiss_index": self.faiss_index,
        }

        try:
            config["faiss_runtime_parameters"] = self.faiss_runtime_parameters
        except AttributeError:
            pass

        with open(config_file, "w") as f:
            json.dump(config, f)

    @staticmethod
    def load(config_file: str, url: str, connect_args: Dict = {}):
        """
        Load a VekterDB from a configuration file (JSON format) and connect to the
        specified database engine.

        Parameters
        ----------
        config_file : str
            Name of the configuration file for the VekterDB to load.
        url : str
            URL string to connect to the database. See ``sa.create_engine()`` for
            details.
        connect_args: Dict, optional
            Connection arguments to pass to ``sa.create_engine()``. Default is {}

        Returns
        -------
        VekterDB
        """
        with open(config_file, "r") as f:
            config = json.load(f)
        config["url"] = url
        config["connect_args"] = connect_args
        faiss_runtime_parameters = config.pop("faiss_runtime_parameters", "")

        vdb = VekterDB(**config)
        if faiss_runtime_parameters:
            vdb.set_faiss_runtime_parameters(faiss_runtime_parameters)

        return vdb

    def insert(
        self,
        records: Iterable[Dict],
        batch_size: int = 10_000,
        serialize_vectors: bool = True,
        faiss_runtime_params: str = None,
        compression_level: int = 1,
    ) -> int:
        """
        Insert multiple records into the table. Vectors will also be added to the
        FAISS index if it already exists. If the FAISS index is updated, it is saved to
        disk.

        Parameters
        ----------
        records : Iterable[Dict]
            Each dictionary contains the column names as keys and their corresponding
            values.
        batch_size : int, optional
            Number of records to insert at once. Default is 10,000.
        serialize_vectors : bool, optional
            If ``True``, vectors will be serialized and compressed before insertion;
            if ``False``, user has already serialized & compressed the vectors. See
            ``serialized()`` for details. Default is True.
        faiss_runtime_params : str, optional
            Set FAISS index runtime parameters before adding vectors. Likely only
            useful if you have a quantizer index (e.g., IVF12345_HNSW32). The quantizer
            index (HNSW32) will be used during the ``index.add()`` to determine which
            partition to add the vector to. You may want to change from the default,
            whether that is the FAISS default (efSearch=16) or the value saved in
            ``self.faiss_runtime_parameters``. "quantizer_efSearch=40" would be an
            example value for the example index given. If used,
            ``self.faiss_runtime_parameters`` is set back to its value before function
            invocation. Default is ``None``
        compression_level : int, optional
            Zstandard compression level to apply to vectors before inserting into the
            database. Default is 1 (fastest)

        Returns
        -------
        n_records : int
            Number of records added to the table
        """
        start = datetime.now()
        orig_runtime_parameters = ""
        insert_into_faiss = False
        if self.index is not None and self.index.is_trained:
            insert_into_faiss = True
            if faiss_runtime_params:
                try:
                    orig_runtime_parameters = self.faiss_runtime_parameters
                except:
                    pass
                self.set_faiss_runtime_parameters(faiss_runtime_params)

        if not insert_into_faiss:
            self.logger.warning(
                "FAISS index either doesn't exist or isn't trained so "
                + "records will only be inserted into the database table."
                + ". Call create_index() to make the FAISS index."
            )

        n_records = 0
        with self.Session() as session:
            batch = []
            vectors = []
            for i, record in enumerate(records):
                # Make a copy. Otherwise if you serialize the vector, then that changes
                # reference value which is in records[i]["vector"] to bytes.  This has
                # confused me in testing when making records: List[Dict] and calling
                # VekterDB.insert(records) only to find records[0]["vector"] as bytes
                # instead of the expected numpy array.
                # I am worried that this causes lots of memory issues.  But I guess that
                # is what batching is for.
                record = record.copy()

                # Use the first record to set dimension if not already set
                # and do some checking
                if i == 0:
                    if serialize_vectors:
                        vector_d = record[self.vector_name].shape[0]
                    else:
                        vector_d = self.deserialize_vector(
                            record[self.vector_name]
                        ).shape[0]

                    if self.d is None:
                        self.d = vector_d
                    elif self.d != vector_d:
                        raise ValueError(
                            f"New vector dimension {vector_d} != existing dimension {self.d}"
                        )

                if serialize_vectors:
                    if insert_into_faiss:
                        vectors.append(record[self.vector_name].copy())
                    record[self.vector_name] = self.serialize_vector(
                        record[self.vector_name], compression_level
                    )
                elif insert_into_faiss:
                    vectors.append(self.deserialize_vector(record[self.vector_name]))
                batch.append(record)
                n_records += 1
                if len(batch) == batch_size:
                    session.execute(sa.insert(self.Record), batch)
                    batch = []
                    if insert_into_faiss:
                        self.index.add(np.vstack(vectors))
                        vectors = []
                    self.logger.debug(
                        f"insert: batch inserted, cumulatively {n_records} have "
                        + "been inserted"
                    )
            if len(batch) > 0:
                session.execute(sa.insert(self.Record), batch)
                if insert_into_faiss:
                    self.index.add(np.vstack(vectors))
            session.commit()

        if orig_runtime_parameters:
            self.set_faiss_runtime_parameters(orig_runtime_parameters)

        end = datetime.now()
        rate = n_records / (end - start).total_seconds()
        self.logger.info(f"insert: {n_records} inserted at {rate:.2f} records / sec")

        if insert_into_faiss:
            # Save the index to disk
            start = datetime.now()
            faiss.write_index(self.index, self.faiss_index)
            end = datetime.now()
            self.logger.info(
                f"insert: Saving {self.faiss_index} to disk took {end - start}"
            )

        return n_records

    def sample_vectors(
        self, sample_size: int = 0, batch_size: int = 10_000
    ) -> np.ndarray:
        """Retrieve a sample of vectors from the database table.

        Parameters
        ----------
        sample_size : int, optional
            Number of vectors to return. Default 0 returns all vectors
        batch_size : int, optional
            Pull vectors in batches of this size. Default is 10,000.

        Returns
        -------
        np.ndarray
            2-d array of sampled vectors with shape (sample_size, d)
        """
        # Get current total number of records in the database
        with self.Session() as session:
            stmt = sa.select(sa_funcs.max(self.columns[self.idx_name]))
            n_total = session.scalar(stmt) + 1

        if sample_size == 0:
            sample_size = n_total
        else:
            sample_size = min(sample_size, n_total)

        if sample_size == n_total:
            sample_idxs = np.arange(n_total).tolist()
        else:
            rng = np.random.default_rng()
            sample_idxs = rng.choice(n_total, size=sample_size, replace=False).tolist()

        X = np.zeros((sample_size, self.d), dtype=np.float32)
        n_records = len(sample_idxs)
        n_batches = n_records // batch_size + 1
        i = 0
        for n in range(n_batches):
            begin = n * batch_size
            end = begin + batch_size
            for record in self.select(
                self.columns[self.idx_name].in_(sample_idxs[begin:end]),
                self.vector_name,
            ):
                X[i] = record[self.vector_name]
                i += 1

        return X

    def similarity(
        self, v1: np.ndarray, v2: np.ndarray, threshold: float = None
    ) -> float:
        """
        Calculate the similarity between two vectors using the metric specified
        in ``create_index()``. Currently only the inner product and L2 are supported. If
        the similarity fails to meet the threshold, ``None`` is returned.

        Parameters
        ----------
        v1 : np.ndarray
        v2 : np.ndarray
        threshold : float, optional
            Only return the value if similarity equals or exceeds this value. Default
            is ``None``.

        Returns
        -------
        float
            similarity of v1 & v2
        """
        v1 = v1.reshape(-1)
        v2 = v2.reshape(-1)
        if self.metric == "inner_product":
            similarity = v1.dot(v2)
            if threshold is None:
                return similarity
            elif similarity >= threshold:
                return similarity
            else:
                return None
        elif self.metric == "l2":
            similarity = np.linalg.norm(v1 - v2)
            if threshold is None:
                return similarity
            elif similarity <= threshold:
                return similarity
            else:
                return None
        else:
            raise ValueError(
                f"Not properly handling {self.metric} metric. "
                + "Only inner_product and l2 are currently supported"
            )

    def set_faiss_runtime_parameters(self, runtime_params_str: str):
        """
        Change FAISS runtime parameters with a human-readable string. Parameters are
        separated by commas. For example, with the index 'OPQ64,IVF50000_HNSW32,PQ64',
        you can use "nprobe=50,quantizer_efSearch=100" to set both the nprobe in the
        IVF index and the efSearch in the HNSW quantizer index. If a parameter is not
        recognized, an exception is thrown.

        Saves the provided settings in ``self.faiss_runtime_parameters``

        Parameters
        ----------
        runtime_params_str : str
            Comma-separated list of parameters to set. For more details, see
            https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning#parameterspace-as-a-way-to-set-parameters-on-an-opaque-index
        """
        try:
            faiss.ParameterSpace().set_index_parameters(self.index, runtime_params_str)
            self.faiss_runtime_parameters = runtime_params_str
            self.logger.info(f"faiss_runtime_parameters: set to {runtime_params_str}")
        except Exception as exc:
            raise ValueError(f"Unrecognized parameter in {runtime_params_str}. {exc}")

    def create_index(
        self,
        faiss_index: str,
        faiss_factory_str: str,
        metric: str = "inner_product",
        sample_size: int = 0,
        batch_size: int = 50_000,
        faiss_runtime_params: str = None,
    ):
        """
        Create a FAISS index. Train, if needed, using ``sample_size`` vectors. Add
        vectors from database table to index. Save to disk when completed.

        Parameters
        ----------
        faiss_index : str
            Name of the file to save the resulting FAISS index to.
        faiss_factory_str : str
            FAISS index factory string, passed to ``faiss.index_factory()``
        metric : str, optional
            Metric used by FAISS to determine similarity. Valid values are either
            ``inner_product`` or ``L2``. Default is ``inner_product``
        sample_size : int, optional
            Number of training vectors. If 0, uses all vectors. Default is 0.
        batch_size : int, optional
            Passed to ``sample_vectors()`` to pull vectors in batches of this size.
            Default is 50,000.
        faiss_runtime_params : str, optional
            Set FAISS index runtime parameters before adding vectors. Useful if using
            a quantizer index (e.g., IVF50000_HNSW32) since ``index.add()`` searches
            against the quantizer index. E.g., "quantizer_efSearch=40".
            ``self.faiss_runtime_parameters`` is set back to its value before function
            invocation. Default is ``None``.

        Raises
        ------
        FileExistsError
            If a FAISS index is already assigned to this table.
        FileExistsError
            If the file ``faiss_index`` already exists on disk.
        TypeError
            If the metric is not either "inner_product" | "L2"
        """
        if self.index is not None:
            raise FileExistsError(
                f"Index at {self.faiss_index} has already been assigned to this table"
            )
        if faiss_index == self.faiss_index:
            raise FileExistsError(f"{self.faiss_index} file already exists")
        else:
            self.faiss_index = faiss_index

        metric = metric.lower()
        self.metric = metric
        if metric == "inner_product":
            metric = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            metric = faiss.METRIC_L2
        else:
            raise TypeError(f"You gave {metric=}, but it must be inner_product | l2")

        self.index = faiss.index_factory(self.d, faiss_factory_str, metric)

        # See if you can set the runtime parameters provided. If not, then it will throw
        # an exception, but at least do this before training
        orig_runtime_parameters = ""
        if faiss_runtime_params:
            try:
                orig_runtime_parameters = self.faiss_runtime_parameters
            except:
                pass
            self.set_faiss_runtime_parameters(faiss_runtime_params)

        # Needs to be trained
        if not self.index.is_trained:
            self.train_index(sample_size, batch_size, save_to_disk=True)

        # Add records into the index and then write it to disk.
        self.sync_index_to_db(batch_size=batch_size)

        if orig_runtime_parameters:
            self.set_faiss_runtime_parameters(orig_runtime_parameters)

    def train_index(
        self, sample_size: int = 0, batch_size: int = 50_000, save_to_disk: bool = True
    ):
        """
        Pull vectors from database table to train the FAISS index.

        Parameters
        ----------
        sample_size : int, optional
            Number of training vectors. If 0, uses all vectors. Default is 0.
        batch_size : int, optional
            Passed to ``sample_vectors()`` to pull vectors in batches of this size.
            Default is 50,000.
        save_to_disk : bool, optional
            Save trained index to ``self.faiss_index`` on disk. Default is ``True``.
        """
        X_train = self.sample_vectors(sample_size, batch_size)
        start = datetime.now()
        self.index.train(X_train)
        end = datetime.now()
        self.logger.info(
            f"train_index: Training with {X_train.shape[0]} records took {end-start}"
        )
        if save_to_disk:
            # Save the index to disk
            start = datetime.now()
            faiss.write_index(self.index, self.faiss_index)
            end = datetime.now()
            self.logger.info(
                f"train_index: Saving to {self.faiss_index} took {end-start}"
            )

    def sync_index_to_db(
        self, batch_size: int = 100_000, faiss_runtime_params: str = None
    ) -> int:
        """
        Add any vectors from the database that are not in the FAISS index.

        Parameters
        ----------
        batch_size : int, optional
            Add vectors in batches of this size. Default is 100_000.
        faiss_runtime_params : str, optional
            Set FAISS index runtime parameters before adding vectors. Useful if using
            a quantizer index (e.g., IVF50000_HNSW32) since ``index.add()`` searches
            against the quantizer index. E.g., "quantizer_efSearch=40".
            ``self.faiss_runtime_parameters`` is set back to its value before function
            invocation. Default is ``None``.

        Returns
        -------
        int
            Number of records added into FAISS index

        Raises
        ------
        IndexError
            Primary key of the records to be added must have consecutive integer values
        """
        orig_runtime_parameters = ""
        if faiss_runtime_params:
            try:
                orig_runtime_parameters = self.faiss_runtime_parameters
            except:
                pass
            self.set_faiss_runtime_parameters(faiss_runtime_params)

        start_idx = self.index.ntotal

        # Check that we have consecutive idx in the database
        with self.Session() as session:
            idx_col = self.columns[self.idx_name]
            stmt = sa.select(sa.func.max(idx_col))
            stop_idx = session.scalar(stmt) + 1
            stmt = sa.select(
                sa.func.count(idx_col),
            ).where(
                sa.sql.and_(
                    idx_col >= start_idx,
                    idx_col < stop_idx,
                ),
            )
            num_new_rows = session.scalar(stmt)

        if num_new_rows != (stop_idx - start_idx):
            raise IndexError(
                f"Missing records in database for {self.idx_name} between "
                + f"[{start_idx}, {stop_idx}). Must have consecutive values."
            )

        # Add records into the index
        start = datetime.now()
        n_vectors = 0
        with self.Session() as session:
            idx_col = self.columns[self.idx_name]
            vector_col = self.columns[self.vector_name]
            vectors = []
            stmt = (
                sa.select(vector_col)
                .where(sa.sql.and_(idx_col >= start_idx, idx_col < stop_idx))
                .order_by(idx_col)
            )
            for vector_bytes in session.scalars(stmt):
                vectors.append(self.deserialize_vector(vector_bytes))
                if len(vectors) == batch_size:
                    self.index.add(np.vstack(vectors))
                    n_vectors += len(vectors)
                    vectors = []
                    self.logger.debug(
                        f"sync_index_to_db: batch added, cumulatively {n_vectors} have "
                        + "been added"
                    )

            if vectors:
                self.index.add(np.vstack(vectors))
                n_vectors += len(vectors)

        if orig_runtime_parameters:
            self.set_faiss_runtime_parameters(orig_runtime_parameters)

        end = datetime.now()
        rate = n_vectors / (end - start).total_seconds()
        self.logger.info(
            f"sync_index_to_db: {n_vectors} added at {rate:.2f} vectors / sec"
        )

        if num_new_rows != n_vectors:
            raise ValueError(
                f"Expected to add {num_new_rows}, but only added {n_vectors}"
            )

        # Save the index to disk
        start = datetime.now()
        faiss.write_index(self.index, self.faiss_index)
        end = datetime.now()
        self.logger.info(
            f"sync_index_to_db: Saving {self.faiss_index} to disk took {end - start}"
        )

        return num_new_rows

    def search(
        self,
        query_vectors: np.ndarray,
        k_nearest_neighbors: int,
        *col_names: str,
        k_extra_neighbors: int = 0,
        rerank: bool = True,
        threshold: float = None,
        search_parameters=None,
        batch_size: int = 10_000,
    ) -> List[List[Dict]]:
        """
        Search for the ``k_nearest_neighbors`` records in the database table based
        on the similarity of their vectors to the query vectors. Optionally keep
        only neighbors whose similarity exceeds the ``threshold``.

        Parameters
        ----------
        query_vectors : np.ndarray
            The query vectors to search with. Shape is (n, d) and dtype is np.float32
        k_nearest_neighbors : int
            Number of nearest neighbors to return.
        \*col_names : str
            List of columns to use in a neighbor record. Default of ``None`` uses
            all columns.
        k_extra_neighbors : int, optional
            Extra neighbors to return from FAISS index before reranking. If using a
            vector quantizer (e.g., PQ), FAISS orders results based upon the estimated
            similarities which likely differs from the true similarities calculated
            here. Default is 0.
        rerank : bool, optional
            If ``True``, rerank neighbors according to their true similarities.
            Otherwise the order is determined by the FAISS index's ``index.search()``.
            Default is ``True``.
        threshold : float, optional
            Only keep neighbors whose similarities exceed the ``threshold``. Default is
            ``None`` which keeps all neighbors returned.
        search_parameters : faiss.SearchParameters, optional
            Use these search parameters instead of the current runtime FAISS
            parameters. Passed to FAISS's ``index.search()``. See
            [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query)
        batch_size : int, optional
            Batch size to use when retrieving neighbors information from the database
            table. Default is 10,000.

        Returns
        -------
        List[Dict]
            For each query, a dictionary containing "neighbors" key whose value is a list
            of the neighbors' requested information including the "metric" similarity to
            the query vector.
        """

        if len(query_vectors.shape) == 1 and query_vectors.shape[0] == self.d:
            query_vectors = query_vectors.reshape(1, self.d)
        elif len(query_vectors.shape) == 2 and query_vectors.shape[1] != self.d:
            raise ValueError(f"query_vectors dimension is not {self.d}")
        elif len(query_vectors.shape) != 2:
            raise ValueError(
                f"query_vectors is not (d,) or (n, d). You gave {query_vectors.shape}"
            )
        k = k_nearest_neighbors + k_extra_neighbors
        _, I = self.index.search(query_vectors, k, params=search_parameters)
        idx_neighbors = np.unique(I).tolist()

        # Get records for all the neighbors
        neighbor_records = {}
        if not col_names:
            col_names = list(self.columns.keys())
        # I need both the idx and vector columns so add those in if not requested.
        tmp_col_names = list(col_names)
        if self.idx_name not in tmp_col_names:
            tmp_col_names.append(self.idx_name)
        if self.vector_name not in tmp_col_names:
            tmp_col_names.append(self.vector_name)
        tmp_col_names = tuple(tmp_col_names)

        n_records = len(idx_neighbors)
        n_batches = n_records // batch_size + 1
        for n in range(n_batches):
            begin = n * batch_size
            end = begin + batch_size
            for record in self.select(
                self.columns[self.idx_name].in_(idx_neighbors[begin:end]),
                *tmp_col_names,
            ):
                neighbor_records[record[self.idx_name]] = record

        # For each query, check that neighbors are close enough and reorder accordingly
        results = []
        for query_vec, I_row in zip(query_vectors, I):
            query_result = []
            for idx in I_row:
                similarity = self.similarity(
                    query_vec, neighbor_records[idx][self.vector_name], threshold
                )
                if similarity is not None:
                    neighbor = neighbor_records[idx].copy()
                    if self.idx_name not in col_names:
                        neighbor.pop(self.idx_name)
                    if self.vector_name not in col_names:
                        neighbor.pop(self.vector_name)
                    neighbor["metric"] = similarity
                    query_result.append(neighbor)
            if rerank:
                if self.metric == "inner_product":
                    # Use descending order
                    query_result = sorted(
                        query_result, key=lambda x: x.get("metric"), reverse=True
                    )
                elif self.metric == "l2":
                    # Use ascending order
                    query_result = sorted(
                        query_result, key=lambda x: x.get("metric"), reverse=False
                    )
                else:
                    raise ValueError(f"Not properly handling {self.metric} metric")
            results.append({"neighbors": query_result[:k_nearest_neighbors]})

        return results

    def nearest_neighbors(
        self,
        where_clause: sa.sql.ClauseElement,
        k_nearest_neighbors: int,
        *col_names: str,
        k_extra_neighbors: int = 0,
        rerank: bool = True,
        threshold: float = None,
        search_parameters=None,
        batch_size: int = 10_000,
    ) -> List[Dict]:
        """
        Return nearest neighbors of the records in the database table that match the
        ``where_clause``. Optionally keep only neighbors whose similarity exceeds the
        ``threshold``.

        Parameters
        ----------
        where_clause : sa.sql.ClauseElement
            Where clause specifying records of interest. Passed to ``select()``
        k_nearest_neighbors : int
            Number of nearest neighbors to return.
        \*col_names : str
            List of columns to use in the query and neighbor records. Default of
            ``None`` uses all columns.
        k_extra_neighbors : int, optional
            Extra neighbors to return from FAISS index before reranking. If using a
            vector quantizer (e.g., PQ), FAISS orders results based upon the estimated
            similarities which likely differs from the true similarities calculated
            here. Default is 0.
        rerank : bool, optional
            If ``True``, rerank neighbors according to their true similarities.
            Otherwise the order is determined by the FAISS index's ``index.search()``.
            Default is ``True``.
        threshold : float, optional
            Only keep neighbors whose similarities exceed the ``threshold``. Default
            is ``None`` which keeps all neighbors returned.
        search_parameters : faiss.SearchParameters, optional
            Use these search parameters instead of the current runtime FAISS
            parameters. Passed to FAISS's ``index.search()``. See
            [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query)
        batch_size : int, optional
            Batch size to use when retrieving neighbors information from the database
            table. Default is 10,000.

        Returns
        -------
        List[Dict]
            For each query, a dictionary containing the query's record and a list of
            the its neighbors' records. A neighbor record includes the "metric"
            similarity.
        """
        if not col_names:
            col_names = list(self.columns.keys())

        # Both idx and vector columns needed when fetching records
        tmp_col_names = list(col_names)
        if self.idx_name not in tmp_col_names:
            tmp_col_names.append(self.idx_name)
        if self.vector_name not in tmp_col_names:
            tmp_col_names.append(self.vector_name)

        results = []
        query_vectors = []
        for record in self.select(where_clause, *tmp_col_names):
            results.append(record)
            query_vectors.append(record[self.vector_name])

        query_vectors = np.vstack(query_vectors)
        # Need the idx column for neighbors in order to identify yourself
        tmp_col_names_neighbors = list(col_names)
        if self.idx_name not in tmp_col_names_neighbors:
            tmp_col_names_neighbors.append(self.idx_name)

        for record, search_result in zip(
            results,
            self.search(
                query_vectors,
                k_nearest_neighbors + 1,
                *tmp_col_names_neighbors,
                k_extra_neighbors=k_extra_neighbors + 1,
                rerank=rerank,
                threshold=threshold,
                search_parameters=search_parameters,
                batch_size=batch_size,
            ),
        ):
            pop_idx_name = self.idx_name not in col_names
            pop_vector_name = self.vector_name not in col_names
            neighbors_without_yourself = []
            for neighbor in search_result["neighbors"]:
                if neighbor[self.idx_name] != record[self.idx_name]:
                    if pop_idx_name:
                        neighbor.pop(self.idx_name)
                    neighbors_without_yourself.append(neighbor)
            record["neighbors"] = neighbors_without_yourself[:k_nearest_neighbors]
            if pop_idx_name:
                record.pop(self.idx_name)
            if pop_vector_name:
                record.pop(self.vector_name)

        return results

    def select(
        self,
        where_clause: sa.sql.ClauseElement,
        *ret_cols: str,
    ) -> Iterator[Dict]:
        """
        Select records from the database table. Class member ``Record`` is the
        ORM-mapped class for the database table and should be used to construct the
        clause.

        .. code-block:: python

            where = vekter_db.Record.idx == 0
            where = vekter_db.Record.idx.in_([100, 200, 300])
            where = sa.sql.and_(vekter_db.Record.idx>=0, vekter_db.Record.idx<5)

            vekter_db.select(where)                   # Return all columns
            vekter_db.select(where, "idx", "vector")  # Return idx & vector only


        Parameters
        ----------
        where_clause : sa.sql.ClauseElement
            SQLAlchemy Where Clause.
        \*ret_cols : str, optional
            List columns to return from the database table. Default of ``None`` returns
            all columns.

        Yields
        ------
        Iterator[Dict]
            Dictionary of requested fields that match the where clause
        """
        if not isinstance(where_clause, sa.sql.ClauseElement):
            raise ValueError(f"{where_clause=:} is not a sa.sql.ClauseElement")

        if not ret_cols:
            ret_cols = tuple(self.columns.keys())

        with self.Session() as session:
            stmt = sa.select(self.Record).where(where_clause)
            for row in session.scalars(stmt):
                record = {}
                for col_name in ret_cols:
                    value = row.__dict__[col_name]
                    if col_name == self.vector_name:
                        value = self.deserialize_vector(value)
                    record[col_name] = value
                yield record
