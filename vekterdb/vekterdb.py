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
import json
import logging
import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy.sql.functions as sa_funcs
from typing import Dict, Iterable, Iterator, List


class VekterDB:
    """Instantiate a VekterDB"""

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
        Initialize VekterDB with a minimum of two columns: idx_name (BigInteger,
        default 'idx') and vector (LargeBinary, default 'vector'). Vectors are numpy
        arrays of np.float32, serialized to bytes using tobytes(), to comply with
        FAISS requirements.

        Add additional columns with `columns_dict` using SQLAlchemy's `Column`
        arguments. For example, include a unique, indexed id field (str) and a
        non-unique product_category field (str) with:

        ```
        my_db = VekterDB(
            "my_table",
            columns_dict={
                "id": {"type": Text, "unique": True, "nullable": False, "index": True},
                "product_category": {"type": Text},
            }
        )
        ```

        Parameters
        ----------
        table_name : str
            Database table name.
        idx_name : str, optional
            Column name that stores the FAISS index integer ID and is the primary key
            for the database table. It must be unique and consecutive from
            [0, n_records). The default name is "idx".
        vector_name : str, optional
            Column name that stores the vector information. Default is "vector".
        columns_dict : Dict[str, Dict], optional
            Names (key) of additional columns to include in the table. The values are
            arguments that will be passed to SQLAlchemy's `Column`. Default is {}.

            When connecting to an existing database table, this argument is not
            necessary.
        url : str, optional
            URL string to connect to the database. Passed to SQLAlchemy's
            `create_engine`. Default is "sqlite:///:memory"; an in-memory database.
        connect_args: Dict, optional
            Any connection arguments to pass to SQLAlchemy's `create_engine`.
            Default is {}.
        faiss_index : str, optional
            If given, then load an existing FAISS index saved by that name. Default
            is None.
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
            self.logger.warning(f"Creating {table_name} table in the database")
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
        else:
            self.faiss_index = None
            self.index = None
            self.metric = None

    @staticmethod
    def serialize_vector(vector: np.ndarray) -> bytes:
        """
        Static method to construct Python bytes containing the raw data bytes in
        `vector`.

        Parameters
        ----------
        vector : np.ndarray
            1-d numpy array of type np.float32

        Returns
        -------
        bytes
        """
        return vector.tobytes()

    @staticmethod
    def deserialize_vector(vector_bytes: bytes) -> np.ndarray:
        """
        Static method to interpret `vector_bytes` as a 1-dimensional array

        Parameters
        ----------
        vector_bytes : bytes
            Bytes representation of a vector.

        Returns
        -------
        np.ndarray
            1-d numpy array of type np.float32
        """
        return np.frombuffer(vector_bytes, dtype=np.float32)

    def save(self, config_file: str = None):
        """
        Saves configuration info to a JSON file, excluding the URL string for security.
        If `set_faiss_runtime_parameters()` has been called, it also saves and applies
        that setting when loading with `VekterDB.load()`

        Parameters
        ----------
        config_file : str, optional
            JSON file name to save to disk. If not provided, saves the file as
            `table_name`.json. The default is None.
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
            URL string to connect to the database. See sa.create_engine() for details
        connect_args: Dict, optional
            Any connection arguments to pass to sa.create_engine(). Default is {}

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
            If True, then vectors will be serialized before insertion; if False, the
            vectors have already been serialized to bytes. Default is True.
        faiss_runtime_params : str, optional
            Set FAISS index runtime parameters before adding vectors. Likely only
            useful if you have a quantizer index (e.g., IVF12345_HNSW32). The quantizer
            index (HNSW32) will be used during the `index.add()` to determine which
            partition to add the vector to. You may want to change from the default,
            whether that is the FAISS default (efSearch=16) or the value saved in
            `self.faiss_runtime_parameters`. "quantizer_efSearch=40" would be an
            example value for the example index given. If used,
            `self.faiss_runtime_parameters` is set back to its value before function
            invocation. Default is `None`

        Returns
        -------
        n_records : int
            Number of records added to the table
        """
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
                + "records will only be inserted into the database."
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
                        record[self.vector_name]
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
            if len(batch) > 0:
                session.execute(sa.insert(self.Record), batch)
                if insert_into_faiss:
                    self.index.add(np.vstack(vectors))
            session.commit()

        if orig_runtime_parameters:
            self.set_faiss_runtime_parameters(orig_runtime_parameters)

        if insert_into_faiss:
            # Save the index to disk
            faiss.write_index(self.index, self.faiss_index)

        return n_records

    def sample_vectors(
        self, sample_size: int = 0, batch_size: int = 10_000
    ) -> np.ndarray:
        """Retrieve a sample of vectors from the database

        Parameters
        ----------
        sample_size : int, optional
            Number of vectors to return. Default 0 returns all vectors
        batch_size : int, optional
            Number of vectors to retrieve at one time. Default 10_000.

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
        for i, record in enumerate(
            self.fetch_records(
                self.idx_name, sample_idxs, self.vector_name, batch_size=batch_size
            )
        ):
            X[i] = record[self.vector_name]

        return X

    def similarity(
        self, v1: np.ndarray, v2: np.ndarray, threshold: float = None
    ) -> float:
        """
        Calculate the similarity between two vectors using using the metric specified
        in `create_index()`. Currently only the inner product and L2 are supported. If
        the similarity fails to meet the threshold, `None` is returned.

        Parameters
        ----------
        v1 : np.ndarray
        v2 : np.ndarray
        threshold : float, optional
            Only return the value if similarity equals or exceeds this value. Default
            is None.

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

        Saves the provided settings in `self.faiss_runtime_parameters`

        Parameters
        ----------
        runtime_params_str : str
            Comma-separated list of parameters to set. For more details, see
            https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning#parameterspace-as-a-way-to-set-parameters-on-an-opaque-index
        """
        try:
            faiss.ParameterSpace().set_index_parameters(self.index, runtime_params_str)
            self.faiss_runtime_parameters = runtime_params_str
        except Exception as exc:
            raise ValueError(f"Unrecognized parameter in {runtime_params_str}. {exc}")

    def create_index(
        self,
        faiss_index: str,
        faiss_factory_str: str,
        metric: str = "inner_product",
        sample_size: int = 0,
        batch_size: int = 10_000,
        use_gpu: bool = False,
        faiss_runtime_params: str = None,
    ):
        """
        Create a FAISS index and save to disk when completed.

        Parameters
        ----------
        faiss_index : str
            Name of the file to save the resulting FAISS index to.
        faiss_factory_str : str
            FAISS index factory string, passed to `faiss.index_factory()`
        metric : str, optional
            Metric used by FAISS to determine similarity. Valid values are either
            `inner_product` or `L2`. Default is `inner_product`
        sample_size : int, optional
            Number of vectors to sample from the database to train the FAISS index.
            If 0, then all vectors are used. Default is 0.
        batch_size : int, optional
            Number of vectors to add into the index at one time. Also passed to
            `sample_vectors()` to specify number of vectors to pull back from the
            database at a time. Default is 10,000.
        use_gpu : bool, optional
            Whether to use GPU(s). Default is False. NOTE: Not implemented yet
        faiss_runtime_params : str, optional
            Set FAISS index runtime parameters before adding vectors. Likely only
            useful if using a quantizer index (e.g., IVF12345_HNSW32). The quantizer
            index (HNSW32) will be used during the `index.add()` to determine which
            partition to add the vector to. You may want to change from the default,
            whether that is the FAISS default (efSearch=16) or the value saved in
            `self.faiss_runtime_parameters`. "quantizer_efSearch=40" would be an
            example value for the example index given. If used,
            `self.faiss_runtime_parameters` is set back to its value before function
            invocation. Default is None

        Raises
        ------
        FileExistsError
            If a FAISS index is already assigned to this table.
        FileExistsError
            If the file `faiss_index` already exists on disk.
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
            X_train = self.sample_vectors(sample_size, batch_size)
            self.index.train(X_train)
            # TODO : Add option to train IVF on GPU
            # TODO : Add ability to save a trained index and then start from there

        # Add records into the index
        with self.Session() as session:
            vectors = []
            stmt = sa.select(self.columns[self.vector_name]).order_by(
                self.columns[self.idx_name]
            )
            for vector_bytes in session.scalars(stmt):
                vectors.append(self.deserialize_vector(vector_bytes))
                if len(vectors) == batch_size:
                    self.index.add(np.vstack(vectors))
                    vectors = []
            if vectors:
                self.index.add(np.vstack(vectors))

        if orig_runtime_parameters:
            self.set_faiss_runtime_parameters(orig_runtime_parameters)

        # Save the index to disk
        faiss.write_index(self.index, self.faiss_index)

    def search(
        self,
        query_vectors: np.ndarray,
        k_nearest_neighbors: int,
        *col_names: str,
        k_extra_neighbors: int = 0,
        rerank: bool = True,
        threshold: float = None,
        search_parameters: faiss.SearchParameters = None,
    ) -> List[List[Dict]]:
        """
        Search for the `k_nearest_neighbors` records in the database based on the
        similarity of their vectors to the query vectors. Optionally keep only
        neighbors whose similarity exceeds the `threshold`.

        Parameters
        ----------
        query_vectors : np.ndarray
            The query vectors to search with. Shape is (n, d) and dtype is np.float32
        k_nearest_neighbors : int
            Number of nearest neighbors to return.
        \*col_names : str
            List of columns to use in a neighbor record. Default of `None` uses all
            columns.
        k_extra_neighbors : int, optional
            Extra neighbors to return from FAISS index before reranking. If using a
            vector quantizer (e.g., PQ), FAISS orders results based upon the estimated
            similarities which likely differs from the true similarities calculated
            here. Default is 0.
        rerank : bool, optional
            If True, rerank neighbors according to their true similarities. Otherwise
            the order is determined by the FAISS index's `index.search()`. Default is
            True.
        threshold : float, optional
            Only keep neighbors whose similarities exceed the `threshold`. Default is
            `None` which keeps all neighbors returned.
        search_parameters : faiss.SearchParameters, optional
            Use these search parameters instead of the current runtime FAISS
            parameters. Passed to FAISS's `index.search()`. See
            [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query)

        Returns
        -------
        List[List[Dict]]
            For each query, return a list of the neighbors. A neighbor record includes
            the "metric" similarity with the query vector.
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
        for record in self.fetch_records(self.idx_name, idx_neighbors, *tmp_col_names):
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
            results.append(query_result[:k_nearest_neighbors])

        return results

    def nearest_neighbors(
        self,
        fetch_column: str,
        fetch_values: List,
        k_nearest_neighbors: int,
        *col_names: str,
        k_extra_neighbors: int = 0,
        rerank: bool = True,
        threshold: float = None,
        search_parameters: faiss.SearchParameters = None,
        batch_size: int = 10_000,
    ) -> List[Dict]:
        """
        Find the nearest neighbors of query records in the table based on vector
        similarity. Optionally keep only neighbors whose similarity exceeds the
        `threshold`.

        Parameters
        ----------
        fetch_column : str
            Column in the database for query record retrieval.
        fetch_values : List
            Values to match in the `fetch_column`.
        k_nearest_neighbors : int
            Number of nearest neighbors to return.
        \*col_names : str
            List of columns to use in the query and neighbor records. Default of `None`
            uses all columns.
        k_extra_neighbors : int, optional
            Extra neighbors to return from FAISS index before reranking. If using a
            vector quantizer (e.g., PQ), FAISS orders results based upon the estimated
            similarities which likely differs from the true similarities calculated
            here. Default is 0.
        rerank : bool, optional
            If True, rerank neighbors according to their true similarities. Otherwise
            the order is determined by the FAISS index's `index.search()`. Default is
            True.
        threshold : float, optional
            Only keep neighbors whose similarities exceed the `threshold`. Default is
            `None` which keeps all neighbors returned.
        search_parameters : faiss.SearchParameters, optional
            Use these search parameters instead of the current runtime FAISS
            parameters. Passed to FAISS's `index.search()`. See
            [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query)
        batch_size : int, optional
            Number of query records to retrieve from the database at one time. Passed
            to `fetch_records()`. Default is 10,000.

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
        for record in self.fetch_records(
            fetch_column,
            fetch_values,
            *tmp_col_names,
            batch_size=batch_size,
        ):
            results.append(record)
            query_vectors.append(record[self.vector_name])

        query_vectors = np.vstack(query_vectors)
        # Need the idx column for neighbors in order to identify yourself
        tmp_col_names_neighbors = list(col_names)
        if self.idx_name not in tmp_col_names_neighbors:
            tmp_col_names_neighbors.append(self.idx_name)

        for record, neighbors in zip(
            results,
            self.search(
                query_vectors,
                k_nearest_neighbors + 1,
                *tmp_col_names_neighbors,
                k_extra_neighbors=k_extra_neighbors + 1,
                rerank=rerank,
                threshold=threshold,
                search_parameters=search_parameters,
            ),
        ):
            pop_idx_name = self.idx_name not in col_names
            pop_vector_name = self.vector_name not in col_names
            neighbors_without_yourself = []
            for neighbor in neighbors:
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

    def fetch_records(
        self,
        fetch_column: str,
        fetch_values: List,
        *col_names: str,
        batch_size: int = 10_000,
    ) -> Iterator[Dict]:
        """
        Fetch records from the database.

        Parameters
        ----------
        fetch_column : str
            Column in the database used for record retrieval.
        fetch_values : List
            Values to match in the `fetch_column`.
        \*col_names : str, optional
            List columns to return from the database. Default of `None` returns all
            columns.
        batch_size : int, optional
            Number of records to fetch from the database at one time. Default is
            10,000.

        Yields
        ------
        Iterator[Dict]
            Dictionary containing a fetched record.
        """

        if not (
            self.columns[fetch_column].index or self.columns[fetch_column].primary_key
        ):
            self.logger.warning(
                f"{fetch_column} is not indexed in the database. This will be slow."
            )
        n_records = len(fetch_values)
        n_batches = n_records // batch_size + 1

        if not col_names:
            col_names = tuple(self.columns.keys())

        with self.Session() as session:
            for n in range(n_batches):
                begin = n * batch_size
                end = begin + batch_size
                stmt = sa.select(self.Record).where(
                    self.columns[fetch_column].in_(fetch_values[begin:end])
                )
                for row in session.scalars(stmt):
                    record = {}
                    for col_name in col_names:
                        value = row.__dict__[col_name]
                        if col_name == self.vector_name:
                            value = self.deserialize_vector(value)
                        record[col_name] = value
                    yield record
