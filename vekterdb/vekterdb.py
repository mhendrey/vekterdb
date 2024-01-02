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
import logging
import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, sessionmaker
import sqlalchemy.sql.functions as sa_funcs
from typing import Dict, Iterable, Iterator, List


class VekterDB:
    """Instantiate a VekterDB"""

    class Base(DeclarativeBase):
        pass

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
        Initialize the VekterDB. Two columns will be automatically included:
            idx [BigInteger] = Unique integer based id used by FAISS
            vector [List[Float]] = Stores vector of np.float32 values

        For example, let's add two additional columns. The first is a string id field
        which should also be unique and I also want to index it in order to query for
        records by the easier to use "id" field. The second is a product category which
        is not unique.
        ```
        my_db = VekterDB(
            "my_table",
            "idx",
            columns_dict={
                "id": {"type": Text, "unique": True, "nullable": False, "index": True},
                "product_category": {"type": Text},
            }
        )
        ```

        Parameters
        ----------
        table_name : str
            Database table name
        idx_name : str, optional
            Column name that stores the integer ID. This must be unique for each
            record and be consecutive from [0, n_records). This is used by the FAISS
            index. Default is "idx".
        vector_name : str, optional
            Column name that stores the vector information. Default is "vector".
        columns_dict : Dict[str, Dict], optional
            Names (key) of additional columns to include in the table. The value is
            a dictionary that must have the "type" which tells what SQLAlchemy type
            will be stored. Additional values will be passed as kwargs to SQLAlchemy
            Column. Default is an empty dictionary.
        url : str, optional
            URL string to connect to the database, by default "sqlite:///:memory" which
            is an in-memory database.
        connect_args: Dict, optional
            Any connection arguments to pass to the create_engine(). Default is {}
        faiss_index : str, optional
            If given, then load the existing FAISS index. Default is None
        """
        self.idx_name = idx_name
        self.vector_name = vector_name

        self.engine = sa.create_engine(url, **connect_args)
        self.Session = sessionmaker(bind=self.engine)

        # I'd like to try see if the table exists. If it does then I can
        # get the self.columns from the existing table.  And maybe define the
        # class VekterMapping based upon those?

        if faiss_index:
            self.faiss_index = faiss_index
            self.index = faiss.read_index(faiss_index)
            self.d = self.index.d
        else:
            self.faiss_index = None
            self.index = None
            self.d = None

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

        # Declare table mapping
        class VekterMapping(self.Base):
            __table__ = sa.Table(
                table_name,
                self.Base.metadata,
                *self.columns.values(),
            )

        self.Record = VekterMapping
        self.Base.metadata.create_all(self.engine)

    @staticmethod
    def serialize_vector(vector: np.ndarray):
        return vector.tobytes()

    @staticmethod
    def deserialize_vector(vector_bytes: bytes):
        return np.frombuffer(vector_bytes, dtype=np.float32)

    def insert(
        self,
        records: Iterable[Dict],
        batch_size: int = 10_000,
        serialize_vectors: bool = True,
    ) -> int:
        """
        Insert multiple records into the table. Simplest example would be

        ```
        records = [
            {"idx": 0, "id": "a", "vector": np.array([0, 1, 2, 3], dtype=np.float32)},
            {"idx": 1, "id": "b", "vector": np.array([3, 2, 1, 0], dtype=np.float32)},
        ]
        ```

        Parameters
        ----------
        records : Iterable[Dict]
            Each element of the Iterable should be a dictionary with a key the column
            name and the value is the corresponding value.
        batch_size : int, optional
            Batch size to push to the database at one time. Default is 5,000.
        serialize_vectors : bool, optional
            If True, then vectors will be serialized, using numpy's tobytes(), before
            inserting. If False, then the user has already converted numpy arrays to
            bytes and can be put directly into the database. Default is True.

        Returns
        -------
        n_records : int
            Number of records added to the table
        """
        # TODO: Allow passing of SearchParams when adding into faiss index.
        insert_into_faiss = False
        if self.index is not None and self.index.is_trained:
            insert_into_faiss = True
        if not insert_into_faiss:
            logging.warning(
                "FAISS index either doesn't exist or isn't trained so "
                + "records will only be inserted into the database."
                + ". Call create_index() to make the FAISS index."
            )

        n_records = 0
        with self.Session() as session:
            batch = []
            vectors = []
            for i, record in enumerate(records):
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
            if insert_into_faiss:
                # Save the index to disk
                faiss.write_index(self.index, self.faiss_index)

        return n_records

    def sample_vectors(
        self, sample_size: int = 0, batch_size: int = 10_000
    ) -> np.ndarray:
        """Retrieve sample vectors from the database

        Parameters
        ----------
        sample_size : int, optional
            Number of vectors to return. Default 0 means get all of them.
        batch_size : int, optional
            Number of vectors to retrieve at one time. Default 10_000.

        Returns
        -------
        np.ndarray
            2-d array of sampled vectors with shape is (sample_size, d)
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
        """Return the appropriate similarity metric between vector v1 and v2.
        Currently 'inner_product' and 'l2' are supported. If the similarity
        falls short of the threshold, then None is returned.

        Parameters
        ----------
        v1 : np.ndarray
            Shape should be (d,) and dtype is np.float32
        v2 : np.ndarray
            Shape should be (d,) and dtype is np.float32
        threshold : float, optional
            Only return the value if v1 & v2's similarity equals or exceeds this value.
            Default is None which always returns

        Returns
        -------
        float
            similarity of v1 & v2
        """

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

    def create_index(
        self,
        faiss_index: str,
        faiss_factory_str: str,
        metric: str = "inner_product",
        sample_size: int = 0,
        batch_size: int = 10_000,
        use_gpu: bool = False,
    ):
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
            self.keep_below = False
        else:
            raise TypeError(f"You gave {metric=}, but it must be inner_product | l2")

        self.index = faiss.index_factory(self.d, faiss_factory_str, metric)

        # Needs to be trained
        if not self.index.is_trained:
            X_train = self.sample_vectors(sample_size, batch_size)
            self.index.train(X_train)
            # TODO : Add option to train IVF on GPU

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
        For each query vector, find the `k_nearest_neighbors` records in the database
        based whose vectors are the most similar to the query vector. If `threshold`
        is provided, then drop any candidate neighbors whose similarity fails to meet
        that value.

        If you proved any `col_names`, then only those columns from the database will
        be returned for each neighbor in the dict. Each dict will add "metric" key
        which holds the similarity between the query vector and that neighbor's vector

        Parameters
        ----------
        query_vectors : np.ndarray
            Vector(s) to search with. Shape should be (n, d). This is the same as FAISS
        k_nearest_neighbors : int
            Number of nearest neighbors to return.
        \*col_names : str
            Specify specific columns from the database to be returned for each
            neighbor. If none are provided, then all columns are returned.
        k_extra_neighbors : int, optional
            Extra neighbors to return from FAISS index before reranking. Default 0
        rerank : bool, optional
            If True, then retrieve neighbors original vector and rerank accordingly.
            Default is True
        threshold : float, optional
            Only keep neighbors if similarity exceeds threshold. Default is None which
            keeps all neighbors returned.
        search_parameters : faiss.SearchParameters, optional
            Specify specific search parameters for this query. Each FAISS index has its
            own class. For example, faiss.SearchParametersIVF(nprobe=20) will set the
            nprobe value for an IVF index. This can also be nested. For example, for an
            IVF_HNSW use
            faiss.SearchParametersIVF(
                nprobe=20,
                quantizer_params=faiss.SearchParamsHNSW(efSearch=40),
            )

        Returns
        -------
        List[List[Dict]]
            For each query, return a list of the neighbors. For each neighbor of a
            query, return a dictionary of the neighbor's record with "metric" add to
            it that holds the similarity between query and vector.
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
        results = []
        query_vectors = []
        tmp_col_names = tuple(list(col_names) + [self.idx_name, self.vector_name])
        for record in self.fetch_records(
            fetch_column,
            fetch_values,
            *tmp_col_names,
            batch_size=batch_size,
        ):
            results.append(record)
            query_vectors.append(record[self.vector_name])

        query_vectors = np.vstack(query_vectors)
        tmp_col_names_neighbors = tuple(list(col_names) + [self.idx_name])
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
        """_summary_

        Parameters
        ----------
        fetch_column : str
            Column name to fetch by
        fetch_values : List
            Values of fetch_column to fetch
        \*col_names : str, optional
            Specify which columns to be fetched. If not provided, then all columns will
            be fetched.
        batch_size : int, optional
            Number of values to fetch at one time. Default 10_000

        Yields
        ------
        Iterator[Dict]
            Keys are the elements of columns and values are corresponding values
        """

        if (
            self.columns[fetch_column].index is None
            and not self.columns[fetch_column].primary_key
        ):
            logging.warning(
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
