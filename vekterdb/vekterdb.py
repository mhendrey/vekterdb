import faiss
import logging
import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
import sqlalchemy.sql.functions as sa_funcs
from typing import Dict, Iterable, List


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
        url: str = "sqlite://",
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
            URL string to connect to the database, by default "sqlite://" which is an
            in-memory database.
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

    def serialize_vector(self, vector: np.ndarray):
        return vector.tobytes()

    def deserialize_vector(self, vector_bytes: bytes):
        return np.frombuffer(vector_bytes, dtype=np.float32)

    def insert(
        self,
        records: Iterable[Dict],
        batch_size: int = 5000,
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
        n_records = 0
        with self.Session() as session:
            batch = []
            for i, record in enumerate(records):
                # Set the dimension of the vectors using the first record
                if i == 0:
                    if serialize_vectors:
                        if self.d is None:
                            self.d = record[self.vector_name].shape[0]
                    elif self.d is None:
                        self.d = self.deserialize_vector(
                            record[self.vector_name]
                        ).shape[0]
                if serialize_vectors:
                    record[self.vector_name] = self.serialize_vector(
                        record[self.vector_name]
                    )
                batch.append(record)
                n_records += 1
                if len(batch) == batch_size:
                    session.execute(sa.insert(self.Record), batch)
                    batch = []
            if len(batch) > 0:
                session.execute(sa.insert(self.Record), batch)
            session.commit()

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
                self.idx_name, sample_idxs, columns=["vector"], batch_size=batch_size
            )
        ):
            X[i] = record[self.vector_name]

        return X

    def similarity_metric(
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
            raise FileExistsError(f"Index has already been assigned to this table")
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
            batch = []
            stmt = sa.select(self.columns[self.vector_name]).order_by(
                self.columns[self.idx_name]
            )
            for vector_bytes in session.scalars(stmt):
                batch.append(self.deserialize_vector(vector_bytes))
                if len(batch) == batch_size:
                    X = np.vstack(batch)
                    self.index.add(X)
                    batch = []
            if batch:
                X = np.vstack(batch)
                self.index.add(X)

        # Save the index to disk
        faiss.write_index(self.index, faiss_index)

    def search_by_vectors(
        self,
        query_vectors: np.ndarray,
        k_nearest_neighbors: int = 10,
        k_extra_neighbors: int = 0,
        rerank: bool = True,
        threshold: float = None,
        search_parameters: faiss.SearchParameters = None,
    ) -> List[Dict]:
        """_summary_

        Parameters
        ----------
        query_vectors : np.ndarray
            Vector(s) to search with. Shape should be (n, d). This is the same as FAISS
        k_nearest_neighbors : int, optional
            Number of nearest neighbors to return. Default is 10
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
        """
        k = k_nearest_neighbors + k_extra_neighbors
        _, I = self.index.search(query_vectors, k, params=search_parameters)
        idx_neighbors = list(set(I.reshape(-1).tolist()))

        # Get records for all the neighbors
        neighbor_records = {}
        for record in self.fetch_records(self.idx_name, idx_neighbors):
            neighbor_records[record[self.idx_name]] = record

        # Now loop through the queries
        results = []
        for query_vec, I_row in zip(query_vectors, I):
            query_result = []
            for idx in I_row:
                similarity = self.similarity_metric(
                    query_vec, neighbor_records[idx][self.vector_name], threshold
                )
                if similarity is not None:
                    neighbor = neighbor_records[idx].copy()
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
            results.append(query_result)

        return results

    def fetch_records(
        self,
        fetch_column: str,
        fetch_values: List,
        columns: List[str] = None,
        batch_size: int = 10_000,
    ) -> Dict:
        """_summary_

        Parameters
        ----------
        fetch_column : str
            Column name to fetch by
        fetch_values : List
            Values of fetch_column to fetch
        columns : List[str], optional
            List of column names to be returned. Include fetch_column if you want that
            returned. Default is None which gets all columns.
        batch_size : int, optional
            Number of values to fetch at one time. Default 10_000

        Yields
        -------
        Dict
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
        if columns is None:
            columns = list(self.columns.keys())

        with self.Session() as session:
            for n in range(n_batches):
                begin = n * batch_size
                end = begin + batch_size
                stmt = sa.select(self.Record).where(
                    self.columns[fetch_column].in_(fetch_values[begin:end])
                )
                for row in session.scalars(stmt):
                    record = {}
                    for col_name in columns:
                        value = row.__dict__[col_name]
                        if col_name == self.vector_name:
                            value = self.deserialize_vector(value)
                        record[col_name] = value
                    yield record
