from sqlalchemy import Column, Table, create_engine, insert
from sqlalchemy.types import *
from sqlalchemy.orm import DeclarativeBase, Session
from typing import Any, Dict, List


class VekterDB:
    """Instantiate a VekterDB"""

    class Base(DeclarativeBase):
        pass

    def __init__(
        self,
        table_name: str,
        idx_name: str,
        columns_dict: Dict[str, Dict] = {},
        url: str = "sqlite://",
        connect_args: Dict = {},
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
        idx_name : str,
            Column name of the integer ID that is unique for each record. The lowest
            value must be zero and this must be consecutive values for each record in
            the table. This is what is used by the FAISS index.
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
        """
        columns = [Column(idx_name, BigInteger, primary_key=True)]
        for column_name, column_args in columns_dict.items():
            column_type = column_args.pop("type")
            columns.append(Column(column_name, column_type, **column_args))
        # Plan is to serialize each np.ndarray(dtype=np.float32) using .tobytes()
        # And then back using np.frombuffer(bytes_from_db)
        columns.append(Column("vector", LargeBinary, nullable=False))

        # Declare table mapping
        class RecordMapping(self.Base):
            __table__ = Table(
                table_name,
                self.Base.metadata,
                *columns,
            )

        self.Record = RecordMapping
        self.engine = create_engine(url, **connect_args)
        self.Base.metadata.create_all(self.engine)

    def insert(self, records: List[Dict]):
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
        records : List[Dict]
            Each element of the list should be a dictionary with a key the column name
            and the the value the corresponding value.
        """
        with Session(self.engine) as session:
            # Convert the vector to bytes for storage in the database
            for record in records:
                record["vector"] = record["vector"].tobytes()
            session.execute(insert(self.Record), records)
            session.commit()
