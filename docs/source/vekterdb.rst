VekterDB
========

The Basics
----------
Transform any `SQLAlchemy <https://www.sqlalchemy.org/>`_ compliant database into a
vector database by adding **any** type of a `FAISS <https://ai.meta.com/tools/faiss/>`_
index to index the vectors in order to utilize the many different types of approximate
nearest neighbor (ANN) algorithms available in FAISS.

Almost all available vector databases only allow users to select a limited subset of
the ANN algorithms available in libraries like FAISS. The typical indices that are
permitted are Hierarchical Navigable Small-World (HNSW) or Inverted File System (IVF),
but both of these struggle to scale effectively beyond 10-100 million
vectors [#f1]_ [#f2]_ [#f3]_. The aim of VekterDB is to provide users with greater
flexibility in terms of both the ANN index and the type of databases they can use.

VekterDB requires a minimum of two columns in the database table. The first is an
integer based identification [0, N) that is used by the FAISS index to refer to records
and serves as the primary key for the table. The second required column stores the
vector as bytes. Since VekterDB leverages FAISS, these vectors must be numpy arrays of
type float32. Of course, additional columns may be specified. For example, you may have
another column that serves as a more recognizable ID field that is a string.

Usage
-----

::

    import numpy as np
    from vekterdb import VekterDB

    # Let's use some random data
    def generate_data(N:int=10_000, d:int=64, seed:int=1024):
        rng = np.random.default_rng(seed=seed)
        vectors = rng.normal(size=(N, d)).astype(np.float32)
        # Make vectors[1] be similar to vectors[0]
        vectors[1] = vectors[0] + rng.normal(scale=0.2, size=d).astype(np.float32)
        for idx, vector in enumerate(vectors):
            yield {"idx": idx, "vector": vector}
    
    records = generate_data()

    # Create new table "my_table" in an in-memory SQLite database
    vekter_db = VekterDB("my_table")
    # Insert vectors into the database
    n_records = vekter_db.insert(records)

    # Create a Flat FAISS index and save it in the file "my_table.index"
    vekter_db.create_index("my_table.index", faiss_factory_str="Flat")

    # Let's find the 3 nearest neighbors of idx=0 and idx=1
    # These should be each other
    results = vekter_db.nearest_neighbors("idx", [0, 1], 3, "idx")

    """
    Outputs the following
    [
        {'idx': 0,
         'neighbors': [
            {'idx': 1, 'metric': 73.78341},
            {'idx': 7537, 'metric': 39.142662},
            {'idx': 9831, 'metric': 31.672077}
         ],
        },
        {'idx': 1,
         'neighbors': [
            {'idx': 0, 'metric': 73.78341},
            {'idx': 7537, 'metric': 38.97898},
            {'idx': 9831, 'metric': 32.063244}
         ],
        },
    ]
    """

Tutorial
----------------------------------------------------------------------------------------
In this tutorial, we will work through an example using one of the datasets found in the
`Approximate Nearest Neighbors Benchmark
<https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file#data-sets>`_, SIFT-1M.
The data is stored in an HDF5 file and can be downloaded from this `link
<http://ann-benchmarks.com/sift-128-euclidean.hdf5>`_ (~500MB).

Set up
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We will begin by setting up a new conda environment and installing needed packages.
We need to install `FAISS
<https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda>`_
from conda and FAISS currently supports up to python 3.10.  We will also install h5py to
be able to read the downloaded SIFT-1M dataset.::

    $ conda create -n vekterdb_tutorial python=3.10 ipython
    $ conda activate vekterdb_tutorial
    $ conda install -c pytorch -c conda-forge -c defaults h5py faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
    $ pip install vekterdb
    $ ipython

The rest of the code assumes you are inside a Python interpretter (ipython, Jupyter,
or whatever you like).

Initialize VekterDB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We begin by initializing a VekterDB. For this tutorial, we will use a SQLite database,
but you could just as easily use a Postgres, MySQL, or any other of the SQLAlchemy
`dialects <https://docs.sqlalchemy.org/en/20/dialects/index.html>`_. The SQLite
database will be stored in local file "sift1m.db" in a table called "tutorial".
Though the h5py file has just the two required fields for VekterDB, namely an integer
identifier and the vector, we will add an additional string identifier for
demonstration purposes. This column will be indexed by SQLite in order to allow a
user to query for nearest neighbors using this identifier as well.

::

    import h5py
    import numpy as np
    import sqlalchemy as sa
    from vekterdb import VekterDB

    vekter_db = VekterDB(
        "tutorial",
        idx_name = "idx",
        vector_name = "vector",
        columns_dict = {
            "id": {"type": sa.Text, "unique": True, "nullable": False, "index": True},
        },
        url = "sqlite:///sift1m.db"
    )

Insert Records into the DB Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With the database table, tutorial, now created in the database, sift1m.db, it is time
to add the records from the HDF5 file.  We will specify a function that will yield the
records that will be inserted into the database table. Since we haven't specified a
FAISS index yet, these records will only be added to the database table.

The ``records_gen`` is a good candidate for parallelization.

::

    def records_gen(h5_file:str):
        with h5py.File(h5_file, 'r') as f:
            for i in range(n):
                yield {"id": str(i), "idx": i, "vector": f["data"][i]}

    records = records_gen("sift-128-euclidean.hdf5")
    n_records = vekter_db.insert(records)

Create FAISS Index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With the database table now populated, we can construct the desired FAISS index which
handles the vector similarity queries. In this tutorial we will utilize a more
complicated index than is probably necessary, but we want to demonstrate using an index
appropriate for way more than 1M vectors.

For scalability, we will use an "IVF_HNSW,PQ" index.  Specifically, let's use a
"IVF5000_HNSW32,PQ32" index. This splits the 128-dimensional space into
5 * sqrt(1_000_000) = 5,000 partitions. The centroids of the 5,000 partitions will
themselves be indexed using an HNSW32. To help save space, we will also use a Product
Quantization to shrink the size of each vector into ~ 32 bytes, down from 512 bytes.

The FAISS index will be saved to local disk in the "ivf_hnsw.index" file. The metric
is set to "L2" to match the Euclidean distance used for the SIFT-1M dataset. We use
half of the data, 500k, to train the index. We pull 50,000 records from the database
at any one time and also insert into FAISS at this amount. When adding vectors into
the FAISS index, we will select the closest centroid from amongst a candidate pool
of the nearest 25 centroids. If we had used just an "IVF5000,PQ32" index, we would
compare each vector to all 5,000 centroids to determine which partition to insert
the vector.

::

    vekter_db.create_index(
        "ivf_hnsw.index",
        "IVF5000_HNSW32,PQ32",
        metric="L2",
        sample_size=500_000,
        batch_size=50_000,
        faiss_runtime_params="quantizer_efSearch=25",
    )

Querying for Similar Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
VekterDB offers two ways in which to query for nearest neighbors. The first handles the
cases when you have a vector that is **not** part of the database table, but want to
find the records in the database table that are the most similar to the query. This is
done with the ``search()`` method.  The second is when you want to find the nearest
neighbors in the database table to an existing record in the table. This is done with
the ``nearest_neighbors()`` method.

Before we begin searching, it's likely that you want to change from the default FAISS
runtime search parameters. For our "IVF_HNSW,PQ" index, the default values are
``nprobe=1`` and ``quantizer_efSearch=16``. This means that only the nearest partition,
out of a candidate pool of 16, will be searched for the nearest to a given query. This
is likely too little of the search space. Typically, setting ``nprobe`` to ~ 2-5% of
the number of partitions (5,000 for this index) yields acceptable results. So let's use
``nprobe=175`` (3.5%). If ``nprobe=175``, then we want to increase ``quantizer_efSearch``
too, so that the candidate pool is bigger than ``nprobe``. Let's use
``quantizer_efSearch=350``.

Let's begin with the ``search()`` approach. This returns a list where each element is
a list of the nearest records for the corresponding query vector. For the sake of
demonstration, we will pull some query vectors from the database, but this could more
easily be done with the ``nearest_neighbors()``, as we will show in a minute. Let's
pull idx = 100 & idx = 200, but since we only need the vector we only request that
column be returned from the ``fetch_records()``.

We will search for the nearest 5 vectors.

::

    fetched_records = vekter_db.fetch_records("idx", [100, 200], "vector")
    query_vectors = np.vstack([r["vector"] for r in fetched_records])

    results = vekter_db.search(query_vectors, 5, "idx", "id", k_extra_neighbors=30)

.. rubric:: Footnotes

.. [#f1] I. Doshi, D. Da, A. Bhutani, R. Kumar, R. Bhatt, N. Balasubramanian,
         *LANNS: a web-scale approximate nearest neighbor lookup system*,
         Proceedings of the VLDB Endowment **15(4)**, 850 (2021).
         See also `arXiv:2010.09426 <https://arxiv.org/abs/2010.09426>`_
.. [#f2] C. Fu, C. Xiang, C. Wang, and D. Cai.
         *Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph*,
         `arXiv:1707.00143 <https://arxiv.org/abs/1707.00143>`_  (2017).
.. [#f3] B. Riggs and G. Williams,
         `ANN Benchmarks: A Data Scientist's Journey to Billion Scale Performance <https://medium.com/gsi-technology/ann-benchmarks-a-data-scientists-journey-to-billion-scale-performance-db191f043a27>`_ 
         (Note: they actually only tested on 54M vectors)
