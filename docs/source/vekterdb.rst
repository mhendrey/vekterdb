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
We will begin by setting up a new conda environment and installing needed packages.::

    $ conda create -n vekterdb_tutorial python=3.11 ipython
    $ conda activate vekterdb_tutorial
    $ conda install -c pytorch -c conda-forge -c defaults faiss h5py
    $ pip install vekterdb





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
