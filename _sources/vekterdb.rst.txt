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

The sift-128-euclidean.hdf5 file has the following keys:

    * train : 1M 128-dimensional numpy vectors
    * test : 10,000 128-dimensional numpy vectors
    * neighbors : 10,000 x 100 that list the idx's of top 100 neighbors of the
                  corresponding test vector
    * distances : 10,000 x 100 that list the Euclidean distance between test & its
                  neighbors

Set up
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We will begin by setting up a new conda environment and installing needed packages.
We need to install `FAISS
<https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda>`_
from conda and FAISS currently supports up to python 3.10 (see link if you want to use
your gpu).  We will also install h5py to be able to read the downloaded SIFT-1M
dataset.::

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
            "id": {"type": sa.types.Text, "unique": True, "nullable": False, "index": True},
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

    def records_gen(h5_file: str, test_data: bool = False):
        with h5py.File(h5_file, "r") as f:
            if test_data:
                vectors = f["test"]
            else:
                vectors = f["train"]

            for i, vector in enumerate(vectors):
                if test_data:
                    i += 1_000_000
                yield {"id": str(i), "idx": i, "vector": vector}


    train_records = records_gen("sift-128-euclidean.hdf5")
    n_records = vekter_db.insert(train_records)

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
all of the training data, 1M, to train the index. We pull 50,000 records from the
database at any one time and also insert into FAISS at this amount. When adding vectors
into the FAISS index, we will select the closest centroid from amongst a candidate pool
of the nearest 25 centroids. If we had used just an "IVF5000,PQ32" index, we would
compare each vector to all 5,000 centroids to determine which partition to insert
the vector.

::

    vekter_db.create_index(
        "ivf_hnsw.index",
        "IVF5000_HNSW32,PQ32",
        metric="L2",
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

So far, we have only added the training data to the database and then used all those
vectors to train the FAISS index. We begin by getting reading the testing data from the
file and then using those to query our vector database for similar records.  We will
compare those to the "neighbors" stored in the HDF5 file. Let's start slow and just
use the first test vector and get the five nearest neighbors. Since the FAISS
``index.search()`` requires that the vector shape be at least (n, d), we reshape the
first vector to match. To easily compare, we only need to see the "idx" column.

The ``search()`` returns a list with one element for each of the query vectors. Each
element of the list is a dictionary with a single key, "neighbors", whose value is a
list of dicts that are the neighbors for that query vectory. The dict's keys are the
columns in the database table (or only those you specified with any ``*col_names``)
and an additional ``"metric"`` which holds the similarity between that neighbor and
the query vector.

::

    test_data = list(records_gen("sift-128-euclidean.hdf5", test_data=True))
    
    f = h5py.File("sift-128-euclidean.hdf5", "r")
    true_neighbors = f["neighbors"]
    true_distances = f["distances"]

    neighbors = vekter_db.search(test_data[0]["vector"], 5, "idx")[0]["neighbors"]
    # Let's see just the first neighbor
    print(neighbors[0])          # {'idx': 695756, 'metric': 258.86288}

    # True nearest neighbor is
    print(true_neighbors[0][0])  # 932085
    print(true_distances[0][0])  # 232.87122

You will likely notice that the true nearest neighbor, idx=932085 with distance=232.87,
does NOT match the nearest record returned by our ``search()`` method.  For me, I get
idx=695756 with metric=258.86.

But don't lose hope, because the search we did was using the default FAISS runtime
search parameters. For the "IVF_HNSW,PQ" index, this means we are using an ``nprobe=1``
and ``quantizer_efSearch=16``. With only searching the nearest partition choosen from
a candidate pool of 16, it is not surprising that our search failed to return the
expected neighbor. Typically, setting ``nprobe`` to 2 - 5% of the number of partitions
(5,000 for this index) yields acceptable results. We will use ``nprobe=175`` (3.5%). If
``nprobe=175``, then we should also increase ``quantizer_efSearch`` too so that the
candidate pool is bigger than ``nprobe``. We will use ``quantizer_efSearch=350`` and
then retry our test query.

::

    vekter_db.set_faiss_runtime_parameters("nprobe=175,quantizer_efSearch=350")

    neighbors = vekter_db.search(test_data[0]["vector"], 5, "idx")[0]["neighbors"]
    # Let's see just the first neighbor
    print(neighbors[0])          # {'idx': 932085, 'metric': 232.87122}

    # True nearest neighbor is
    print(true_neighbors[0][0])  # 932085
    print(true_distances[0][0])  # 232.87122

Now that is more like it! If you still don't get the right answer, there are a few
additional things you can try. If we had been more aggressive with the PQ, say as low
as PQ8, then it's possible that the estimated distances that FAISS uses to find the
nearest records is not great. In this case, you can increase the ``k_extra_neighbors``
from 0 (default) to say 30. This will return 5 + 30 nearest vectors and then they will
be reranked based upon the true distance between the query vector and its neighbors
with only the top 5 after reranking returned to you. If that doesn't work, you can
experiment with increase ``nprobe`` and/or ``quantizer_efSearch`` some more. But be
warned! Increase these values necessary slows down the querying time since more of
the data is being checked.

With a simple test done, let's query with all the test data and see how we do with the
recall@1 metric. That is, do we find the true nearest neighbor in our top 1 nearest
neighbor returned from the search?

::

    q_vecs = np.vstack([t["vector"] for t in test_data])

    search_results = vekter_db.search(q_vecs, 1, "idx")
    found_nearest = 0
    for i, search_result in enumerate(search_results):
        if true_neighbors[i][0] == search_result["neighbors"][0]["idx"]:
            found_nearest += 1
    print(f"Recall@1 = {found_nearest / len(search_results):.04f}")

In my running, I get a value of recall@1 = 0.6646. This is ok, but maybe a little
disheartening. However, we only allowed FAISS to return the nearest approximate
neighbor and we have quite a few parameters to tweak if needed as we mentioned above.
We will start with raising the ``k_extra_neighbors`` value from 0 (default) to 4.

::

    search_results = vekter_db.search(q_vecs, 1, "idx", k_extra_neighbors=4)
    found_nearest = 0
    for i, search_result in enumerate(search_results):
        if true_neighbors[i][0] == search_result["neighbors"][0]["idx"]:
            found_nearest += 1
    print(f"Recall@1 = {found_nearest / len(search_results):.04f}")

Much better. I got 0.9450. In fact, if we increase even further ``k_extra_neighbors=49``,
then our result goes up to 0.9902. Again, our query time is increasing so it is always
a trade off. On my machine the query time for ``q_vecs`` went from 0.996s
(``k_extra_neighbors=0``) to 1.25s (4) to 3.47s (49).

Querying for Similar Records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this section, we will show how to query for the nearest records of a given record
that is already in the database table. We begin by adding in the test vectors to our
database table and FAISS index. We will then repeat our simple test, but this time
using the ``nearest_neighbors()`` method.

This method allows you to select records whose nearest neighbors you are querying for
by specifying which column of the database to use ``fetch_column`` and then a list of
values whose records you want to use for querying, ``fetch_values``. The rest follows
the same as ``search()`` since ``nearest_neighbors`` is just combining a
``fetch_records()`` to retrieve the vectors of the request records and then using
``search()``. Caution is taken to remove the query record from the neighbors list.

The ``nearest_neighbors()`` returns a list of dictionaries with each dictionary list
the query record's values; either all the column values or just the values of the
columns names specified with ``*col_names``. In addition, the "neighbors" key contains
the list of neighbors and the metric between the neighbor and the query record. This
list is sorted in appropriate order with nearest neighbor listed first.

::

    n_records = vekter_db.insert(
        test_data, batch_size=50_000, faiss_runtime_params="quantizer_efSearch=25"
    )

    neighbors = vekter_db.nearest_neighbors("idx", [1_000_000], 5, "idx")[0][
        "neighbors"
    ]

    if true_neighbors[0][0] == neighbors[0]["idx"]:
        print("We found the true nearest neighbor!")
        print(f"       Found {neighbors[0]}")
        ground_truth = {"idx": true_neighbors[0][0], "metric": true_distances[0][0]}
        print(f"Ground truth {ground_truth}")
    else:
        print(
            "Yikes! something still went wrong. Some things to try\n"
            + "Increase the k_extra_neighbors from 0 to say 20."
            + " This pulls back some additional records and then reranks by true L2."
            + "\nOr increase nprobe some more"
        )

Having passed the simple test, let's do the full test data.  To mix things up, let's
show that we can also use the ``id`` column to specify the records to fetch. Because
we had this column indexed in the specification of the database table, this should
equivalant as using the primary key, ``idx``, of the database table.

::

    nn_results = vekter_db.nearest_neighbors(
        "id",
        [str(i) for i in range(1_000_000, 1_010_000)],
        1,
        "idx",
        k_extra_neighbors=4,
    )
    found_nearest = 0
    for i, nn_result in enumerate(nn_results):
        if true_neighbors[i][0] == nn_result["neighbors"][0]["idx"]:
            found_nearest += 1
    print(f"{found_nearest / len(nn_results):.04f}")

We get a recall@1 = 0.9303 with these runtime search parameters and
``k_extra_neighbors`` of 4. Notice that this value is a little lower than we had
before, 0.9450. This is caused by the additional 10,000 test vectors added into
the database table with a small percentage of them being the nearest neighbor for
other records in the test data.

Saving & Loading a VekterDB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Having created a database table, populated it with records, created a FAISS index
(including both training it and adding in vectors from the database table), we are now
ready to save the VekterDB so that it can be loaded back up.

Since the records are already stored in the database, there is nothing to do there.
In fact, the FAISS index has also been saved off already as well. First right at the
end of ``create_index()`` and then again after we called ``insert()`` to add in the
test data. This leaves just saving off some of the metadata needed to reinitialize
``VekterDB`` which is done using the ``save()``.  This writes a JSON file to disk of
the needed information. If you don't provide a ``config_file`` name, ``save`` will
write the file {table_name}.json to the local directory. In particular, this saves
off any default FAISS runtime search parameters that you may have set with
``set_faiss_runtime_parameters()``.

::

    vekter_db.save()
    del vekter_db



To test this, let's exit out of the Python interpretter/IPython/Jupyter notebook and
import the needed libraries, load our ``VekterDB`` from disk, and then run a test
query.  Notice, that you do need to provide the connection URL to the database when
calling ``load()``. This is done for security reasons since any username/password may
be needed to pass in that URL string that we don't want to save to disk in plaintext.

::

    vekter_db = VekterDB.load("tutorial.json", url="sqlite:///sift1m.db")

    nn_results = vekter_db.nearest_neighbors(
        "id",
        [str(i) for i in range(1_000_000, 1_010_000)],
        1,
        "idx",
        k_extra_neighbors=4,
    )
    found_nearest = 0
    for i, nn_result in enumerate(nn_results):
        if true_neighbors[i][0] == nn_result["neighbors"][0]["idx"]:
            found_nearest += 1
    print(f"{found_nearest / len(nn_results):.04f}")

And we get back the same 0.9303 for the recall@1.

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
