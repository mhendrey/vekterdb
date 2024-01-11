VekterDB
========

Transform any [SQLAlchemy](https://www.sqlalchemy.org/) compliant database into a vector database by adding ***any*** type of a [FAISS](https://ai.meta.com/tools/faiss/) index in order to perform approximate nearest neighbor (ANN) search on the vector column.

Almost all available vector databases only allow users to select a limited subset of the ANN algorithms available in libraries like FAISS. The typical indices that are permitted are Hierarchical Navigable Small-World (HNSW) or Inverted File System (IVF), but both of these struggle to scale effectively beyond 10-100 million vectors ([1], [2], and [3]). The aim of VekterDB is to provide users with greater flexibility in terms of both the ANN index and the type of database to use.

Installation
============
VekterDB may be install using pip

    pip install vekterdb


Documentation
=============
Documentation can be found at https://mhendrey.github.io/vekterdb


[1] I. Doshi, D. Da, A. Bhutani, R. Kumar, R. Bhatt, N. Balasubramanian, *LANNS: a web-scale approximate nearest neighbor lookup system*, Proceedings of the VLDB Endowment **15(4)**, 850 (2021). See also arXiv:2010.09426

[2] C. Fu, C. Xiang, C. Wang, and D. Cai. *Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph*, arXiv:1707.00143 (2017).

[3] B. Riggs and G. Williams, *ANN Benchmarks: A Data Scientist's Journey to Billion Scale Performance*, https://medium.com/gsi-technology/ann-benchmarks-a-data-scientists-journey-to-billion-scale-performance-db191f043a27 (Note: they actually only tested on 54M vectors)