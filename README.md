VekterDB
========

Transform any SQLAlchemy compliant database into a vector database by adding **any** type of a FAISS index in order to perform approximate nearest neighbor (ANN) search on the vector column.

Most vector databases available resort to using the Hierarchical Navigable Small World (HNSW) algorithm for ANN. While HNSW is a highly performant ANN, it struggles to scale beyond ~10M vectors [1]. In this reference they report building an HNSW index on 2.7M 256-dimensional vectors takes 2hrs 20mins and memory resources on a single machine get exceeded as well.

While many data

A cheap imitation for a vector database that has none of the guarantees, but it is highly flexibly and thus scalable. Most vector database resort to using on HNSW for the vector indexing, but in vekterdb you can utilize any FAISS index type you want.

[1] I. Doshi, D. Da, A. Bhutani, R. Kumar, R. Bhatt, N. Balasubramanian, *LANNS: a web-scale approximate nearest neighbor lookup system*, Proceedings of the VLDB Endowment **15(4)**, 850 (2021). See also arXiv:2010.09426