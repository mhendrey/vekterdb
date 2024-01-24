import argparse
import h5py
import numpy as np
import sqlalchemy as sa


try:
    from ..vekterdb.vekterdb import VekterDB
except ImportError:
    from vekterdb.vekterdb import VekterDB


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sift_1M_file", type=str, help="HDF5 file containing the SIFT-1M data"
    )

    return parser.parse_args()


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


def main():
    args = parse_cmd_line()

    vekter_db = VekterDB(
        "tutorial",
        idx_name="idx",
        vector_name="vector",
        columns_dict={
            "id": {
                "type": sa.types.Text,
                "unique": True,
                "nullable": False,
                "index": True,
            },
        },
        url="sqlite:///sift1m.db",
    )

    train_records = records_gen(args.sift_1M_file)
    n_records = vekter_db.insert(train_records, batch_size=50_000)

    vekter_db.create_index(
        "ivf_hnsw.index",
        "IVF5000_HNSW32,PQ32",
        metric="L2",
        batch_size=50_000,
        faiss_runtime_params="quantizer_efSearch=25",
    )

    test_data = list(records_gen(args.sift_1M_file, test_data=True))

    f = h5py.File(args.sift_1M_File, "r")
    true_neighbors = f["neighbors"]
    true_distances = f["distances"]

    neighbors = vekter_db.search(test_data[0]["vector"].reshape(1, -1), 5, "idx")

    if true_neighbors[0][0] == neighbors[0][0]["idx"]:
        print("You got lucky")
    else:
        print(
            "Don't worry. You forgot to increase search parameters from their defaults"
        )
    # Let's see just the first neighbor
    print(neighbors[0][0])

    # True nearest neighbor is
    print(true_neighbors[0][0])
    print(true_distances[0][0])

    vekter_db.set_faiss_runtime_parameters("nprobe=175,quantizer_efSearch=350")
    neighbors = vekter_db.search(test_data[0]["vector"].reshape(1, -1), 5, "idx")
    if true_neighbors[0][0] == neighbors[0][0]["idx"]:
        print("We found the true nearest neighbor!")
    else:
        print(
            "Yikes! something still went wrong. Some things to try\n"
            + "Increase the k_extra_neighbors from 0 to say 20."
            + " This pulls back some additional records and then reranks by true L2."
            + "\nOr increase nprobe some more"
        )

    # Query with all test vectors
    q_vecs = np.vstack([t["vector"] for t in test_data])

    # Do recall@1 metric.  Not great 0.6646
    neighbors = vekter_db.search(q_vecs, 1, "idx")
    found_nearest = 0
    for i in range(len(neighbors)):
        if neighbors[i][0]["idx"] == true_neighbors[i][0]:
            found_nearest += 1
    print(f"{found_nearest / len(neighbors):.04f}")

    # Still doing recall@1, but first getting back 5 neighbors, reranking by true
    # distance and then only returning the top result greatly improves performance
    # 0.6646 -> 0.9450
    neighbors = vekter_db.search(q_vecs, 1, "idx", k_extra_neighbors=4)
    found_nearest = 0
    for i in range(len(neighbors)):
        if neighbors[i][0]["idx"] == true_neighbors[i][0]:
            found_nearest += 1
    print(f"{found_nearest / len(neighbors):.04f}")

    # Add test data into database table & FAISS index
    n_records = vekter_db.insert(test_data, batch_size=50_000)

    neighbors = vekter_db.nearest_neighbors(
        "idx", [1_000_000], 5, "idx", k_extra_neighbors=19
    )
    if true_neighbors[0][0] == neighbors[0]["neighbors"][0]["idx"]:
        print("We found the true nearest neighbor!")
    else:
        print(
            "Yikes! something still went wrong. Some things to try\n"
            + "Increase the k_extra_neighbors from 0 to say 20."
            + " This pulls back some additional records and then reranks by true L2."
            + "\nOr increase nprobe some more"
        )

    # recall@1 test
    neighbors = vekter_db.nearest_neighbors(
        "id",
        [str(i) for i in range(1_000_000, 1_010_000)],
        1,
        "idx",
        k_extra_neighbors=4,
    )

    found_nearest = 0
    for i in range(len(neighbors)):
        if neighbors[i]["neighbors"][0]["idx"] == true_neighbors[i][0]:
            found_nearest += 1
    print(f"{found_nearest / len(neighbors):.04f}")

    # Save / Load
    vekter_db.save()
    vekter_db = VekterDB.load("tutorial.json", url="sqlite:///sift1m.db")

    # recall@1 test
    neighbors = vekter_db.nearest_neighbors(
        "id",
        [str(i) for i in range(1_000_000, 1_010_000)],
        1,
        "idx",
        k_extra_neighbors=4,
    )

    found_nearest = 0
    for i in range(len(neighbors)):
        if neighbors[i]["neighbors"][0]["idx"] == true_neighbors[i][0]:
            found_nearest += 1
    print(f"{found_nearest / len(neighbors):.04f}")


if __name__ == "__main__":
    main()
