import faiss

# Your existing code to read the Faiss index
faiss_index_path = "C:\\office\\chatbots\\index_store\\faiss_index.index"
index = faiss.read_index(faiss_index_path)

# Access the underlying index
underlying_index = index.index

# Check if the underlying index has the 'reconstruct' method
if hasattr(underlying_index, 'reconstruct'):
    # Get the total number of vectors in the index
    total_vectors = underlying_index.ntotal
    print(f"Total vectors in the index: {total_vectors}")

    # Access a single vector (for example, the first one)
    vector_index = 0
    reconstructed_vector = underlying_index.reconstruct(vector_index)

    # Print the class name of the reconstructed vector
    classname = reconstructed_vector.__class__.__name__
    print(f"Class name of the reconstructed vector: {classname}")

    # Get the associated page number using the id_map
    page_number = index.id_map.at(vector_index)
    print(f"Page number associated with the vector: {page_number}")

    # Print the example vector directly
    print("Example vector: ", reconstructed_vector)
else:
    print("The 'reconstruct' method is not available for this type of index.")

