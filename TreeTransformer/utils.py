import pickle
import torch
from .custom_sequence_parser import CustomSequenceParser
from tqdm import tqdm
from networkx import Graph

def get_input_vectors(graph_path: str, MAXLEN: int = 50, EMBEDDING_DIM: int = 87 , device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print("Initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Loading the Graph from pickle file...")
    with open(graph_path, 'rb') as f:
        Graph = pickle.load(f)

    print("Identifying nodes with NaN attributes to delete...")
    nodes_to_delete = []
    for node, data in Graph.nodes(data=True):
        tensor = data["attr_dict"]
        if torch.isnan(tensor).any():
            nodes_to_delete.append(node)
    for node in nodes_to_delete:
        Graph.remove_node(node)

    print("Extracting sequences from the Graph...")
    sequences = []
    error = []

    def extract_sequence(Graph: Graph, start_node: int):
        print(f"    Extracting sequences starting from node {start_node}...")
        sequence = []
        def dfs(node, path):
            if path and node == path[-1]:
                return
            new_path = path + [node]
            if Graph.out_degree(node) == 0:
                sequence.append(new_path)
                return
            for _, child_node in Graph.out_edges(node):
                dfs(child_node, new_path)
        dfs(start_node, [])
        return sequence

    for node in Graph.nodes():
        try:
            sequence = extract_sequence(Graph, node)
            sequences.extend(sequence)
        except Exception as e:
            print("Error while creating sequences in node", node)
            error.append(node)
            print(e)
            pass

    print("Preparing input vectors from sequences...")
    def prepare_input_vectors(sequences, Graph):
        vector_sequences = []
        for sequence in sequences:
            sequenceOfVectors = []
            for node in sequence:
                vector = Graph.nodes[node]['attr_dict'].to(device)
                sequenceOfVectors.append(vector)
            sequenceOfVectors = torch.stack(sequenceOfVectors)
            vector_sequences.append(sequenceOfVectors.squeeze(1))
        return vector_sequences

    input_vectors = prepare_input_vectors(sequences, Graph)

    print("Sorting sequences by length...")
    input_vectors.sort(key=lambda x: len(x), reverse=True)

    print("Adding special tokens to sequences...")
    seqpar = CustomSequenceParser(MAXLEN, sequences,EMBEDDING_DIM , device)
    input_vectors_with_special_tokens = []
    for sequence in tqdm(input_vectors, total=len(input_vectors)):
        input_vectors_with_special_tokens.append(seqpar.add_special_tokens(sequence))

    print("Processing complete.")
    return input_vectors_with_special_tokens , seqpar , input_vectors
