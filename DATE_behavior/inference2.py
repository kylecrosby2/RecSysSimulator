import torch
import torch.nn as nn
from modules import fet_mac, cap_mac

# Configuration Parameters
Nu = 10**7  # Number of users
Ni = 10**6  # Number of items
d = 64      # Embedding dimension
qbit = 32   # Quantization bit-width (e.g., 32-bit floats)
sa_rows, sa_cols = 128, 128  # Size of the systolic array
candidate_items = 100  # Number of candidate items

# Mock latency and energy parameters
fet_mac_lat, fet_mac_en = 5, 0.2
cap_mac_lat, cap_mac_en = 10, 0.1

# Define Embedding Tables
user_embedding_table = nn.Embedding(Nu, d)
item_embedding_table = nn.Embedding(Ni, d)

# Function to Simulate the Workload
def run_workload(lat_func, en_func, is_capacitor=False):
    """
    Simulate the workload using specified multiplication functions for latency and energy modeling.
    """
    lat, en = 0, 0  # Initialize latency and energy

    # Step 1: Fetch User Embedding
    user_id = torch.tensor([1234567])  # Example user ID
    user_embedding = user_embedding_table(user_id)

    # Step 2: Fetch Item Embeddings
    item_ids = torch.arange(Ni)
    item_embeddings = item_embedding_table(item_ids)

    # Step 3: Compute Nearest Neighbor Search (NNS)
    if is_capacitor:
        # cap_mac takes both x (user_embedding) and y (item_embeddings)
        lat, en = cap_mac(
            user_embedding.unsqueeze(0).unsqueeze(0),  # Reshape for cap_mac
            item_embeddings.unsqueeze(0).unsqueeze(0),  # Reshape for cap_mac
            qbit,
            sa_rows,
            sa_cols,
            cap_mac_lat,
            cap_mac_en,
            lat,
            en,
        )
    else:
        # fet_mac takes only x (user_embedding)
        lat, en = fet_mac(
            user_embedding.unsqueeze(0),  # Reshape for fet_mac
            qbit,
            sa_rows,
            sa_cols,
            fet_mac_lat,
            fet_mac_en,
            lat,
            en,
        )

    # Step 4: Select Candidate Embeddings
    candidate_indices = torch.topk(
        (user_embedding @ item_embeddings.T).squeeze(0), k=candidate_items, dim=0
    ).indices
    candidate_embeddings = item_embeddings[candidate_indices]

    # Memory for Candidate Embeddings
    candidate_memory = candidate_items * d * (qbit // 8)  # In bytes
    print(f"Memory for candidate embeddings: {candidate_memory / 1024:.2f} KB")

    return lat, en

# Run workload with FET multiplication
print("Running workload with FET multiplication...")
fet_lat, fet_en = run_workload(fet_mac, fet_mac_en, is_capacitor=False)

# Run workload with Capacitor multiplication
print("Running workload with Capacitor multiplication...")
cap_lat, cap_en = run_workload(cap_mac, cap_mac_en, is_capacitor=True)

# Output Comparison
print("\n=== Comparison ===")
print(f"FET Latency: {fet_lat / 1e3:.2f} ms, Energy: {fet_en / 1e6:.2f} mJ")
print(f"Capacitor Latency: {cap_lat / 1e3:.2f} ms, Energy: {cap_en / 1e6:.2f} mJ")

lat_diff = ((fet_lat - cap_lat) / max(fet_lat, cap_lat)) * 100
en_diff = ((fet_en - cap_en) / max(fet_en, cap_en)) * 100

print(f"Latency Difference: {lat_diff:.2f}%")
print(f"Energy Difference: {en_diff:.2f}%")