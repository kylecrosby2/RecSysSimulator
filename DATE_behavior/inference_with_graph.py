import torch
import torch.nn as nn
from modules import fet_mac, cap_mac
import matplotlib.pyplot as plt
from config2 import *

# For demonstration, reduce the candidate_items to 10 for a cleaner plot
candidate_items = 10

# Mock latency and energy parameters (increased energy for visibility)
fet_mac_lat, fet_mac_en = 5, 5000.0
cap_mac_lat, cap_mac_en = 10, 1000.0

# Define Embedding Tables
user_embedding_table = nn.Embedding(Nu, d)
item_embedding_table = nn.Embedding(Ni, d)

# Function to simulate workload
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
            user_embedding.unsqueeze(0).unsqueeze(0),  # (1,1,d)
            item_embeddings.unsqueeze(0).unsqueeze(0), # (1,Ni,d)
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
            user_embedding.unsqueeze(0),  # (1,1,d)
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
        (user_embedding @ item_embeddings.T).squeeze(0),
        k=candidate_items, dim=0
    ).indices
    candidate_embeddings = item_embeddings[candidate_indices]

    # Memory for Candidate Embeddings
    candidate_memory = candidate_items * d * (qbit // 8)  # In bytes
    print(f"Memory for candidate embeddings: {candidate_memory / 1024:.2f} KB")

    # Return additional info for the plot
    # Also return the similarity scores for the top candidates for plotting
    similarities = (user_embedding @ item_embeddings.T).squeeze(0)
    topk_values = similarities[candidate_indices]

    return lat, en, candidate_indices, topk_values, candidate_memory

print("Running workload with FET multiplication...")
fet_lat, fet_en, fet_candidate_indices, fet_topk_values, fet_candidate_memory = run_workload(fet_mac, fet_mac_en, is_capacitor=False)

print("Running workload with Capacitor multiplication...")
cap_lat, cap_en, cap_candidate_indices, cap_topk_values, cap_candidate_memory = run_workload(cap_mac, cap_mac_en, is_capacitor=True)

# Output Comparison
print("\n=== Comparison ===")
print(f"FET Latency: {fet_lat / 1e3:.2f} ms, Energy: {fet_en / 1e6:.2f} mJ")
print(f"Capacitor Latency: {cap_lat / 1e3:.2f} ms, Energy: {cap_en / 1e6:.2f} mJ")

lat_diff = ((fet_lat - cap_lat) / max(fet_lat, cap_lat)) * 100
en_diff = ((fet_en - cap_en) / max(fet_en, cap_en)) * 100

print(f"Latency Difference: {lat_diff:.2f}%")
print(f"Energy Difference: {en_diff:.2f}%")

# Visualization with matplotlib
methods = ['FET', 'Capacitor']
latencies = [fet_lat / 1e3, cap_lat / 1e3]  # ns to ms
energies = [fet_en / 1e6, cap_en / 1e6]     # nJ to mJ

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Latency Comparison
axes[0].bar(methods, latencies, color=['#1f77b4', '#ff7f0e'])
axes[0].set_title('Latency Comparison')
axes[0].set_ylabel('Latency (ms)')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot Energy Comparison
axes[1].bar(methods, energies, color=['#1f77b4', '#ff7f0e'])
axes[1].set_title('Energy Comparison')
axes[1].set_ylabel('Energy (mJ)')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Plot Stage 2 NNS: Top-k Similarity Scores for FET and Capacitor
# For simplicity, we'll just plot FET top-k similarities.
# If you want to combine both, you could plot them side-by-side or overlay.
x_positions = range(candidate_items)
width = 0.4

axes[2].bar([x - width/2 for x in x_positions], fet_topk_values.detach().numpy(), width=width, label='FET', color='#1f77b4')
axes[2].bar([x + width/2 for x in x_positions], cap_topk_values.detach().numpy(), width=width, label='Capacitor', color='#ff7f0e')
axes[2].set_title(f'Top-{candidate_items} Similarity Scores')
axes[2].set_xlabel('Item Rank')
axes[2].set_ylabel('Similarity Score')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)
axes[2].legend()

plt.tight_layout()
plt.show()