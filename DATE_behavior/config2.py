
# Configuration Parameters
Nu = 10**7  # Number of users
Ni = 10**6  # Number of items
d = 64      # Embedding dimension
qbit = 32   # Quantization bit-width (e.g., 32-bit floats)
sa_rows, sa_cols = 128, 128  # Size of the systolic array
candidate_items = 100  # Number of candidate items

# Mock latency and energy parameters
fet_mac_lat, fet_mac_en = 5, 5e7
cap_mac_lat, cap_mac_en = 10, 0.1