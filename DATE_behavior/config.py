relu_lat = 0  # Asif...Confirm skip or not.
relu_en = 0

fet_mac_lat = 10.1  # 1.1/10.1
fet_mac_en = 4.0  # 74/60.72
cap_mac_lat = 10.1  # 1.1/10.1
cap_mac_en = 3.8  # 71.07/56.99

cap_in_wr_rm_lat = 32    # xbar write
cap_in_wr_rm_en = 3.68   # xbar write
fet_in_wr_rm_lat = 1280  # xbar write
fet_in_wr_rm_en = 2.68   # xbar write

fet_bl_to_sa_wr_rm_lat = 1280
fet_bl_to_sa_wr_rm_en = 2.68
cap_bl_to_sa_wr_rm_lat = 32
cap_bl_to_sa_wr_rm_en = 3.68

# LUT only
p_row_rd_fet_lat = 0.34    # 8bit fet read latency/energy 
p_row_rd_fet_en  = 144.13  # fJ
p_row_wr_cap_lat = 0.0625  # 8 bit cap write latency/energy
p_row_wr_cap_en =  7.1875  # fJ...3.68/64/8
lut_cap_add_cap_wr_lat = 3.45
lut_cap_add_cap_wr_en  = 4.68

add_cap_rm_lat = 22.95  # For 64 bits. write back included
add_cap_rm_en = 4.64    # For 64 bits. Write back included

add_fet_rm_lat = 21.93  # For 64 bits. Write back included
add_fet_rm_en = 2.27    # For 64 bits. Write back included

mov_bl_to_wl_lat = 0
mov_bl_to_wl_en = 0
mov_fet_to_wl_lat = 0
mov_fet_to_wl_en = 0
mov_cap_to_wl_lat = 0
mov_cap_to_wl_en = 0

sa_rows = 64
sa_cols = 64
qbit = 8
d_model = 64     # 768/1024
num_heads = 1    # 12/16
num_layers = 1   # 12/24
head_dim = int(d_model/num_heads)
vocab_size = 30522
ff_hidden_dim = 3072  # 3072/4096
dropout = 0.1
seq_len = 64 #subh 512
max_length = 64 #subh 512
lat = 0
en = 0

# Embedding only
row_rd_fet_lat = 0.34   # read latency in ns per 64 bit.
row_rd_fet_en  = 514.57 # read energy per 64 bit in fJ.
row_wr_cap_lat = 0.5    # write latency per 64 bit.
row_wr_cap_en  = 57.5   # write energy per 64 bit in fJ.
