from config import *
from modules import *
from datasets import load_dataset


from transformers import AutoTokenizer

# Make dataset here

import torch



import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# User Embedding Table
# Item Embedding Table


# Hardware start

# User Embedding Lookup 

# Item Embedding Lookup


# Concatentation (matrix addition)
lat,en = embedding(input_ids,vocab_size,d_model,qbit,sa_rows,sa_cols,row_rd_fet_lat,row_rd_fet_en,row_wr_cap_lat,row_wr_cap_en,add_cap_rm_lat,add_cap_rm_en,lat,en)
lat_after_we = lat
en_after_we = en
print('we and pe write: before add_cap to get xpe: latency in us: energy in uJ:')
print(lat/1000,en/1000000)
lat,en = add_cap_rm(x,qbit,sa_rows,sa_cols,add_cap_rm_lat,add_cap_rm_en,lat,en)
print(f'we+pe = xpe: output: latency in us, energy in uJ')
print((lat-lat_after_we)/1000,(en-en_after_we)/1000000)
print('\n')
lat_embedding_only = lat/1000
en_embedding_only = en/1000000

print(f'Operation0: Embedding Layer latency: {lat_embedding_only} us, and Energy: {en_embedding_only} uJ')

#lat,en = cap_in_write_rm(x,qbit,sa_rows,sa_cols,cap_in_wr_rm_lat,cap_in_wr_rm_en,lat,en)
#print('lat,en: cap_in_write output after writing x and pe')
#print(lat,en)
#lat,en = add_cap_rm(x,qbit,sa_rows,sa_cols,add_cap_rm_lat,add_cap_rm_en,lat,en)
#print('lat,en: add_cap_to_fet: output after adding x and pe and writing xpe')
#print(lat,en)
#lat,en = fet_bl_to_sa_wr_rm(x,qbit,sa_rows,sa_cols,fet_bl_to_sa_wr_rm_lat,fet_bl_to_sa_wr_rm_en,lat,en)
#print('lat,en: after writing xpe in FET')
#print(lat,en)
lat_before_mha = lat
en_before_mha = en
# Hardware end

w_q = nn.Linear(d_model,d_model,bias=False)
w_k = nn.Linear(d_model,d_model,bias=False)
w_v = nn.Linear(d_model,d_model,bias=False)
w_o = nn.Linear(d_model,d_model,bias=False)

#print('w_q.weight')
#print(w_q.weight)
print('w_q.shape')
print(w_q.weight.shape)

# Hardware begin
lat,en = fet_mac(x,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en)
lat,en = fet_mac(x,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en)
lat,en = fet_mac(x,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en)
print('lat,en: fet mac output compute q,k,v')
print(lat/1000,en/1000000)

fet_mac_lat1 = (lat - lat_before_mha)/3000
fet_mac_en1 = (en  - en_before_mha)/3000000
print(f'Operation1: Latency and energy from MUL of X and W_Q: {fet_mac_lat1} us and {fet_mac_en1} uJ')

# Hardware end

query = w_q(x)
key   = w_k(x) 
value = w_v(x)
print("query.shape")
print(query.shape)

N        = query.shape[0]        # batch size. Keep it intact.
seq_len  = query.shape[1]  # sequence_length
head_dim = int(query.shape[2]/num_heads)

query = query.reshape(N, seq_len, num_heads, head_dim)
key   = key.reshape(N, seq_len, num_heads, head_dim)
value = value.reshape(N, seq_len, num_heads, head_dim)

query = query.transpose(1,2)
key = key.transpose(1,2)
value = value.transpose(1,2)

print('query.shape')
print(query.shape)

# Hardware begin
lat1 = lat
en1 = en
lat,en = cap_bl_to_sa_wr_rm(query,qbit,sa_rows,sa_cols,cap_bl_to_sa_wr_rm_lat,cap_bl_to_sa_wr_rm_en,lat,en)
lat,en = cap_bl_to_sa_wr_rm(query,qbit,sa_rows,sa_cols,cap_bl_to_sa_wr_rm_lat,cap_bl_to_sa_wr_rm_en,lat,en)
lat,en = cap_bl_to_sa_wr_rm(query,qbit,sa_rows,sa_cols,cap_bl_to_sa_wr_rm_lat,cap_bl_to_sa_wr_rm_en,lat,en)
print('lat,en: after cap_bl_to_sa_wr_rm: q,k,v write')
print(lat/1000,en/1000000)
cap_wr_lat = (lat-lat1)/3000
cap_wr_en = (en-en1)/1000000
lat = lat1+cap_wr_lat
print(f'Operation2: cap_write q,k,v: Latency = {cap_wr_lat} us and Energy = {cap_wr_en} uJ')

lat2 = lat
en2 = en
lat,en = cap_mac(query,key,qbit,sa_rows,sa_cols,cap_mac_lat,cap_mac_en,lat,en)
print('lat,en: after qkt mac computation')
print(lat/1000,en/1000000)
cap_mac_lat3 = (lat-lat2)/1000
cap_mac_en3 = (en-en2)/1000000
print(f'Operation3: cap mac QK_T: Latency = {cap_mac_lat3} us and Energy = {cap_mac_en3} uJ')

# Hardware end

attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
print('attention_scores.shape')
print(attention_scores.shape)
# Hardware Start
lat3 = lat
en3 = en
lat,en = cap_bl_to_sa_wr_rm(attention_scores,qbit,sa_rows,sa_cols,cap_bl_to_sa_wr_rm_lat,cap_bl_to_sa_wr_rm_en,lat,en)
print('qk_t write output: lat,en in us and uJ')
print(lat/1000,en/1000000)
lat_before_lut = lat
en_before_lut = en
lat,en = lut(attention_scores,qbit,sa_rows,sa_cols,p_row_rd_fet_lat,p_row_rd_fet_en,p_row_wr_cap_lat,p_row_wr_cap_en,lut_cap_add_cap_wr_lat,lut_cap_add_cap_wr_en,cap_mac_lat,cap_mac_en,lat,en)
lat_only_lut = (lat-lat_before_lut)/1000
en_only_lut = (en-en_before_lut)/1000000
print('softmax output: lat,en in us and uJ')
print(lat/1000,en/1000000)
print('\n')

lut_lat = (lat-lat3)/1000
lut_en = (en-en3)/1000000
print(f'Operation4: Softmax and writing QK_T together:  Latency = {lut_lat} us and Energy = {lut_en} uJ')

print(f'Operation softmax: Latency = {lat_only_lut} us and Energy = {en_only_lut}')
# Hardware End



attention_scores = attention_scores.softmax(dim=-1)

print("attention_scores.shape")
print(attention_scores.shape)

# Hardware start
lat4 = lat
en4 = en
lat,en = cap_mac(attention_scores,value,qbit,sa_rows,sa_cols,cap_mac_lat,cap_mac_en,lat,en)
print('qkt*v output: lat,en in us and uJ')
print(lat/1000,en/1000000)
lat_cap_mac = (lat-lat4)/1000
lat_cap_en = (en-en4)/1000000
print(f'Operation5: CAP-MAC Softmax output*V computation:  Latency = {lat_cap_mac} us and Energy = {lat_cap_en} uJ')
print('\n')
# Hardware end


attention_matrix = attention_scores @ value
print('attention_matrix.shape')
print(attention_matrix.shape)
attention_matrix = attention_matrix.transpose(1, 2).contiguous().view(attention_matrix.shape[0], -1, num_heads * head_dim)

# Hardware start
lat_before_write_mha = lat
en_before_write_mha = en
lat,en = cap_bl_to_sa_wr_rm3(attention_matrix,qbit,sa_rows,sa_cols,cap_bl_to_sa_wr_rm_lat,cap_bl_to_sa_wr_rm_en,lat,en)
lat_only_write_mha = (lat-lat_before_write_mha)/1000
en_only_write_mha = (en-en_before_write_mha)/1000000
print(f'Operation5.5: lat_only_write_mha = {lat_only_write_mha} us and en_only_write_mha = {en_only_write_mha} uJ\n')
print('after writing soft(qkt)*v in us and uJ')
print(lat/1000,en/1000000)
lat_after_mha = lat
en_after_mha = en
print('\n')

lat5 = lat
en5 = en
lat,en = fet_mac(attention_matrix,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en)
print('MHA output lat,en results in us and uJ')
print(lat/1000,en/1000000)
lat_fet_mac = (lat-lat5)/1000
en_fet_mac = (en-en5)/1000000
print(f'Operation6: FET MAC: Multiplication with W_O:  Latency = {lat_fet_mac} us and Energy = {en_fet_mac} uJ')
print('\n')
# Hardware end

attention_matrix = w_o(attention_matrix)
attention_matrix = dropout_layer(attention_matrix)

print('attention_matrix.shape')
print(attention_matrix.shape)


# Hardware start
lat,en = fet_bl_to_sa_wr_rm(attention_matrix,qbit,sa_rows,sa_cols,fet_bl_to_sa_wr_rm_lat,fet_bl_to_sa_wr_rm_en,lat,en)
print('After writing MHA output to the FET: lat,en in us and uJ')
print(lat/1000,en/1000000)

lat_before_norm_add = lat
en_before_norm_add = en
lat,en = add_fet_rm(x,qbit,sa_rows,sa_cols,add_fet_rm_lat,add_fet_rm_en,lat,en)
lat_only_norm_add = (lat-lat_before_norm_add)/1000
en_only_norm_add = (en-en_before_norm_add)/1000000

print('add_dram:x with mha output:  lat,en in us and uJ')
print(lat/1000,en/1000000)
print('\n')

print(f'Operation norm ADD: lat_only_norm_add: {lat_only_norm_add} us and en_only_norm_add = {en_only_norm_add} uJ\n')
# Hardware end

layernorm1 = nn.LayerNorm(d_model)
layernorm2 = nn.LayerNorm(d_model)

x1 = x+attention_matrix  # Delete?

x = layernorm1(x+attention_matrix)

print('layernorm output1 shape:')
print(x.shape)

# Hardware start
#subh: comment: included in add_fet_rm-->> lat,en = fet_bl_to_sa_wr_rm(x,qbit,sa_rows,sa_cols,fet_bl_to_sa_wr_rm_lat,fet_bl_to_sa_wr_rm_en,lat,en)
print('after writing the output of the x+mha_output in the fet: lat,en in us and uJ')
print(lat/1000,en/1000000)
print('\n')
# Hardware end


feed_forward_layer1 = nn.Linear(d_model,ff_hidden_dim)
feed_forward_layer2 = nn.Linear(ff_hidden_dim,d_model)
ff1 = feed_forward_layer1(x)

print('ff1 layer output shape:')
print(ff1.shape)

# Hardware start
lat7 = lat
en7 = en
lat,en = fet_mac_ff1(x,ff_hidden_dim,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en)
print('ff1 mac output: lat,en in us and uJ')
print(lat/1000,en/1000000)
lat_fet_mac_ff1 = (lat-lat7)/1000
en_fet_mac_ff1 = (en-en7)/1000000
print(f'Operation7: FET MAC FF1:  Latency = {lat_fet_mac_ff1} us and Energy = {en_fet_mac_ff1} uJ')
lat,en = fet_bl_to_sa_wr_rm(ff1,qbit,sa_rows,sa_cols,fet_bl_to_sa_wr_rm_lat,fet_bl_to_sa_wr_rm_en,lat,en)
print('ff1 mac output FET write: lat,en in us and uJ')
print(lat/1000,en/1000000)
print('\n')
# Hardware end


# DATE lat, en = my_relu(ff1,qbit,sa_rows,sa_cols,relu_read_latency,relu_read_energy,relu_update_latency,relu_update_energy,lat,en)
# DATE print('ReLU layer output: lat and en is: ')

ff_relu = F.relu(ff1)
print('ff_relu shape:')
print(ff_relu.shape)

ff2 = feed_forward_layer2(ff_relu)

# Hardware start
lat8 = lat
en8 = en
lat,en = fet_mac_ff2(ff_relu,d_model,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en)
print('FF2 mac operation output: lat,en in us and uJ')
print(lat/1000,en/1000000)
lat_fet_mac_ff2 = (lat-lat8)/1000
en_fet_mac_ff2 = (en-en8)/1000000
print(f'Operation8: FET MAC FF2:  Latency = {lat_fet_mac_ff2} us and Energy = {en_fet_mac_ff2} uJ')
lat,en = fet_bl_to_sa_wr_rm(ff2,qbit,sa_rows,sa_cols,fet_bl_to_sa_wr_rm_lat,fet_bl_to_sa_wr_rm_en,lat,en)
print('ff2 mac output FET write: lat,en in us and uJ')
print(lat/1000,en/1000000)
print('\n')
# Hardware end

ff_out = dropout_layer(ff2)
print('ff_out.shape')
print(ff_out.shape)
print(x.shape)
x = layernorm2(x+ff_out)

# Hardware start
lat_before_norm_add2 = lat
en_before_norm_add2 = en
lat,en = add_fet_rm(x,qbit,sa_rows,sa_cols,add_fet_rm_lat,add_fet_rm_en,lat,en)
lat_only_norm_add2 = (lat-lat_before_norm_add2)/1000
en_only_norm_add2  = (en-en_before_norm_add2)/1000000
print(f'Operation norm ADD2: latency = {lat_only_norm_add2} us and energy = {lat_only_norm_add2} uJ\n')
print('Encoder layer output: add_dram x+ff2 : lat,en in us and uJ')
print(lat/1000,en/1000000)
# subh: comment: included in add_fet_rm-->> lat,en = fet_bl_to_sa_wr_rm(x,qbit,sa_rows,sa_cols,fet_bl_to_sa_wr_rm_lat,fet_bl_to_sa_wr_rm_en,lat,en)
print('Encoder output write FET: lat,en in us and uJ')
print(lat/1000,en/1000000)
lat_after_encoder = lat
en_after_encoder = en
# Hardware end
print('\n')

#print('Encoder layer output shape:')
#print(x.shape)

#print("input_ids")
#print(input_ids)
#print(input_ids.shape)


print("lat_before_mha,lat_after_mha,lat_after_encoder in us and uJ")
print(lat_before_mha/1000,lat_after_mha/1000,lat_after_encoder/1000)
print("en_before_mha,en_after_mha,en_after_encoder in us and uJ")
print(en_before_mha/1000000,en_after_mha/1000000,en_after_encoder/1000000)
print('\n')

end_to_end_latency = lat_before_mha + num_layers*(lat_after_encoder - lat_before_mha)
end_to_end_energy  = en_before_mha  + num_layers*(en_after_encoder  - en_before_mha)
end_to_end_latency = end_to_end_latency/1000000 # ns-> ms
end_to_end_energy = end_to_end_energy/1000000000 # pJ-> mJ
print(f'end_to_end_latency is {end_to_end_latency} ms and end_to_end_energy is {end_to_end_energy} mJ')


