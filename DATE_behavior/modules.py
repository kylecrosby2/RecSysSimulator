import math
def cap_in_write_rm(x,qbit,sa_rows,sa_cols,cap_in_wr_rm_lat,cap_in_wr_rm_en,lat,en):
  batch,seq_len,d_model = x.shape
  xbar_count = math.ceil((batch*seq_len*d_model*qbit) / (sa_rows*sa_cols))
  operations = xbar_count*2 # Two operands x and pe 96.
  lat = lat+1*cap_in_wr_rm_lat
  en  = en+operations*cap_in_wr_rm_en
  return lat,en

def add_cap_rm(x,qbit,sa_rows,sa_cols,add_cap_rm_lat,add_cap_rm_en,lat,en):
  batch,seq_len,d_model = x.shape
  xbar_count = math.ceil((batch*seq_len*d_model*qbit) / (sa_rows*sa_cols))  # 48
  operation_count = (batch*seq_len*d_model*qbit)/sa_cols  # bit count/sa_cols. NB: We are computing all cols in parallel.
  lat = lat+32*add_cap_rm_lat
  en  = en+operation_count*add_cap_rm_en
  return lat,en


def fet_mac(x,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en):
  print(f'fet_mac_lat: {fet_mac_lat} fet_mac_en: {fet_mac_en}')
  batch,seq_len,d_model = x.shape

  xbar_count = (d_model*d_model*qbit) / (sa_rows*sa_cols)
  wl_count   = seq_len  # Asif: qbit we do not need. WL do not need to be quantized.

  lat_operation_count = batch*wl_count
  en_operation_count  = batch*xbar_count*wl_count
  print(f'debug fet_mac: xbar_count: {xbar_count} wl_count: {wl_count} lat_operation_count: {lat_operation_count} en_operation_count: {en_operation_count} fet_mac_lat: {fet_mac_lat} fet_mac_en: {fet_mac_en}')
  lat = lat+fet_mac_lat*lat_operation_count 
  en  = en+fet_mac_en*en_operation_count
  return lat,en

def cap_mac(x,y,qbit,sa_rows,sa_cols,cap_mac_lat,cap_mac_en,lat,en):
  print(f'cap_mac_lat: {cap_mac_lat} cap_mac_en: {cap_mac_en}')
  x_batch,x_num_heads,x_seq_len,x_dh = x.shape
  y_batch,y_num_heads,y_seq_len,y_dh = y.shape

  xbar_count = (y_seq_len*y_dh*qbit) / (sa_rows*sa_cols)
  wl_count = x_seq_len

  lat_operation_count = wl_count*x_batch
  en_operation_count  = xbar_count*wl_count*x_batch*x_num_heads

  lat = lat+lat_operation_count*cap_mac_lat  
  en  = en+en_operation_count*cap_mac_en

  print(f'debug cap_mac: xbar_count: {xbar_count} wl_count: {wl_count} lat_operation_count: {lat_operation_count} en_operation_count: {en_operation_count} cap_mac_lat: {cap_mac_lat} cap_mac_en: {cap_mac_en}')

  return lat,en
  

def cap_bl_to_sa_wr_rm(q,qbit,sa_rows,sa_cols,cap_bl_to_sa_wr_rm_lat,cap_bl_to_sa_wr_rm_en,lat,en):
  batch,num_heads,seq_len,d_model = q.shape  # 4d
  xbar_count = math.ceil((batch*num_heads*seq_len*d_model*qbit) / (sa_rows*sa_cols))  #q,k,v. Mehdi
  lat = lat+1*cap_bl_to_sa_wr_rm_lat  # 144
  en  = en+cap_bl_to_sa_wr_rm_en*xbar_count
  return lat,en
def cap_bl_to_sa_wr_rm3(q,qbit,sa_rows,sa_cols,cap_bl_to_sa_wr_rm_lat,cap_bl_to_sa_wr_rm_en,lat,en):
  batch,seq_len,d_model = q.shape  # 4d
  xbar_count = math.ceil((batch*seq_len*d_model*qbit) / (sa_rows*sa_cols))  #q,k,v. Mehdi
  lat = lat+cap_bl_to_sa_wr_rm_lat*1
  en  = en+cap_bl_to_sa_wr_rm_en*xbar_count
  return lat,en

def fet_bl_to_sa_wr_rm(x,qbit,sa_rows,sa_cols,fet_bl_to_sa_wr_rm_lat,fet_bl_to_sa_wr_rm_en,lat,en):
  batch,seq_len,d_model = x.shape  # 3d
  xbar_count = math.ceil((batch*seq_len*d_model*qbit) / (sa_rows*sa_cols))  #q,k,v. Mehdi
  lat = lat+fet_bl_to_sa_wr_rm_lat*1
  en  = en+fet_bl_to_sa_wr_rm_en*xbar_count
  return lat,en

def lut(x,qbit,sa_rows,sa_cols,p_row_rd_fet_lat,p_row_rd_fet_en,p_row_wr_cap_lat,p_row_wr_cap_en,lut_cap_add_cap_wr_lat,lut_cap_add_cap_wr_en,cap_mac_lat,cap_mac_en,lat,en):
  batch,num_heads,seq_len,d_model = x.shape
  lat_ix0 = lat
  en_ix0 = en
  # softmax lut
  operations = (batch*num_heads*seq_len*d_model)  # Mehdi. seq_len = d_model here. Assumption: How much parallelism can be done here? 256 is LUT length.
  parallelism = 2048 
  lat = lat+(p_row_rd_fet_lat+p_row_wr_cap_lat)*operations/parallelism  # Did not assume parallelism here. latency/energy is for 8 bit scaled.
  en  = en+(p_row_rd_fet_en+p_row_wr_cap_en)*operations/1000 # Convert back to pJ
  lat_ix1 = lat
  en_ix1 = en
  # cap_add
  lat = lat+batch*seq_len*qbit*lut_cap_add_cap_wr_lat/sa_rows
  en  = en+batch*num_heads*seq_len*d_model*qbit*lut_cap_add_cap_wr_en/sa_rows
  lat_ix2 = lat
  en_ix2 = en
  # scaling
  xbar_count = (seq_len*d_model*qbit) / (sa_rows*sa_cols)
  wl_count = 1  # scaling not mac

  lat_operation_count = wl_count*batch
  en_operation_count  = xbar_count*wl_count*batch*num_heads

  lat = lat+lat_operation_count*cap_mac_lat
  en  = en+en_operation_count*cap_mac_en
  print(f'Operation Softmax Layerwise: Softmax lat: {(lat_ix1-lat_ix0)/1000} us en: {(en_ix1-en_ix0)/1000000}uJ')
  print(f'Operation Softmax Layerwise: cap_add lat: {(lat_ix2-lat_ix1)/1000} us en: {(en_ix2-en_ix1)/1000000}uJ')
  print(f'Operation Softmax Layerwise: scaling lat: {(lat-lat_ix2)/1000} us en: {(en-en_ix2)/1000000}uJ')
  return lat,en

def add_fet_rm(x,qbit,sa_rows,sa_cols,add_fet_rm_lat,add_fet_rm_en,lat,en):
  batch,seq_len,d_model = x.shape
  operation_count = (batch*seq_len*d_model*qbit)/sa_cols  # bit count/sa_cols. NB: We are computing all cols in parallel.
  lat = lat + 32*add_fet_rm_lat  # Assumption: How much parallelism can be done here? x32x8x64?
  en = en + operation_count*add_fet_rm_en
  return lat,en


def fet_mac_ff1(x,ff_hidden_dim,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en):
  batch,seq_len,d_model = x.shape
  print(f'fet_mac_lat: {fet_mac_lat} fet_mac_en: {fet_mac_en}')
  print(f'batch: {batch}, seq_len: {seq_len}, ff_hidden_dim: {ff_hidden_dim}, d_model: {d_model}')

  xbar_count = (d_model*ff_hidden_dim*qbit)/(sa_rows*sa_cols)
  wl_count   = seq_len

  lat_operation_count = wl_count*batch
  en_operation_count = xbar_count*wl_count*batch   

  lat = lat+fet_mac_lat*lat_operation_count 
  en  = en +fet_mac_en*en_operation_count

  print(f'debug fet_mac: xbar_count: {xbar_count} wl_count: {wl_count} lat_operation_count: {lat_operation_count} en_operation_count: {en_operation_count} fet_mac_lat: {fet_mac_lat} fet_mac_en: {fet_mac_en}')

  return lat,en

def fet_mac_ff2(x,d_model,qbit,sa_rows,sa_cols,fet_mac_lat,fet_mac_en,lat,en):
  batch,seq_len,ff_hidden_dim = x.shape
  print(f'fet_mac_lat: {fet_mac_lat} fet_mac_en: {fet_mac_en}')
  print(f'batch: {batch}, seq_len: {seq_len}, ff_hidden_dim: {ff_hidden_dim}, d_model: {d_model}')
  xbar_count = (d_model*ff_hidden_dim*qbit)/(sa_rows*sa_cols)
  wl_count   = seq_len

  lat_operation_count = wl_count*batch
  en_operation_count = xbar_count*wl_count*batch

  lat = lat+fet_mac_lat*lat_operation_count  
  en  = en +fet_mac_en*en_operation_count

  print(f'debug fet_mac: xbar_count: {xbar_count} wl_count: {wl_count} lat_operation_count: {lat_operation_count} en_operation_count: {en_operation_count} fet_mac_lat: {fet_mac_lat} fet_mac_en: {fet_mac_en}')

  return lat,en

def embedding(input_ids,vocab_size,d_model,qbit,sa_rows,sa_cols,row_rd_fet_lat,row_rd_fet_en,row_wr_cap_lat,row_wr_cap_en,add_cap_rm_lat,add_cap_rm_en,lat,en):
  count = 0
  for _ in input_ids:
    for input_id in _:
      if input_id !=0:
        count = count+1
  print(f"input_ids have {count} tokens")
  
  bit_count = count*d_model*qbit
  operation_count = bit_count/sa_cols    # We are reading 1 row at a time. 
  lat = lat+2*operation_count*(row_rd_fet_lat+row_wr_cap_lat)/count   # Each word is searched in parallel. Upper limit 512. 2 multiplication for we+pe
  en = en+2*operation_count*(row_rd_fet_en+row_wr_cap_en)/1000  # from fJ to pJ

  return lat,en


