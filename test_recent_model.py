from envi_rnn import *

n = load_model("models/lr_5e-05_ep_5_bs_100_ps_15_RNN_L3_H10_log_T_norm_T_prune_F_state_dict")
test(n)
