from envi_rnn import *

n = load_model("models/lr_2e-06_ep_60000_bs_500_ps_10_RNN_L3_H10_log_T_norm_T_prune_F_state_dict")
test(n)
