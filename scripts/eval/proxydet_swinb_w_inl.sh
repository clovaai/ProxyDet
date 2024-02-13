export num_gpus="8"
export seed="7"
export config_file="configs/ProxyDet_SwinB_Lbase_INL.yaml"
export num_workers="2"
export eval_only="True"
export resume="True"
export pretrained_weight="models/proxydet_swinb_w_inl.pth"
bash scripts/train.sh