import wandb

wandb.login()  # 或依赖环境变量 WANDB_API_KEY
run = wandb.init(
    project="test-init-timeout",
    settings=wandb.Settings(init_timeout=180)  # 临时加长到 180 秒
)
print("Init ok:", run is not None)
wandb.finish()
