params = {
    # data config
    'height': 28,
    'width': 28,
    'depth': 1,
    'num_classes': 10,
    # training
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 50,
    # model params
    'reg_scale': 0.392,
    'm_plus': 0.9,
    'm_minus': 0.1,
    'mask_with_y': True,
    'lambda': 0.5,
    # Dataset
    'num_threads': 8,
    # config
    'step_save_summaries': 100,
    'checkpoint_path': ""
}
