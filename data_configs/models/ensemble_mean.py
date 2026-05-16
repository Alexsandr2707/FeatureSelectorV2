EXECUTE_CONFIG = {
    "random_seed": 0,  # 793
    "dataset": {
        "name": "UZK",
        "features_path": "data/uzk.csv",
        "target_path": "data/uzk_lab.csv",
        "target_column": "UZK.Q.81AY00108.FINALPOINT",
        "freq": "1h",
    },
    "preprocess": {
        "steps_order": [
            "feature_selector",
            "drop_intervals",
            "filter",
            "splitter",
            "scaler",
            "shifter",
        ],
        "steps_configs": {
            "shifter": {"enabled": True, "horizon": 1, "freq": "1h"},
            "drop_intervals": {
                "enabled": True,
                "intervals": [["2021", "2022-10-16"], ["2023-09-20", "2024"]],
            },
            "filter": {
                "enabled": True,
                "X": {"enabled": False},
                "y": {
                    "enabled": True,
                    "params": {"freq": "1h", "filter_freq": "1W", "max_diff": 30},
                },
            },
            "scaler": {
                "enabled": True,
                "X": {"enabled": True, "dtype": "standard"},
                "y": {"enabled": True, "dtype": "standard"},
            },
            "feature_selector": {
                "enabled": True,
                "dtype": "static",
                "params": {
                    "select_features": [
                        "81TI10143",
                        "81TI10126",
                        "81FIL30066",
                        "81LILH40012",
                        "81TIH11209",
                        "81FCL30063",
                        "81FI30052",
                        "81TI10123",
                    ],
                },
            },
            "splitter": {
                "enabled": True,
                "params": {"train_size": 0.6},
            },
        },
    },
    "model": {
        "model_type": "ensemble",
        "params": {
            "est_type": "rnn",
            "n_est": 10,
            "meta_model": "mean",
            "meta_model_params": {
                "weights_type": "equal",
                "fit_intercept": True,
            },
            "split_method": "expanded_window",  # expanded_window, basic
            "split_method_params": {"init_frac": 0.8, "end_frac": 0.2},
            "est_params": {
                "trainer": {
                    "epochs": 200,
                    "batch": 128,
                    "early_stoping": 200,
                },
                "model": {
                    "lag": 48,
                    "gru": [16, 1],
                    "decay": 0.01,
                    "lr": 1e-2,
                    "min_lr": 1e-2,
                    "use_best_model": True,
                },
            },
        },
    },
}
