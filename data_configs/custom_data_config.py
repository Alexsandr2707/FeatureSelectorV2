EXECUTE_CONFIG = {
    "dataset": {
        "name": "UZK",
        "features_path": "data/uzk.csv",
        "target_path": "data/uzk_lab.csv",
        "target_column": "UZK.Q.81AY00108.FINALPOINT",
        "freq": "1h",
    },
    "preprocess": {
        "steps_order": [
            "shifter",
            "drop_intervals",
            "filter",
            "scaler",
            "feature_selector",
            "splitter",
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
                    # same as pls with pls_depth=2
                    "select_features": [
                        "81TI10126",
                        "81LILH40012",
                        "81FCL30063",
                        "81FI30052",
                    ],
                    # same as pls with pls_depth=3
                    # "select_features": [
                    #     "81TI10143",
                    #     "81TI10126",
                    #     "81FIL30066",
                    #     "81LILH40012",
                    #     "81TIH11209",
                    #     "81FCL30063",
                    #     "81FI30052",
                    #     "81TI10123",
                    # ],
                },
            },
            "splitter": {
                "enabled": True,
                "params": {"train_size": 0.6},
            },
        },
    },
    "model": {
        # "model_type": "RNN",
        "trainer": {
            "epochs": 200,
            "batch": 128,
            "early_stoping": 50,
        },
        "model": {
            "lag": 48,
            "gru": [16, 1],
            "l2": 0.00,
            "decay": 0.01,
            "lr": 1e-2,
            "min_lr": 1e-2,
        },
    },
}
