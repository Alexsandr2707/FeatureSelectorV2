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
            "drop_intervals",
            "filter",
            "interpolation",
            "scaler",
            "feature_selector",
            "splitter",
        ],
        "steps_configs": {
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
            "interpolation": {
                "enabled": True,
                "X": {"enabled": False},
                "y": {
                    "enabled": True,
                    "freq": "1h",
                    "params": {
                        "method": "spline",
                        "order": 3,
                        "limit": 3,
                        "limit_area": "inside",
                        "limit_direction": "both",
                    },
                },
            },
            "scaler": {
                "enabled": True,
                "X": {"enabled": True, "dtype": "robust"},
                "y": {"enabled": True, "dtype": "robust"},
            },
            "feature_selector": {
                "enabled": True,
                "dtype": "pls",
                "params": {"pls_depth": 3},
            },
            "splitter": {
                "enabled": True,
                "params": {"train_size": 0.6},
            },
        },
    },
    "model": {
        "trainer": {
            "epochs": 200,
            "batch": 128,
            "early_stoping": 30,
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
