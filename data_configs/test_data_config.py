EXECUTE_CONFIG = {
    "dataset": {
        "name": "UZK",
        "features_path": "data/uzk.csv",
        "target_path": "data/uzk_lab.csv",
        "target_column": "UZK.Q.81AY00108.FINALPOINT",
        "freq": "1h",
    },
    "preprocess": {
        "drop_intervals": {
            "enabled": True,
            "intervals": [["2021", "2022-10-16"], ["2023-09-20", "2024"]],
        },
        "filter": {
            "enabled": True,
            "X": {
                "enabled": False,
                "params": {"freq": "1h", "filter_freq": "1W", "max_diff": 200},
            },
            "y": {
                "enabled": True,
                "params": {"freq": "1h", "filter_freq": "1W", "max_diff": 50},
            },
        },
        "outliers": {
            "enabled": True,
            "X": {
                "enabled": True,
                "scope": "global",
                "params": {"dtype": "drop", "window": 7 * 24, "k": 30},
            },
            "y": {
                "enabled": False,
                "scope": "local",
                "params": {"dtype": "clip", "window": 128, "k": 1.5},
            },
        },
        "interpolation": {
            "enabled": True,
            "X": {
                "enabled": False,
                "freq": "1h",
                "params": {
                    "method": "spline",
                    "order": 3,
                    "limit": 24,
                    "limit_area": "inside",
                    "limit_direction": "both",
                },
            },
            "y": {
                "enabled": True,
                "freq": "1h",
                "params": {
                    "method": "spline",
                    "order": 3,
                    "limit": 6,
                    "limit_area": "inside",
                    "limit_direction": "both",
                },
            },
        },
        "scaler": {
            "enabled": True,
            "X": {"enabled": True, "dtype": "standard"},
            "y": {"enabled": True, "dtype": "standard"},
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
    "model": {
        # "model_type": "RNN",
        "trainer": {
            "epochs": 300,
            "batch": 128,
            "early_stoping": 30,
        },
        "model": {
            "lag": 48,
            "gru": [16, 1],
            "l2": 0.00,
            "decay": 0.01,
            "lr": 1e-2,
            "min_lr": 1e-4,
        },
    },
}
