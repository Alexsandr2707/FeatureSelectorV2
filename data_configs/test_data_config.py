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
            "outliers",
            "filter",
            "interpolation",
            "smoother",
            "differ",
            "scaler",
            "feature_selector",
            "splitter",
        ],
        "steps_configs": {
            "drop_intervals": {
                "enabled": True,
                "intervals": [["2021", "2022-10-16"], ["2023-09-20", "2024"]],
            },
            "differ": {"enabled": False, "params": {"how": "add"}},
            "filter": {
                "enabled": True,
                "X": {"enabled": False},
                "y": {
                    "enabled": True,
                    "params": {"freq": "1h", "filter_freq": "1W", "max_diff": 30},
                },
            },
            "outliers": {
                "enabled": True,
                "X": {
                    "enabled": False,
                    "scope": "local",
                    "params": {"dtype": "drop", "window": 24, "k": 1.5},
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
                    "enabled": True,
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
                        "limit": 48,
                        "limit_area": "inside",
                        "limit_direction": "both",
                    },
                },
            },
            "smoother": {
                "enabled": True,
                "X": {"enabled": True, "params": {"limit": 12}},
                "y": {"enabled": True, "params": {"limit": 24}},
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
                    "pls_depth": 3,
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
            "decay": 0.5,
            "lr": 1e-2,
            "min_lr": 1e-2,
        },
    },
}
