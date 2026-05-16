EXECUTE_CONFIG_FFILL = {
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
            "interpolation",
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
            "interpolation": {
                "enabled": True,
                "interp_valid": False,
                "X": {
                    "enabled": False,
                },
                "y": {
                    "enabled": True,
                    "freq": "1h",
                    "sparsify_step": 3,
                    "params": {
                        "method": "ffill",
                        "limit": 24,
                        "limit_area": "inside",
                        "limit_direction": "forward",
                    },
                },
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
            "meta_model": "ridge",
            "meta_model_params": {
                "alpha": 0.2,
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

EXECUTE_CONFIG_TIME = {
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
            "interpolation",
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
            "interpolation": {
                "enabled": True,
                "interp_valid": False,
                "X": {
                    "enabled": False,
                },
                "y": {
                    "enabled": True,
                    "freq": "1h",
                    "sparsify_step": 3,
                    "params": {
                        "method": "time",
                        "limit": 24,
                        "limit_area": "inside",
                        "limit_direction": "forward",
                    },
                },
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
            "meta_model": "ridge",
            "meta_model_params": {
                "alpha": 0.2,
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

EXECUTE_CONFIG_SPLINE = {
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
            "interpolation",
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
            "interpolation": {
                "enabled": True,
                "interp_valid": False,
                "X": {
                    "enabled": False,
                },
                "y": {
                    "enabled": True,
                    "freq": "1h",
                    "sparsify_step": 3,
                    "params": {
                        "method": "spline",
                        "order": 3,
                        "limit": 24,
                        "limit_area": "inside",
                        "limit_direction": "forward",
                    },
                },
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
            "meta_model": "ridge",
            "meta_model_params": {
                "alpha": 0.2,
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

EXECUTE_CONFIG_KNN = {
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
            "knn",
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
            "knn": {
                "enabled": True,
                "interp_valid": False,
                "params": {
                    "freq": "1h",
                    "n_neighbors": 3,
                    "weight": "distance",
                    "index_as_feature": False,
                    "drop_big_gap": True,
                    "max_gap": 14 * 24,
                },
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
            "meta_model": "ridge",
            "meta_model_params": {
                "alpha": 0.2,
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

EXECUTE_CONFIG_LOESS = {
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
            "interpolation",
            "smoother",
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
            "interpolation": {
                "enabled": True,
                "interp_valid": False,
                "X": {
                    "enabled": False,
                },
                "y": {
                    "enabled": True,
                    "freq": "1h",
                    "sparsify_step": 3,
                    "params": {
                        "method": "time",
                        "order": 3,
                        "limit": 24,
                        "limit_area": "inside",
                        "limit_direction": "forward",
                    },
                },
            },
            "smoother": {
                "enabled": True,
                "smooth_valid": False,
                "X": {"enabled": False, "method": "mean", "params": {"limit": 12}},
                "y": {"enabled": True, "method": "loess", "params": {"frac": 0.005}},
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
            "meta_model": "ridge",
            "meta_model_params": {
                "alpha": 0.2,
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

EXECUTE_CONFIG_GPR = {
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
            "gpr",
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
            "gpr": {
                "enabled": True,
                "interp_valid": False,
                "params": {
                    "freq": "1h",
                    "index_as_feature": False,
                    "kernel": "matern",
                    "nu": 2.5,  # float("inf"),
                    "n_restarts_optimizer": 30,
                    "k_confidence": 1,
                    "drop_big_gap": False,
                    "max_gap": 14 * 24,
                },
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
            "meta_model": "ridge",
            "meta_model_params": {
                "alpha": 0.2,
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
