//! Parsing real config.json files (KV-relevant fields only, see tests/fixtures).

use kvscope_core::{parse_config, ConfigError, KvDtype, KvShape, LayerKind, NoKvReason};

fn count(layers: &[LayerKind], pred: impl Fn(&LayerKind) -> bool) -> usize {
    layers.iter().filter(|l| pred(l)).count()
}

#[test]
fn pythia_is_pure_mha() {
    let m = parse_config(include_str!("fixtures/pythia-1.4b-deduped.json")).unwrap();
    let shape = KvShape {
        kv_heads: 16,
        k_head_dim: 128,
        v_head_dim: 128,
    };
    assert_eq!(m.layers, vec![LayerKind::Full(shape); 24]);
    assert_eq!(shape.values_per_token(), 4096);
    assert_eq!(m.config_dtype, KvDtype::F16);
}

#[test]
fn gpt_oss_alternates_sliding_and_full() {
    let m = parse_config(include_str!("fixtures/gpt-oss-120b.json")).unwrap();
    let shape = KvShape {
        kv_heads: 8,
        k_head_dim: 64,
        v_head_dim: 64,
    };
    assert_eq!(m.layers.len(), 36);
    assert_eq!(m.layers[0], LayerKind::Sliding { shape, window: 128 });
    assert_eq!(m.layers[1], LayerKind::Full(shape));
    assert_eq!(
        count(&m.layers, |l| matches!(l, LayerKind::Sliding { .. })),
        18
    );
    // No dtype in this config: the planner must say it assumed one.
    assert_eq!(m.config_dtype, KvDtype::Bf16);
    assert!(m.notes.iter().any(|n| n.contains("assuming a bf16")));
}

#[test]
fn gemma4_31b_global_layers_have_their_own_shape() {
    let m = parse_config(include_str!("fixtures/gemma-4-31b-it.json")).unwrap();
    let local = KvShape {
        kv_heads: 16,
        k_head_dim: 256,
        v_head_dim: 256,
    };
    let global = KvShape {
        kv_heads: 4,
        k_head_dim: 512,
        v_head_dim: 512,
    };
    assert_eq!(m.layers.len(), 60);
    assert_eq!(
        count(&m.layers, |l| *l
            == LayerKind::Sliding {
                shape: local,
                window: 1024
            }),
        50
    );
    assert_eq!(count(&m.layers, |l| *l == LayerKind::Full(global)), 10);
    assert_eq!(m.layers[5], LayerKind::Full(global));
    assert!(m.notes.iter().any(|n| n.contains("attention_k_eq_v")));
}

#[test]
fn qwen36_35b_only_full_attention_layers_hold_kv() {
    let m = parse_config(include_str!("fixtures/qwen3.6-35b-a3b.json")).unwrap();
    let shape = KvShape {
        kv_heads: 2,
        k_head_dim: 256,
        v_head_dim: 256,
    };
    assert_eq!(m.layers.len(), 40);
    assert_eq!(m.kv_layer_count(), 10);
    assert_eq!(count(&m.layers, |l| *l == LayerKind::Full(shape)), 10);
    assert_eq!(
        count(&m.layers, |l| *l
            == LayerKind::NoKv(NoKvReason::LinearAttention)),
        30
    );
}

#[test]
fn qwen36_27b_differs_from_35b() {
    // The measured run had 10 KV layers with 2 KV heads: this is why it was the 35B.
    let m = parse_config(include_str!("fixtures/qwen3.6-27b.json")).unwrap();
    assert_eq!(m.kv_layer_count(), 16);
    assert_eq!(
        m.layers[3],
        LayerKind::Full(KvShape {
            kv_heads: 4,
            k_head_dim: 256,
            v_head_dim: 256
        })
    );
}

#[test]
fn glm47_flash_is_mla_with_rope_part() {
    let m = parse_config(include_str!("fixtures/glm-4.7-flash.json")).unwrap();
    let expanded = KvShape {
        kv_heads: 20,
        k_head_dim: 256,
        v_head_dim: 256,
    };
    assert_eq!(
        m.layers,
        vec![
            LayerKind::Latent {
                latent_dim: 512 + 64,
                expanded
            };
            47
        ]
    );
}

#[test]
fn nemotron_h_has_four_attention_layers() {
    let m = parse_config(include_str!("fixtures/nemotron-h-8b-base-8k.json")).unwrap();
    assert_eq!(m.layers.len(), 52);
    assert_eq!(m.kv_layer_count(), 4);
    assert_eq!(
        count(&m.layers, |l| *l == LayerKind::NoKv(NoKvReason::Mamba)),
        24
    );
    assert_eq!(
        count(&m.layers, |l| *l == LayerKind::NoKv(NoKvReason::Mlp)),
        24
    );
    let attn = KvShape {
        kv_heads: 8,
        k_head_dim: 128,
        v_head_dim: 128,
    };
    assert_eq!(count(&m.layers, |l| *l == LayerKind::Full(attn)), 4);
}

#[test]
fn errors_are_specific() {
    assert_eq!(parse_config("[]"), Err(ConfigError::NotAnObject));
    assert!(matches!(parse_config("{"), Err(ConfigError::Json(_))));
    assert_eq!(
        parse_config(r#"{"num_attention_heads": 8}"#),
        Err(ConfigError::Missing("num_hidden_layers"))
    );
    let mismatch = r#"{"num_hidden_layers": 3, "num_attention_heads": 4, "head_dim": 8,
                       "layer_types": ["full_attention", "full_attention"]}"#;
    assert!(matches!(
        parse_config(mismatch),
        Err(ConfigError::Invalid {
            field: "num_hidden_layers",
            ..
        })
    ));
    let unknown = r#"{"num_hidden_layers": 1, "num_attention_heads": 4, "head_dim": 8,
                      "layer_types": ["chunked_attention"]}"#;
    assert!(matches!(
        parse_config(unknown),
        Err(ConfigError::Unsupported(_))
    ));
    let no_window = r#"{"num_hidden_layers": 1, "num_attention_heads": 4, "head_dim": 8,
                        "layer_types": ["sliding_attention"]}"#;
    assert_eq!(
        parse_config(no_window),
        Err(ConfigError::Missing("sliding_window"))
    );
    let negative = r#"{"num_hidden_layers": -1, "num_attention_heads": 4}"#;
    assert!(matches!(
        parse_config(negative),
        Err(ConfigError::Invalid {
            field: "num_hidden_layers",
            ..
        })
    ));
}
