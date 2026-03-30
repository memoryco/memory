//! Diesel schema definitions for memory storage
//!
//! Note: Manually maintained to ensure correct types.
//! SQLite doesn't have strict types, so we specify what Rust expects.

diesel::table! {
    associations (from_id, to_id) {
        from_id -> Text,
        to_id -> Text,
        weight -> Float,
        created_at -> BigInt,
        last_activated -> BigInt,
        co_activation_count -> BigInt,
        ordinal -> Nullable<Integer>,
    }
}

diesel::table! {
    config (id) {
        id -> Integer,
        data -> Text,
    }
}

diesel::table! {
    memories (id) {
        id -> Text,
        content -> Text,
        energy -> Float,
        state -> Text,
        confidence -> Float,
        created_at -> BigInt,
        last_accessed -> BigInt,
        access_count -> BigInt,
        tags -> Text,
        embedding -> Nullable<Binary>,
    }
}

diesel::table! {
    identity (id) {
        id -> Integer,
        data -> Text,
    }
}

diesel::table! {
    metadata (key) {
        key -> Text,
        value -> Text,
    }
}

diesel::table! {
    access_log (id) {
        id -> Integer,
        timestamp -> BigInt,
        query_text -> Text,
        result_ids -> Text,
        recalled_ids -> Text,
    }
}
