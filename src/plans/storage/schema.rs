//! Diesel schema for plans database
//!
//! This defines the table structure for plans.db

diesel::table! {
    plans (id) {
        id -> Text,
        description -> Text,
        created_at -> BigInt,
    }
}

diesel::table! {
    steps (plan_id, step) {
        plan_id -> Text,
        step -> Integer,
        description -> Text,
        completed -> Integer,
    }
}

diesel::allow_tables_to_appear_in_same_query!(plans, steps);
