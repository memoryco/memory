//! Diesel schema for identity items table
//!
//! This defines the flat table structure for identity.db
//! All identity types (values, preferences, antipatterns, etc.) are stored
//! in one table with a type discriminator.

diesel::table! {
    identity_items (id) {
        id -> Text,
        item_type -> Text,
        content -> Text,
        secondary -> Nullable<Text>,
        tertiary -> Nullable<Text>,
        category -> Nullable<Text>,
        created_at -> BigInt,
    }
}
