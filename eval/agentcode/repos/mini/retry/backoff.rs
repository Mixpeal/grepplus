pub fn retry_with_backoff(max_attempts: u32) {
    for attempt in 0..max_attempts {
        if try_once(attempt) {
            return;
        }
    }
}

fn try_once(_attempt: u32) -> bool {
    false
}
