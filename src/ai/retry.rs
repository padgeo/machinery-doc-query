use std::time::Duration;

use tokio::time::sleep;

/// How often `retry` tries an operation and how long it waits in between. The wait starts
/// at `base_delay` and doubles up to `max_delay`; `max_attempts` counts the first try, so 1
/// means no retries.
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(1),
        }
    }
}

pub async fn retry<T, E, Fut>(
    policy: &RetryPolicy,
    is_transient: impl Fn(&E) -> bool,
    mut op: impl FnMut() -> Fut,
) -> Result<T, E>
where
    Fut: Future<Output = Result<T, E>>,
{
    let max_attempts = policy.max_attempts.max(1);
    let mut delay = policy.base_delay;
    let mut attempt = 0;

    loop {
        attempt += 1;
        match op().await {
            Ok(res) => return Ok(res),
            Err(e) if attempt < max_attempts && is_transient(&e) => {
                sleep(delay).await;
                delay = delay.saturating_mul(2).min(policy.max_delay);
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::Instant,
    };

    use super::*;

    #[derive(Debug, PartialEq)]
    enum TestError {
        Transient(u32),
        Permanent(u32),
    }

    fn is_transient(error: &TestError) -> bool {
        matches!(error, TestError::Transient(_))
    }

    // Tiny delays keep the suite fast: these tests are about how often the
    // operation runs, not about how long the backoff is.
    fn fast_policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(4),
        }
    }

    // Runs `retry` with an operation that answers every call with the next
    // scripted outcome. Asking for more calls than scripted fails the test,
    // which is how "retried too often" gets caught. Returns the final result
    // and how many times the operation ran.
    async fn run_script(
        policy: &RetryPolicy,
        script: Vec<Result<&'static str, TestError>>,
    ) -> (Result<&'static str, TestError>, usize) {
        let script = Mutex::new(script.into_iter());
        let calls = AtomicUsize::new(0);

        let result = retry(policy, is_transient, || {
            calls.fetch_add(1, Ordering::SeqCst);
            let outcome = script
                .lock()
                .unwrap()
                .next()
                .expect("the operation ran more often than the script allows");
            async move { outcome }
        })
        .await;

        (result, calls.load(Ordering::SeqCst))
    }

    #[tokio::test]
    async fn success_on_the_first_attempt_runs_the_operation_once() {
        let (result, calls) = run_script(&fast_policy(3), vec![Ok("done")]).await;

        assert_eq!(result, Ok("done"));
        assert_eq!(calls, 1);
    }

    #[tokio::test]
    async fn transient_errors_are_retried_until_the_operation_succeeds() {
        let script = vec![
            Err(TestError::Transient(1)),
            Err(TestError::Transient(2)),
            Ok("done"),
        ];

        let (result, calls) = run_script(&fast_policy(5), script).await;

        assert_eq!(result, Ok("done"));
        assert_eq!(calls, 3);
    }

    // Boundary: the last allowed attempt is still a real attempt.
    #[tokio::test]
    async fn success_on_the_final_allowed_attempt_is_returned() {
        let script = vec![
            Err(TestError::Transient(1)),
            Err(TestError::Transient(2)),
            Ok("done"),
        ];

        let (result, calls) = run_script(&fast_policy(3), script).await;

        assert_eq!(result, Ok("done"));
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn permanent_errors_are_not_retried() {
        let (result, calls) = run_script(&fast_policy(5), vec![Err(TestError::Permanent(1))]).await;

        assert_eq!(result, Err(TestError::Permanent(1)));
        assert_eq!(calls, 1);
    }

    #[tokio::test]
    async fn a_permanent_error_after_transient_ones_stops_the_retrying() {
        let script = vec![Err(TestError::Transient(1)), Err(TestError::Permanent(2))];

        let (result, calls) = run_script(&fast_policy(5), script).await;

        assert_eq!(result, Err(TestError::Permanent(2)));
        assert_eq!(calls, 2);
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts_and_returns_the_last_error() {
        let script = (1..=3).map(|n| Err(TestError::Transient(n))).collect();

        let (result, calls) = run_script(&fast_policy(3), script).await;

        assert_eq!(result, Err(TestError::Transient(3)));
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn one_attempt_means_no_retries() {
        let (result, calls) = run_script(&fast_policy(1), vec![Err(TestError::Transient(1))]).await;

        assert_eq!(result, Err(TestError::Transient(1)));
        assert_eq!(calls, 1);
    }

    // 0 attempts is a misconfiguration; the sane reading is "try once".
    #[tokio::test]
    async fn zero_max_attempts_still_tries_once() {
        let (result, calls) = run_script(&fast_policy(0), vec![Err(TestError::Transient(1))]).await;

        assert_eq!(result, Err(TestError::Transient(1)));
        assert_eq!(calls, 1);
    }

    // Lower bound only (a sleep never returns early), so this cannot flake.
    #[tokio::test]
    async fn waits_between_attempts() {
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(20),
            max_delay: Duration::from_millis(200),
        };
        let script = (1..=3).map(|n| Err(TestError::Transient(n))).collect();

        let started = Instant::now();
        let (_, calls) = run_script(&policy, script).await;

        assert_eq!(calls, 3);
        // 3 attempts = 2 waits, each at least the base delay
        assert!(started.elapsed() >= Duration::from_millis(40));
    }
}
