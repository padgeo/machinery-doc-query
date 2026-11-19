// Shared by the integration tests. Every file in tests/ is its own crate, so a
// helper that one of them does not use would otherwise warn as dead code.
#![allow(dead_code)]

use std::{
    process::Command,
    sync::{OnceLock, mpsc},
    thread,
    time::Duration,
};

use qdrant_client::Qdrant;
use testcontainers::{
    ContainerAsync, GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

const GRPC_PORT: u16 = 6334;
const CONTAINER_NAME: &str = "machinery-doc-query-test-qdrant";

/// The gRPC address (the form `QdrantStore::connect` takes) of a throwaway Qdrant in
/// Docker. It is started the first time a test asks and shared by every test of the
/// binary, so each test has to use collections of its own.
///
/// One container instead of one per test, because starting a dozen of them made
/// Docker stall now and then: a container that came up without its port published,
/// which left the test waiting forever. A single start, with a timeout and a retry,
/// is far less exposed.
///
/// The container is not removed when the tests end. The next run replaces it, and
/// `docker rm -f machinery-doc-query-test-qdrant` removes it by hand.
pub fn qdrant_url() -> &'static str {
    static URL: OnceLock<String> = OnceLock::new();
    URL.get_or_init(|| {
        let (sender, receiver) = mpsc::channel();
        // A thread and a runtime of their own: the container has to outlive the
        // runtime of whichever test happens to ask first.
        thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a runtime for the Qdrant container");
            let (url, container) = runtime.block_on(start_with_retries());
            sender.send(url).expect("the test that asked is waiting");
            // Keep the container, and the runtime that started it, until the process ends.
            let _alive = (container, runtime);
            loop {
                thread::park();
            }
        });
        receiver
            .recv_timeout(Duration::from_secs(150))
            .expect("Qdrant did not start: is Docker running?")
    })
}

/// A plain client of our own, so a test can look at what is stored without going
/// through the code under test.
pub fn client() -> Qdrant {
    Qdrant::from_url(qdrant_url())
        .build()
        .expect("a Qdrant client")
}

async fn start_with_retries() -> (String, ContainerAsync<GenericImage>) {
    for attempt in 1..=3 {
        // A container from an earlier run, or from the attempt that just failed, still
        // holds the name. Best effort: there may be none, and the CLI may be missing.
        let _ = Command::new("docker")
            .args(["rm", "-f", CONTAINER_NAME])
            .output();
        match tokio::time::timeout(Duration::from_secs(40), start_once()).await {
            Ok(Ok(started)) => return started,
            Ok(Err(why)) => eprintln!("Qdrant start, attempt {attempt}: {why}"),
            Err(_) => eprintln!("Qdrant start, attempt {attempt}: gave up after 40 s"),
        }
    }
    panic!("Qdrant did not start after 3 attempts");
}

async fn start_once() -> Result<(String, ContainerAsync<GenericImage>), String> {
    // Same image as docker-compose.yml, so the tests run against what the app uses.
    let container = GenericImage::new("qdrant/qdrant", "v1.19.0")
        .with_exposed_port(GRPC_PORT.tcp())
        .with_wait_for(WaitFor::message_on_stdout("Qdrant gRPC listening on 6334"))
        .with_container_name(CONTAINER_NAME)
        .start()
        .await
        .map_err(|e| format!("could not start the container: {e}"))?;
    let host = container
        .get_host()
        .await
        .map_err(|e| format!("no container host: {e}"))?;

    // Docker can report the published port a moment after the container has logged
    // that it is up, so ask again instead of failing on the first answer.
    let mut port = None;
    for _ in 0..50 {
        if let Ok(found) = container.get_host_port_ipv4(GRPC_PORT.tcp()).await {
            port = Some(found);
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let port = port.ok_or("Docker did not publish the gRPC port")?;
    let url = format!("http://{host}:{port}");

    // The log line only says the port is bound; wait until the server answers.
    let client = Qdrant::from_url(&url).build().map_err(|e| e.to_string())?;
    for _ in 0..100 {
        if client.health_check().await.is_ok() {
            return Ok((url, container));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("Qdrant did not answer within 10 s".to_string())
}
